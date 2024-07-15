import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的函数操作模块
import torch.distributed as dist  # 导入PyTorch的分布式模块

import transformers  # 导入transformers库
from transformers.models.qwen2 import Qwen2Model, Qwen2PreTrainedModel
from transformers.activations import gelu  # 导入gelu激活函数
from transformers.file_utils import (  # 导入transformers库的文件实用工具
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import ModelOutput, dataclass
from typing import Optional, Tuple, List

@dataclass
class CLOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    pos_last_hidden_state: torch.FloatTensor = None
    pos_pooler_output: torch.FloatTensor = None
    pos_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    pos_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    pos_past_key_values: Optional[List[torch.FloatTensor]] = None
    neg_last_hidden_state: torch.FloatTensor = None
    neg_pooler_output: torch.FloatTensor = None
    neg_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    neg_past_key_values: Optional[List[torch.FloatTensor]] = None
    neg_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class Qwen2PoolerForCL(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, config):
        super().__init__()
        self.pooler_type = config.pooler_type  # 初始化池化器类型
        assert self.pooler_type in ["eos", "eos_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type  # 确保池化器类型合法
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 初始化全连接层
        self.activation = nn.Tanh()  # 初始化tanh激活函数

    def apply_mlp(self, x):
        return self.activation(self.dense(x))

    # 定义前向传播函数，接收注意力掩码和模型输出作为输入
    def forward(self, outputs, attention_mask=None):
        # 获取最后一层隐藏状态
        last_hidden = outputs.last_hidden_state
        # 获取池化器输出
        # pooler_output = outputs.pooler_output
        # 获取所有隐藏状态
        hidden_states = outputs.hidden_states
        
        if attention_mask is None:
            bs, seqlen, _ = last_hidden.shape
            attention_mask = torch.ones([bs, seqlen], device=last_hidden.device, dtype=torch.int)
        attention_mask_3d = attention_mask.unsqueeze(-1)

        # 如果池化器类型是 'eos_before_pooler' 或 'eos'，则返回最后一层隐藏状态的最后一个位置的输出
        if self.pooler_type in ['eos_before_pooler', 'eos']:
            pooler_res = last_hidden[:, -1]
            if self.pooler_type == 'eos':
                pooler_res = self.apply_mlp(pooler_res)
        # 如果池化器类型是 'avg'，则计算加权平均值
        elif self.pooler_type == "avg":
            pooler_res = ((last_hidden * attention_mask_3d).sum(1) / attention_mask_3d.sum(1))
        # 如果池化器类型是 'avg_first_last'，则计算首尾两个隐藏状态的平均值
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooler_res = (first_hidden + last_hidden) / 2.0
            pooler_res = ((pooler_res * attention_mask_3d).sum(1) / attention_mask_3d.sum(1))
        # 如果池化器类型是 'avg_top2'，则计算倒数第二和最后一个隐藏状态的平均值
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooler_res = (last_hidden + second_last_hidden) / 2.0
            pooler_res = ((pooler_res * attention_mask_3d).sum(1) / attention_mask_3d.sum(1)) 
        # 如果池化器类型不在已知的类型中，则抛出未实现的错误
        else:
            raise NotImplementedError(f'pooler type cannot be {self.pooler_type}')
        return pooler_res
        
        
class Qwen2ForCL(Qwen2PreTrainedModel):
    # 在加载模型时需要忽略的键列表
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # 初始化方法，接收配置和其他模型参数
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # 使用给定的BERT配置初始化BERT模型，不添加池化层
        self.model = Qwen2Model(config)
        self.pooler = Qwen2PoolerForCL(config)
        self.temp = self.config.temp
        assert self.config.loss_func in ['ce', 'bce', 'logsumexp']

        # 调用 `cl_init` 函数进行其他自定义初始化
        self.post_init()
        
    def calc_loss(self, sim_mat, labels, ignore_idx=-100):
        if self.config.loss_func != 'ce':
            label_mask = labels != ignore_idx
            sim_mat = sim_mat[label_mask]
            labels = labels[label_mask]
            idcs = torch.arange(sim_mat.shape[0], device=sim_mat.device)
            pos_sim_vec = sim_mat[idcs, labels].unsqueeze(-1)
            mask = torch.nn.functional.one_hot(labels, sim_mat.shape[1]).to(bool)
            neg_sim_mat = torch.masked_fill(sim_mat, mask, -1e9)
        
        if self.config.loss_func == 'ce':
            loss = torch.nn.functional.cross_entropy(sim_mat, labels, ignore_index=ignore_idx)
        elif self.config.loss_func == 'bce':
            loss = - torch.nn.functional.logsigmoid(pos_sim_vec - neg_sim_mat).mean()
        elif self.config.loss_func == 'logsumexp':
            loss = torch.logsumexp(neg_sim_mat - pos_sim_vec, -1).mean()
        else:
            raise NotImplementedError(f'loss func cannot be {self.config.loss_func}')
        
        return loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        pos_input_ids=None,
        pos_attention_mask=None,
        pos_position_ids=None,
        pos_past_key_values=None,
        pos_inputs_embeds=None,
        neg_input_ids=None,
        neg_attention_mask=None,
        neg_position_ids=None,
        neg_past_key_values=None,
        neg_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        if pos_input_ids is not None:
            assert input_ids.shape[0] == pos_input_ids.shape[0]
        
        gpt_res = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        
        pooler_res = self.pooler(gpt_res, attention_mask)
        
        if pos_input_ids is None:
            # 不提供正面句子，仅仅执行嵌入提取
            return CLOutput(
                pooler_output=pooler_res,
                last_hidden_state=gpt_res.last_hidden_state,
                hidden_states=gpt_res.hidden_states if output_hidden_states else None,
                attentions=gpt_res.attentions,
                past_key_values=gpt_res.past_key_values,
            )
            
        
        pos_gpt_res = self.model(
            input_ids=pos_input_ids,
            attention_mask=pos_attention_mask,
            position_ids=pos_position_ids,
            past_key_values=pos_past_key_values,
            inputs_embeds=pos_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        pos_pooler_res = self.pooler(pos_gpt_res, pos_attention_mask)
        # print('pos_pooler_res', pos_pooler_res.shape)
        
        if neg_input_ids is None:
            # 没有提供负面句子：使用其他样本的正面句子当负面句子
            sim_mat = torch.cosine_similarity(
                pooler_res.unsqueeze(1), 
                pos_pooler_res.unsqueeze(0), -1
            ) / self.temp
            labels = torch.arange(0, sim_mat.shape[0], 
                dtype=torch.long, 
                device=sim_mat.device,
            )
            loss = self.calc_loss(sim_mat, labels)

            return CLOutput(
                loss=loss,
                logits=sim_mat,
                pooler_output=pooler_res,
                last_hidden_state=gpt_res.last_hidden_state,
                hidden_states=gpt_res.hidden_states if output_hidden_states else None,
                attentions=gpt_res.attentions,
                past_key_values=gpt_res.past_key_values,
                pos_pooler_output=pos_pooler_res,
                pos_last_hidden_state=pos_gpt_res.last_hidden_state,
                pos_hidden_states=pos_gpt_res.hidden_states if output_hidden_states else None,
                pos_attentions=pos_gpt_res.attentions,
                pos_past_key_values=pos_gpt_res.past_key_values,
            )
            
        
        neg_gpt_res = self.model(
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask,
            position_ids=neg_position_ids,
            past_key_values=neg_past_key_values,
            inputs_embeds=neg_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        neg_pooler_res = self.pooler(neg_gpt_res, neg_attention_mask)
        
        pos_sim_vec = torch.cosine_similarity(
            pooler_res, pos_pooler_res, -1
        ).unsqueeze(-1) / self.temp
        
        # print('pos_sim_vec', pos_sim_vec.shape)
        
        # 负面句子在所有样本间共享
        neg_sim_mat = torch.cosine_similarity(
            pooler_res.unsqueeze(1),
            neg_pooler_res.unsqueeze(0), -1
        ) / self.temp
        sim_mat = torch.cat([pos_sim_vec, neg_sim_mat], -1)
        labels = torch.zeros(sim_mat.shape[0], dtype=torch.long, device=sim_mat.device)
        loss = self.calc_loss(sim_mat, labels)
        return CLOutput(
            loss=loss,
            logits=sim_mat,
            pooler_output=pooler_res,
            last_hidden_state=gpt_res.last_hidden_state,
            hidden_states=gpt_res.hidden_states if output_hidden_states else None,
            attentions=gpt_res.attentions,
            past_key_values=gpt_res.past_key_values,
            pos_pooler_output=pos_pooler_res,
            pos_last_hidden_state=pos_gpt_res.last_hidden_state,
            pos_hidden_states=pos_gpt_res.hidden_states if output_hidden_states else None,
            pos_attentions=pos_gpt_res.attentions,
            pos_past_key_values=pos_gpt_res.past_key_values,
            neg_pooler_output=neg_pooler_res,
            neg_last_hidden_state=neg_gpt_res.last_hidden_state,
            neg_hidden_states=neg_gpt_res.hidden_states if output_hidden_states else None,
            neg_attentions=neg_gpt_res.attentions,
            neg_past_key_values=neg_gpt_res.past_key_values,
        )
        
    def forward_clsf(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        ignore_idx=-100,
        category_input_ids=None,
        category_attention_mask=None,
        category_position_ids=None,
        category_past_key_values=None,
        category_inputs_embeds=None,  
        use_cache=None,      
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,

    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        
        gpt_res = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        pooler_res = self.pooler(gpt_res, attention_mask)

        cate_gpt_res = self.model(
            input_ids=category_input_ids,
            attention_mask=category_attention_mask,
            past_key_values=category_past_key_values,
            position_ids=category_position_ids,
            inputs_embeds=category_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        cate_pooler_res = self.pooler(cate_gpt_res, category_attention_mask)

        sim_mat = torch.cosine_similarity(
            pooler_res.unsqueeze(1),
            cate_pooler_res.unsqueeze(0), -1
        ) / self.temp

        loss = self.calc_loss(sim_mat, labels, ignore_idx)
        return CLOutput(
            loss=loss,
            logits=sim_mat,
            pooler_output=pooler_res,
            last_hidden_state=gpt_res.last_hidden_state,
            hidden_states=gpt_res.hidden_states if output_hidden_states else None,
            attentions=gpt_res.attentions,
            past_key_values=gpt_res.past_key_values,
            pos_pooler_output=cate_pooler_res,
            pos_last_hidden_state=cate_gpt_res.last_hidden_state,
            pos_hidden_states=cate_gpt_res.hidden_states if output_hidden_states else None,
            pos_attentions=cate_gpt_res.attentions,
            pos_past_key_values=cate_gpt_res.past_key_values,
        )
            
