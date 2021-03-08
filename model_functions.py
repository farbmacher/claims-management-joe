# =============================================================================
# Replication files for 
# „An Explainable Attention Network for Fraud Detection in Claims Management“, 
# Helmut Farbmacher, Leander Löw, Martin Spindler,
# Journal of Econometrics
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
import math
from constants_imports import *
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init,Module
from torch.nn.modules.transformer import TransformerEncoderLayer,TransformerEncoder

info_dict=load_pickle(DATA_PATH+"info_dict.pkl")

class simple_attention(torch.nn.Module):
    def __init__(self, input_dim=100,return_weights=True,activation="logistic"):
        super(simple_attention, self).__init__()
        self.key_extract = nn.Linear(input_dim, input_dim)
        self.value_extract = nn.Linear(input_dim, input_dim)
        self.query = Parameter(torch.Tensor(input_dim))
        self.soft = nn.Sigmoid()
        self.return_weights = return_weights
        self.reset_parameters()
        self.activation=activation

    def reset_parameters(self):
        nn.init.normal(self.query)

    def forward(self, embeds):
        value = self.value_extract(embeds)
        keys = self.key_extract(embeds)
        scores = torch.einsum('btj,j->bt', keys, self.query)
        if self.activation == "logistic":
            weights = logistic(scores)
        if self.activation == "softmax":
            weights = F.softmax(scores, dim=-1)
        aggregations = torch.einsum('btj,bt->bj', value, weights)
        if self.return_weights == True:
            return aggregations, weights
        else:
            return aggregations

class multi_attention(torch.nn.Module):
    def __init__(self, input_dim=100, key_dim=50, value_dim=50, nheads=1, activation="logistic",return_weights=True):
        super(multi_attention, self).__init__()
        self.key_extract = nn.Linear(input_dim, key_dim)
        self.value_extract = nn.Linear(input_dim, value_dim)
        self.query = Parameter(torch.Tensor(key_dim, nheads))
        self.soft = nn.Sigmoid()
        self.return_weights = return_weights
        self.reset_parameters()
        self.activation=activation
        self.nheads=nheads

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.query, a=math.sqrt(5))

    def forward(self, embeds):
        value = self.value_extract(embeds)
        keys = self.key_extract(embeds)
        scores = torch.einsum('btj,jk->btk', keys, self.query)
        weights = logistic(scores)
        aggregations = torch.einsum('btj,btk->bjk', value, weights)
        if self.nheads==1:
            aggregations=aggregations.squeeze(2)
        if self.return_weights == True:
            return aggregations, weights
        else:
            return aggregations

def logistic(x, c=1, a=20, b=np.e):
    return c / (1 + a * b ** (-x))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, ln_low=True,attention_activation="logistic",return_w=False):
        super().__init__()
        if ln_low== True:
            self.norm_1 = nn.LayerNorm(d_model)
            self.norm_2 = nn.LayerNorm(d_model)
        else:
            self.norm_1 = dummy()
            self.norm_2 = dummy()
        self.attn = MultiHeadAttention(heads, d_model,activation=attention_activation)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.return_w=return_w

    def forward(self, x):
        x1, atn = self.attn(x, x, x)
        x2 = self.norm_1(self.dropout_1(x1)+x)
        x3 = self.dropout_2(self.ff(x2))
        if self.return_w==True:
            return x3, atn
        else:
            return x3

def attention(q, k, v, d_k, mask=None, activation=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    output = torch.matmul(scores, v)
    return output, scores

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        d_ff = int(d_model * 2)
        self.linear_1 = nn.Linear(d_model, d_model)
        self.linear_2 = nn.Linear(d_model, d_model)
        self.ln=nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.ln(F.relu(self.linear_1(x))+x)
        x = self.linear_2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1,activation="logistic"):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        if dropout>0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop=dummy()
        self.out = nn.Linear(d_model, d_model)
        self.activation=activation

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores, w = attention(q, k, v, self.d_k, mask,self.activation)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)
        return output, w

class embed_categorical_layer(nn.Module):
    def __init__(self, ):
        super().__init__()
        embedding_dict = {}
        for variable_name, variable_max_lvl in info_dict['categorical_level_dict'].items():
            embedding_dict.update(
                {variable_name: nn.Embedding(variable_max_lvl, int(np.sqrt(variable_max_lvl)), padding_idx=0)})
        self.embedding_dict = nn.ModuleDict(embedding_dict)
    def forward(self, input_dict):
        embeddings_stacked = torch.cat([self.embedding_dict[x](y).squeeze(2) for x, y in input_dict.items()], 2)
        return embeddings_stacked

class norm_l_res_d(nn.Module):
    def __init__(self, input_dim, output_dim,ln_high, drop=0.2):
        super().__init__()
        self.l = nn.Linear(input_dim, output_dim)
        self.l2 = nn.Linear(output_dim, output_dim)
        if input_dim == output_dim:
            self.res = True
        else:
            self.res = False
        if ln_high==True:
            self.ln = nn.LayerNorm(output_dim)
        else:
            self.ln = dummy()
        if drop>0:
            self.drop = nn.Dropout(drop)
        else:
            self.drop=dummy()
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.drop(self.ln(self.relu(self.l(x))))

class dummy(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class MyLinear(Module):
    __constants__ = ['bias', 'in_features', 'out_features']
    def __init__(self, in_features, out_features, bias=True):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.constant_(self.bias, -3)

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class all_agg(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.atn = multi_attention(int(model_dim),int(model_dim*2),int(model_dim),nheads=1, return_weights=False)
        self.sum = lambda x: torch.sum(x,axis=1)
        self.max = lambda x: torch.max(x, axis=1)[0]
        self.mean = lambda x: torch.mean(x, axis=1)

    def forward(self, x):
        atn_ag=self.atn(x)
        sum_ag = self.sum(x)
        max_ag = self.max(x)
        mean_ag = self.mean(x)
        out=torch.cat([atn_ag,sum_ag,max_ag,mean_ag],axis=1)
        return out

class fraud_model(nn.Module):
    def __init__(self, model_dim=59, feature_method="selfa", heads=1, drop=0.1, n_layers_extract=2, n_layers_fc=2,
                 aggregation="attention",ln_high=True,ln_low=True):
        super().__init__()
        assert model_dim % heads == 0
        self.embedding_layers = embed_categorical_layer()
        total_dim = np.sum([int(np.sqrt(y)) for x, y in info_dict['categorical_level_dict'].items()]) + info_dict[
            "number_numerics"]
        self.init_linear = norm_l_res_d(int(total_dim), int(model_dim), drop)
        if feature_method == "selfa":
            self.extract = [EncoderLayer(int(model_dim), int(heads), drop,ln_low, attention_activation="logistic") for x in
                            range(n_layers_extract)]
            self.extract = nn.Sequential(*self.extract)
        if feature_method == "no":
            self.extract = dummy()
        if feature_method == "rnn":
            self.extract = nn.LSTM(input_size=int(model_dim), hidden_size=int(model_dim / 2),
                               num_layers=int(n_layers_extract), batch_first=True, bidirectional=True)
        if aggregation == "attention":
            self.agg = multi_attention(int(model_dim),int(model_dim*2),int(model_dim),nheads=1, return_weights=False)
        if aggregation == "sum":
            self.agg = lambda x: torch.sum(x,axis=1)
        if aggregation == "all":
            self.agg=all_agg(model_dim)
        if aggregation == "all":
            self.fl_features=[]
            for x in range(n_layers_fc):
                if x == 0:
                    self.fl_features.append(norm_l_res_d(int(model_dim*4), int(model_dim), drop, ln_high))
                else:
                    self.fl_features.append(norm_l_res_d(int(model_dim ), int(model_dim), drop, ln_high))
            self.fl_features = nn.Sequential(*self.fl_features)
            self.bnorm = nn.BatchNorm1d(int(model_dim*4))
        else:
            self.fl_features = [norm_l_res_d(int(model_dim), int(model_dim), drop,ln_high) for x in range(n_layers_fc)]
            self.fl_features = nn.Sequential(*self.fl_features)
            self.bnorm = nn.BatchNorm1d(int(model_dim))
        self.fc_final = MyLinear(int(model_dim), 1)
        self.sig=nn.Sigmoid()

    def forward(self, categorical_batch, numeric_batch,return_hidden=False):
        categorical_embeded = self.embedding_layers(categorical_batch)
        numeric_categorical_combined = torch.cat([categorical_embeded, numeric_batch], 2)
        combined_features = self.init_linear(numeric_categorical_combined)
        features = self.extract(combined_features)
        fixed_length = self.agg(features)
        fixed_length=self.bnorm(fixed_length)
        final_features = self.fl_features(fixed_length)
        prediction = self.sig(self.fc_final(final_features))
        if return_hidden == False:
            return prediction
        else:
            return prediction,fixed_length
        