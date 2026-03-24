import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchpack.utils.config import configs

from models.position_encoding import SinusoidalPositionalEmbedding


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                # layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([copy.deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        # self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1)), prob


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message, prob = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1)), prob
    

class QFormer(nn.Module):
    def __init__(self, num_query: int, feature_dim: int, ffn_dim: int, ffn_dropout: 0.5, num_block: int, use_scene_id: bool):
        super().__init__()
        # learnable queries
        self.queries = nn.Parameter(torch.randn(num_query, feature_dim))
        self.use_scene_id = configs.model.use_scene_id
        self.q_scene_encoder = SinusoidalPositionalEmbedding(d_model=feature_dim)
        # self attn
        self.sa_layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(num_block)])
        # cross attn
        self.ca_layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(num_block)])
        # feed forward
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_dim, feature_dim, bias=True),
            nn.Dropout(ffn_dropout)
        )
    
    def forward(self, x, scene_id=None):
        ''' x: B x N x C, scene_id: B x 1 '''
        assert len(self.sa_layers) == len(self.ca_layers)
        num_block = len(self.sa_layers)
        query = self.queries.unsqueeze(0).repeat(x.shape[0], 1, 1)  # B x num_query x feature_dim
        if scene_id is not None and configs.model.use_scene_id:
            query_scene = self.q_scene_encoder(scene_id).view(x.shape[0], -1).unsqueeze(1)  # B x 1 - > B x 1 x feature_dim
            query = torch.cat([query, query_scene], dim=1)
        query = query.permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        for i in range(num_block):
            # self attn
            delta_query, prob = self.sa_layers[i](query, query)
            query = query + delta_query
            # cross attn
            delta_x, prob = self.ca_layers[i](x, query)
            x = x + delta_x
            # ffn
            x = x + self.ffn(x.permute(0,2,1)).permute(0,2,1)
        x = x.permute(0, 2, 1)
        return x


class Promptor(nn.Module):
    def __init__(self, prompt_dim, use_scene_id) -> None:
        super().__init__()
        self.mlp_in = nn.Sequential(
                nn.Linear(3, 128),
                nn.ReLU(),
                nn.Linear(128, 8)
            )
        num_block = configs.model.num_prompt_block
        self.qformer = QFormer(num_query=64, feature_dim=8, ffn_dim=64, ffn_dropout=0.5, num_block=num_block, use_scene_id=use_scene_id)
        self.mlp_out = nn.Sequential(
                nn.Linear(8, 64),
                nn.ReLU(),
                nn.Linear(64, prompt_dim)
            )
    
    def forward(self, batch):
        ''' x: B x N x 3 '''
        scene_id = batch['scene_id'] if 'scene_id' in batch else None
        return self.mlp_out(self.qformer(self.mlp_in(batch['cloud']), scene_id=scene_id))


class PromptorL2P(nn.Module):
    ''' Refer to L2P (CVPR'22) '''
    def __init__(self, query_dim, prompt_dim=256, out_dim=1, num_prompt=20, num_select=4) -> None:
        super().__init__()
        # L2P param
        self.num_select = num_select
        self.keys = nn.Parameter(torch.randn(num_prompt, query_dim))
        self.prompts = nn.Parameter(torch.randn(num_prompt, prompt_dim))  # num_prompt x prompt_dim
        self.use_query_func = False
        self.mlp_out = nn.Sequential(
            nn.Linear(prompt_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        self.training = False
        self.count = torch.zeros(num_prompt, dtype=torch.int64).cuda()  # for training, diversity selection
    
    def train(self, mode=True):
        super(PromptorL2P, self).train(mode)
        self.training = mode
    
    def eval(self):
        return self.train(False)
    
    def forward(self, batch, query):
        ''' query: # B x query_dim '''
        # get prompt
        dist = (1-torch.einsum('bq,nq->bn', F.normalize(query, dim=-1), F.normalize(self.keys, dim=-1))) / 2  # (B x query_dim, num_prompt x query_dim) -> B x num_prompt, cos distance
        
        frequency = torch.ones_like(self.count)
        if self.training and torch.sum(self.count).item() > 0:
            frequency = self.count / torch.sum(self.count)
        
        _, selected_indices = torch.topk(torch.mul(dist, frequency.view(1, -1)),
                                         k=self.num_select, dim=-1, largest=False)  # B x num_select, B x num_select
        selected_dist = [dist[i].index_select(dim=-1, index=selected_indices[i]) for i in range(selected_indices.shape[0])]
        selected_dist = torch.stack(selected_dist, dim=0)  # B x num_select
        score = F.softmax(1-selected_dist, dim=-1)  # B x num_select
        
        if self.training:
            indices = selected_indices.view(-1)
            for i in indices:
                self.count[i.item()] += 1
        
        selected_prompts = [self.prompts.index_select(dim=0, index=selected_indices[i]) for i in range(selected_indices.shape[0])]
        selected_prompts = torch.stack(selected_prompts, dim=0)  # B x num_select x prompt_dim
        avg_prompt = torch.mean(torch.mul(score.unsqueeze(-1), selected_prompts), dim=1)  # B x prompt_dim
        # encode prompt into pc
        x = self.mlp_out(avg_prompt).unsqueeze(1).repeat(1, batch['cloud'].shape[1], 1)  # B x N x out_dim
        return x, selected_dist


if __name__ == '__main__':
    # test
    x = torch.randn((2, 32, 5))
    input1 = torch.randn((2, 32, 10))
    input2 = torch.randn((2, 32, 20))
    ca_layer = AttentionalPropagation(32, 4)
    output1 = ca_layer(x, input1)
    output2 = ca_layer(x, input2)
    