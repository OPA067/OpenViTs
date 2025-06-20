import torch
import torch.nn as nn
from models.cluster import PCM, Att_Block_Patch

class PatchBranch(nn.Module):
    def __init__(self, embed_dim, sample_ratio=0.5, num_heads=8, k=3):
        super().__init__()
        self.pcm = PCM(sample_ratio=sample_ratio, embed_dim=embed_dim, dim_out=embed_dim, k=k)
        self.att = Att_Block_Patch(dim=embed_dim, num_heads=num_heads)

    def forward(self, token_dict):
        token_dict = self.pcm(token_dict)
        token_dict = self.att(token_dict)
        return token_dict

class PatchProcessing(nn.Module):
    def __init__(self, embed_dim=512, sample_ratio=0.5, num_heads=8, k=3, num_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            PatchBranch(embed_dim, sample_ratio, num_heads, k)
            for _ in range(num_blocks)
        ])

    def forward(self, x_list):
        out_list = []

        for x in x_list:
            b, n, _ = x.shape
            idx_token = torch.arange(n, device=x.device)[None, :].repeat(b, 1)
            agg_weight = x.new_ones(b, n, 1)
            mask = torch.ones(b, n, device=x.device)

            token_dict = {
                'x': x,
                'token_num': n,
                'idx_token': idx_token,
                'agg_weight': agg_weight,
                'mask': mask
            }

            for block in self.blocks:
                token_dict = block(token_dict)

            out_list.append(token_dict['x'])

        return out_list
