import os
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn

from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import Transformer as TransformerClip
from .mpp import PatchProcessing
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, KL

allgather = AllGather.apply
allgather2 = AllGather2.apply

class ResidualLinear(nn.Module):
    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(nn.Linear(d_int, d_int), nn.ReLU(inplace=True))

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x

class MyModel(nn.Module):
    def __init__(self, config):

        super(MyModel, self).__init__()

        self.config = config

        self.interaction = config.interaction
        self.agg_module = getattr(config, 'agg_module', 'meanP')
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        assert backbone in _PT_NAME
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])
        if os.path.exists(model_path):
            FileNotFoundError
        try:
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)

        if torch.cuda.is_available():
            convert_weights(self.clip)

        cross_config = SimpleNamespace(**{
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 2048,
            "max_position_embeddings": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "vocab_size": 512,
            "soft_t": 0.07,
        })
        cross_config.max_position_embeddings = context_length
        cross_config.hidden_size = transformer_width
        self.cross_config = cross_config

        if self.agg_module in ["seqLSTM", "seqTransf"]:
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
            if self.agg_module == "seqTransf":
                self.transformerClip = TransformerClip(width=transformer_width,
                                                       layers=config.num_hidden_layers,
                                                       heads=transformer_heads)
            if self.agg_module == "seqLSTM":
                self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                           batch_first=True, bidirectional=False, num_layers=1)
        self.apply(self.init_weights)
        self.clip.load_state_dict(state_dict, strict=False)

        new_state_dict = OrderedDict()
        if self.agg_module in ["seqLSTM", "seqTransf"]:
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in state_dict.items():
                    if key == "positional_embedding":
                        new_state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if self.agg_module in ["seqTransf"] and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        if num_layer < config.num_hidden_layers:
                            new_state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue

        self.loss_fct = CrossEn(config)
        self.embed_dim = state_dict["text_projection"].shape[1]
        self.PatchListProcessing = PatchProcessing(embed_dim=self.embed_dim, sample_ratio=0.5, num_heads=8, k=3, num_blocks=3)

        self.p_feat_w = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        self.f_feat_w = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        self.w_feat_w = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, text, text_mask, video, video_mask, idx=None, global_step=0):

        text_mask = text_mask.view(-1, text_mask.shape[-1])
        text = text.view(-1, text.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        s_feat, w_feat, f_feat_list, p_feat_list = self.get_text_video_feat(text, text_mask, video, video_mask, shaped=True)

        if self.training:
            if torch.cuda.is_available():
                idx = allgather(idx, self.config)
                text_mask = allgather(text_mask, self.config)
                s_feat = allgather(s_feat, self.config)
                w_feat = allgather(w_feat, self.config)
                video_mask = allgather(video_mask, self.config)
                f_feat_list = [allgather(x, self.config) for x in f_feat_list]
                p_feat_list = [allgather(x, self.config) for x in p_feat_list]
                torch.distributed.barrier()

            idx = idx.view(-1, 1)
            idx_all = idx.t()
            pos_idx = torch.eq(idx, idx_all).float()
            logit_scale = self.clip.logit_scale.exp()
            total_loss = 0.
            loss_wp, loss_sp, loss_sf = 0., 0., 0.

            p_feat_list = self.PatchListProcessing(p_feat_list)

            # w_feat & p_feat_list
            w_feat_w = self.w_feat_w(w_feat).squeeze(-1)
            for p_feat in p_feat_list:
                p_feat_w = self.p_feat_w(p_feat).squeeze(-1)
                sims_wp = torch.einsum('awd,bpd->abwp', [self.norm(w_feat), self.norm(p_feat)])
                w2p_logits, _ = sims_wp.max(dim=-1)
                w2p_logits = torch.einsum('abw,aw->ab', [w2p_logits, w_feat_w])
                p2w_logits, _ = sims_wp.max(dim=-2)
                p2w_logits = torch.einsum('abp,bp->ab', [p2w_logits, p_feat_w])
                sims_wp = (w2p_logits + p2w_logits) / 2.0
                loss_sims_wp = self.loss_fct(sims_wp * logit_scale) + self.loss_fct(sims_wp.T * logit_scale)
                loss_wp += loss_sims_wp

            # s_feat & p_feat_list
            for p_feat in p_feat_list:
                p_feat_w = self.p_feat_w(p_feat).squeeze(-1)
                sims_sp = torch.einsum('ad,bpd->abp', [self.norm(s_feat), self.norm(p_feat)])
                sims_sp = torch.einsum('abp,bp->ab', [sims_sp, p_feat_w])
                loss_sims_sp = self.loss_fct(sims_sp * logit_scale) + self.loss_fct(sims_sp.T * logit_scale)
                loss_sp += loss_sims_sp

            # s_feat & f_feat_list
            for f_feat in f_feat_list:
                f_feat_w = self.f_feat_w(f_feat).squeeze(-1)
                sims_sf = torch.einsum('ad,bfd->abf', [self.norm(s_feat), self.norm(f_feat)])
                sims_sf = torch.einsum('abf,bf->ab', [sims_sf, f_feat_w])
                loss_sims_sf = self.loss_fct(sims_sf * logit_scale) + self.loss_fct(sims_sf.T * logit_scale)
                loss_sf += loss_sims_sf

            total_loss = loss_wp + loss_sp + loss_sf

            return total_loss
        else:
            return None

    def sim_matrix_training(self, t_feat, v_feat):
        t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
        v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)

        sims = torch.mm(t_feat, v_feat.t())

        return sims

    def norm(self, feat):
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def get_text_feat(self, text_ids, text_mask, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])

        bs_pair = text_ids.size(0)
        s_feat, w_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask)
        s_feat = s_feat.float()
        s_feat = s_feat.view(bs_pair, -1, s_feat.size(-1))
        w_feat = w_feat.float()
        w_feat = w_feat.view(bs_pair, -1, w_feat.size(-1))
        return s_feat, w_feat

    def get_video_feat(self, video, video_mask, shaped=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.size()) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        bs_pair, n_v = video_mask.size()

        f_feat_list, p_feat_list = self.clip.encode_image(video, return_hidden=True)

        f_feat_list = [x.float() for x in f_feat_list]
        p_feat_list = [x.float() for x in p_feat_list]

        f_feat_list = [x.float().view(bs_pair, -1, x.size(-1)) for x in f_feat_list]
        p_feat_list = [x.float().view(bs_pair, -1, x.size(-1)) for x in p_feat_list]

        return f_feat_list, p_feat_list

    def get_text_video_feat(self, text, text_mask, video, video_mask, shaped=False):
        if shaped is False:
            text = text.view(-1, text.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.shape) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        s_feat, w_feat = self.get_text_feat(text, text_mask, shaped=True)
        f_feat_list, p_feat_list = self.get_video_feat(video, video_mask, shaped=True)

        return s_feat.squeeze(1), w_feat, f_feat_list, p_feat_list

    def get_similarity_logits(self, text_mask, s_feat, w_feat, video_mask, f_feat_list, p_feat_list, shaped=False):
        if shaped is False:
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        sims = 0.

        p_feat_list = self.PatchListProcessing(p_feat_list)
        # w_feat & p_feat_list
        w_feat_w = self.w_feat_w(w_feat).squeeze(-1)
        for p_feat in p_feat_list:
            p_feat_w = self.p_feat_w(p_feat).squeeze(-1)
            sims_wp = torch.einsum('awd,bpd->abwp', [self.norm(w_feat), self.norm(p_feat)])
            w2p_logits, _ = sims_wp.max(dim=-1)
            w2p_logits = torch.einsum('abw,aw->ab', [w2p_logits, w_feat_w])
            p2w_logits, _ = sims_wp.max(dim=-2)
            p2w_logits = torch.einsum('abp,bp->ab', [p2w_logits, p_feat_w])
            sims_wp = (w2p_logits + p2w_logits) / 2.0
            sims = sims + sims_wp

        # s_feat & p_feat_list
        for p_feat in p_feat_list:
            p_feat_w = self.p_feat_w(p_feat).squeeze(-1)
            sims_sp = torch.einsum('ad,bpd->abp', [self.norm(s_feat), self.norm(p_feat)])
            sims_sp = torch.einsum('abp,bp->ab', [sims_sp, p_feat_w])
            sims = sims + sims_sp

        # s_feat & f_feat_list
        for f_feat in f_feat_list:
            f_feat_w = self.f_feat_w(f_feat).squeeze(-1)
            sims_sf = torch.einsum('ad,bfd->abf', [self.norm(s_feat), self.norm(f_feat)])
            sims_sf = torch.einsum('abf,bf->ab', [sims_sf, f_feat_w])
            sims = sims + sims_sf

        return sims

