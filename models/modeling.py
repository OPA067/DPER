import os
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn

from models.local_feat_agg import LFA_Net
from models.prob_emb_rep import PER_Net
from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import Transformer as TransformerClip
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, MSE, KL, KL_Divergence

allgather = AllGather.apply
allgather2 = AllGather2.apply

class TIR_Model(nn.Module):
    def __init__(self, config):
        super(TIR_Model, self).__init__()

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

        self.clip.load_state_dict(state_dict, strict=False)
        self.apply(self.init_weights)

        embed_dim = state_dict["text_projection"].shape[1]
        self.LFA_Net_text = LFA_Net(1, embed_dim, embed_dim, embed_dim // 2)
        self.PER_Net_text = PER_Net(embed_dim, embed_dim, embed_dim // 2)

        self.LFA_Net_video = LFA_Net(1, embed_dim, embed_dim, embed_dim // 2)
        self.PER_Net_video = PER_Net(embed_dim, embed_dim, embed_dim // 2)

        self.loss_fct = CrossEn(config)
        self.loss_kl = KL_Divergence(config)

    def forward(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0):

        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])

        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        word_feat, sentence_feat, patch_feat, video_feat = self.get_text_video_feat(text_ids, text_mask, video, video_mask, shaped=True)

        if self.training:
            if torch.cuda.is_available():
                idx = allgather(idx, self.config)
                text_mask = allgather(text_mask, self.config)
                word_feat = allgather(word_feat, self.config)
                sentence_feat = allgather(sentence_feat, self.config)

                video_mask = allgather(video_mask, self.config)
                patch_feat = allgather(patch_feat, self.config)
                video_feat = allgather(video_feat, self.config)
                torch.distributed.barrier()

            idx = idx.view(-1, 1)
            idx_all = idx.t()
            pos_idx = torch.eq(idx, idx_all).float()
            logit_scale = self.clip.logit_scale.exp()
            loss = 0.

            # Similarity loss of sentence_feat & video_feat
            similarity = self.sim_matrix_training(sentence_feat, video_feat)
            sim_loss = self.loss_fct(similarity * logit_scale) + self.loss_fct(similarity.T * logit_scale)

            # Struct probabilistic distance
            prob_text = self.probabilistic_text(word_feat, sentence_feat)
            t_mu = prob_text['mu']
            t_sigma = prob_text['sigma']
            prob_video = self.probabilistic_video(patch_feat, video_feat)
            v_mu = prob_video['mu']
            v_sigma = prob_video['sigma']

            distance = (t_mu - v_mu) ** 2 + (t_sigma ** 2 + v_sigma ** 2)
            print("===", t_mu.shape, t_sigma.shape, v_mu.shape, v_sigma.shape)
            print("=========", similarity.shape, distance.shape)
            distance = - 1 * distance + 1
            dis_loss = -1 * similarity * torch.log(torch.sigmoid(distance)) - (1 - similarity) * torch.log(-1 * distance)

            kl_loss = self.loss_kl(t_mu, t_sigma) + self.loss_kl(v_mu, v_sigma)

            loss = loss + sim_loss + dis_loss + 1e-2 * kl_loss

            return loss
        else:
            return None

    def probabilistic_text(self, word_feat, sentence_feat):
        output = {}
        out = self.LFA_Net_text(word_feat, sentence_feat)
        uncertain_out = self.PER_Net_text(word_feat, sentence_feat, out)

        output['mu'] = uncertain_out['mu']
        output['sigma'] = uncertain_out['sigma']

        return output

    def probabilistic_video(self, patch_feat, video_feat):
        output = {}
        out = self.LFA_Net_video(patch_feat, video_feat)
        uncertain_out = self.PER_Net_video(patch_feat, video_feat, out)

        output['mu'] = uncertain_out['mu']
        output['sigma'] = uncertain_out['sigma']

        return output

    def sim_matrix_training(self, text_feat, video_feat):

        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)

        sims = torch.mm(text_feat, video_feat.t())

        return sims

    def get_text_feat(self, text_ids, text_mask, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])

        bs_pair = text_ids.size(0)
        sentence_feat, word_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask)
        sentence_feat, word_feat = sentence_feat.float(), word_feat.float()
        sentence_feat = sentence_feat.view(bs_pair, -1, sentence_feat.size(-1)).squeeze(1)
        word_feat = word_feat.view(bs_pair, -1, word_feat.size(-1)).squeeze(1)
        return word_feat, sentence_feat

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
        video_feat, patch_feat = self.clip.encode_image(video, return_hidden=True)
        video_feat = video_feat.float()
        patch_feat = patch_feat.float()
        video_feat = video_feat.float().view(bs_pair, -1, video_feat.size(-1)).squeeze(1)
        patch_feat = patch_feat.float().view(bs_pair, -1, patch_feat.size(-1))

        return patch_feat, video_feat

    def get_text_video_feat(self, text_ids, text_mask, video, video_mask, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.shape) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        word_feat, sentence_feat = self.get_text_feat(text_ids, text_mask, shaped=True)
        patch_feat, video_feat = self.get_video_feat(video, video_mask, shaped=True)

        return word_feat, sentence_feat, patch_feat, video_feat


    def get_similarity_logits(self, word_feat, sentence_feat, patch_feat, video_feat, text_mask, video_mask, shaped=False):
        if shaped is False:
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        similarity = self.sim_matrix_training(sentence_feat, video_feat)

        prob_text = self.probabilistic_text(word_feat, sentence_feat)
        t_mu = prob_text['mu']
        t_sigma = prob_text['sigma']
        prob_video = self.probabilistic_video(patch_feat, video_feat)
        v_mu = prob_video['mu']
        v_sigma = prob_video['sigma']

        distance = (t_mu - v_mu) ** 2 + (t_sigma ** 2 + v_sigma ** 2)
        distance = - 1 * distance + 1

        return similarity

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

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