from transformers import PreTrainedModel
from mingrulm.configuration import MinGRUConfig
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from minGRU_pytorch.minGRU import minGRU

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1)

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

# conv

class CausalDepthWiseConv1d(Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size = kernel_size, groups = dim),
            nn.Conv1d(dim, dim, kernel_size = 1)
        )
    def forward(self, x):
        x = x.transpose(1, 2) # b n d -> b d n
        x = F.pad(x, (self.kernel_size - 1, 0), value = 0.)
        x = self.net(x)
        return x.transpose(1, 2) # b d n -> b n d

# main class

class minGRULM(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        ff_mult = 4,
        min_gru_expansion = 1.5,
        conv_kernel_size = 3,
        enable_conv = False
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                CausalDepthWiseConv1d(dim, conv_kernel_size) if enable_conv else None,
                RMSNorm(dim),
                minGRU(dim, expansion_factor = min_gru_expansion),
                RMSNorm(dim),
                FeedForward(dim, mult = ff_mult)
            ]))

        self.norm = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

        self.can_cache = not enable_conv

    def forward(
        self,
        x,
        return_loss = False,
        return_prev_hiddens = False,
        prev_hiddens = None
    ):

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        # handle previous hiddens, for recurrent decoding

        if exists(prev_hiddens):
            x = x[:, -1:]

        next_prev_hiddens = []
        prev_hiddens = iter(default(prev_hiddens, []))

        for conv, norm, mingru, ff_norm, ff in self.layers:

            # conv

            if exists(conv):
                assert not exists(prev_hiddens), 'caching not supported for conv version'
                x = conv(x) + x

            # min gru

            prev_hidden = next(prev_hiddens, None)

            min_gru_out, next_prev_hidden = mingru(
                norm(x),
                prev_hidden,
                return_next_prev_hidden = True
            )

            x = min_gru_out + x
            next_prev_hiddens.append(next_prev_hidden)

            # feedforward

            x = ff(ff_norm(x)) + x

        embed = self.norm(x)
        logits = self.to_logits(embed)

        if not return_loss:
            if not return_prev_hiddens:
                return logits

            return logits, next_prev_hiddens

        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels
        )

        return loss

class MinGRULM(PreTrainedModel):
    config_class = MinGRUConfig

    def __init__(self, config):
        super().__init__(config)
        self.token_emb = nn.Embedding(config.num_tokens, config.dim)

        self.layers = ModuleList([])
        for _ in range(config.depth):
            self.layers.append(ModuleList([
                CausalDepthWiseConv1d(config.dim, config.conv_kernel_size) if config.enable_conv else None,
                RMSNorm(config.dim),
                minGRU(config.dim, expansion_factor=config.min_gru_expansion),
                RMSNorm(config.dim),
                FeedForward(config.dim, mult=config.ff_mult)
            ]))

        self.norm = RMSNorm(config.dim)
        self.to_logits = nn.Linear(config.dim, config.num_tokens, bias=False)
        self.can_cache = not config.enable_conv

    def forward(
        self,
        input_ids=None,
        labels=None,
        attention_mask=None,
        return_loss=False,
        return_prev_hiddens=False,
        prev_hiddens=None,
    ):
        x = self.token_emb(input_ids)

        if labels is not None:
            x, labels = x[:, :-1], labels[:, 1:]

        if exists(prev_hiddens):
            x = x[:, -1:]

        next_prev_hiddens = []
        prev_hiddens = iter(default(prev_hiddens, []))

        for conv, norm, mingru, ff_norm, ff in self.layers:
            if exists(conv):
                assert not exists(prev_hiddens), 'Caching not supported for conv version'
                x = conv(x) + x

            prev_hidden = next(prev_hiddens, None)

            min_gru_out, next_prev_hidden = mingru(
                norm(x),
                prev_hidden,
                return_next_prev_hidden=True
            )

            x = min_gru_out + x
            next_prev_hiddens.append(next_prev_hidden)
            x = ff(ff_norm(x)) + x

        embed = self.norm(x)
        logits = self.to_logits(embed)

        if return_loss:
            loss = F.cross_entropy(logits.transpose(1, 2), labels)
            return loss

        if return_prev_hiddens:
            return logits, next_prev_hiddens

        return logits

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = MinGRUConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        self.config.save_pretrained(save_directory)
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")
