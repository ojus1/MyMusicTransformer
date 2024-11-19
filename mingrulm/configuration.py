from transformers import PretrainedConfig

class MinGRUConfig(PretrainedConfig):
    model_type = "minGRULM"

    def __init__(
        self,
        num_tokens=1000,
        dim=512,
        depth=12,
        ff_mult=4,
        min_gru_expansion=1.5,
        conv_kernel_size=3,
        enable_conv=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.dim = dim
        self.depth = depth
        self.ff_mult = ff_mult
        self.min_gru_expansion = min_gru_expansion
        self.conv_kernel_size = conv_kernel_size
        self.enable_conv = enable_conv
