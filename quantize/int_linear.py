import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from quantize.quantizer import UniformAffineQuantizer
import numpy as np


class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        # self.weight = org_module.weight
        self.register_buffer('weight',org_module.weight)
        if org_module.bias is not None:
            # self.bias = org_module.bias
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.replace_weight_with_quantized = False
        self.is_weight_packed = False
        self.mem_packer = None
        # initialize quantizer
        self.i_cluster_counts = None
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        else:
            self.act_quantizer = None

        self.ignore_reconstruction = False
        self.disable_input_quant = disable_input_quant

        self.record_quant_input = False
        self.recorded_quant_input = None

    def set_ic_cluster_counts(self, counts, w_dim=1, a_dim=2):
        self.weight_quantizer.cluster_dim = w_dim
        self.weight_quantizer.cluster_counts = counts
        if a_dim is not None:
            self.act_quantizer.cluster_dim = a_dim
            self.act_quantizer.cluster_counts = counts

    def forward(self, input: torch.Tensor):
        if self.is_weight_packed:
            weight = self.mem_packer.unpack_tensor(self.weight)
            bias = self.bias
        elif self.replace_weight_with_quantized:
            weight = self.weight
            bias = self.bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:

            input = self.act_quantizer(input)
            if self.record_quant_input:
                # for debug
                self.recorded_quant_input = input

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
