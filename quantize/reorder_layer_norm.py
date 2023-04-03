import torch
import torch.nn as nn
from quantize.quantizer import UniformAffineQuantizer


class ReorderLayerNorm(nn.Module):
    def __init__(self, ori_layer_norm, act_quant_params=None) -> None:
        super().__init__()
        self.ori_layer_norm = ori_layer_norm
        self.use_act_quant = True
        self.out_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        out = self.ori_layer_norm.forward(x)
        if hasattr(self, "reorder_index"):
            if x.ndim == 3:
                out = torch.index_select(out, 2, self.reorder_index)
            elif x.ndim == 2:
                out = torch.index_select(out, 1, self.reorder_index)
        if self.use_act_quant:
            out = self.out_quantizer(out)
        return out

    def set_quant_state(self, use_weight_quant, use_act_quant):
        self.use_act_quant = use_act_quant
