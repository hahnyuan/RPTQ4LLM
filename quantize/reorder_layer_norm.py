import torch
import torch.nn as nn
from quantize.quantizer import UniformAffineQuantizer
from torch.utils.cpp_extension import load

USE_CUDA=False
if USE_CUDA:
    reorder_layer_norm_fp16 = load(
    'reorder_layernorm_fp16', ['./cuda/reorder_layernorm.cu'], 
    extra_cuda_cflags=['--use_fast_math'],
    extra_ldflags=["-L/usr/local/cuda/lib64/"])

class ReorderLayerNorm(nn.Module):
    def __init__(self, ori_layer_norm, act_quant_params=None) -> None:
        super().__init__()
        self.ori_layer_norm = ori_layer_norm
        self.use_act_quant = True
        self.out_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        if USE_CUDA:
            x_view=x.view(-1,x.size(-1))
            var,mean=torch.var_mean(x_view,0)
            out = x_view.new_empty(x_view.size())
            dst_index=torch.argsort(self.reorder_index)
            reorder_layer_norm_fp16.forward(x_view, out,mean,var,self.ori_layer_norm.weight,self.ori_layer_norm.bias,dst_index)
            out.view_as(x)
        else:
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
