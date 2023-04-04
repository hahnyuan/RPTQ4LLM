from torch.utils.cpp_extension import load
import torch
reorder_layer_norm_fp16 = load(
    'reorder_layernorm_fp16', ['reorder_layernorm.cu'], 
    extra_cuda_cflags=['--use_fast_math'],
    extra_ldflags=["-L/usr/local/cuda/lib64/"])

torch.set_default_dtype(torch.float16)

def add_cuda_op(x, y,w,b):
    assert x.size() == y.size()
    output = x.new_empty(x.size())
    index=torch.arange(0,32,1,device='cuda').long()
    
    index[0]=1
    index[1]=31
    index[31]=2
    dst_index=torch.argsort(index)
    var,mean=torch.var_mean(x,0)
    reorder_layer_norm_fp16.forward(x, output,mean,var,w,b,dst_index)
    return output

bias=torch.arange(0,32,1,device='cuda').half()
out=add_cuda_op(torch.ones([32,32],device='cuda'),torch.ones([32,32],device='cuda'),torch.ones([32],device='cuda'),bias)
print(out)