import torch
import torch.nn as nn


class MemoryPacker(nn.Module):
    def __init__(self, scale, round_zero_point, n_bits, pack_dim=0) -> None:
        super().__init__()
        self.scale = nn.Parameter(scale.half())
        self.round_zero_point = nn.Parameter(round_zero_point.half())
        self.qmin = -(2 ** (n_bits - 1))
        self.qmax = 2 ** (n_bits - 1) - 1
        self.n_bits = n_bits
        self.pack_dim = pack_dim
        print("MemPacker will introduce error now, accurate version is in developing")

    def quant(self, x, scale, round_zero_point, uint=False):
        x_int = (x / scale).round_()
        if uint:
            x_int = x_int.add_(round_zero_point - self.qmin)
            x_int = x_int.clamp_(0, self.qmax - self.qmin).to(torch.uint8)
            return x_int
        else:
            if round_zero_point is not None:
                x_int = x_int.add_(round_zero_point)
            x_int = x_int.clamp_(self.qmin, self.qmax).to(torch.int8)
            return x_int

    def dequant(self, x_int, scale, round_zero_point, uint=False):
        if not x_int.dtype == scale.dtype:
            x_int = x_int.to(scale.dtype)
        if uint:
            if round_zero_point is not None:
                x_int = x_int.sub_(round_zero_point - self.qmin)
            x_dequant = x_int.mul_(scale)
            return x_dequant
        else:
            if round_zero_point is not None:
                x_int = x_int.sub_(round_zero_point)
            x_dequant = x_int.mul_(scale)
            return x_dequant

    def pack_tensor(self, x, force_half=True):
        # if force_half and self.scale.dtype != torch.half:
        #     self.scale = self.scale.half()
        #     self.round_zero_point = self.round_zero_point.half()
        x_int = self.quant(x, self.scale, self.round_zero_point, uint=True)
        # x_int = x_int + (1 << (self.n_bits - 1))

        # debug
        # x_int_deq = self.dequant(x_int, self.scale, self.round_zero_point, uint=True)
        # if not (x == x_int_deq).all():
        #     error_max = (x_int_deq - x).abs().max()
        #     error_sum = (x_int_deq != x).sum()
        #     print("error_sum", error_sum, "error_max", error_max)
        # breakpoint()

        if self.n_bits == 4:
            if self.pack_dim == 1:
                x_pack = x_int[:, : x.size(0) // 2]
                x_pack |= x_int[:, x.size(0) // 2 :] << 4
            else:
                x_pack = x_int[: x.size(0) // 2]
                x_pack |= x_int[x.size(0) // 2 :] << 4
        else:
            raise NotImplementedError("Only 4 bits are supported.")
        # debug
        # _x_int=torch.cat([x_pack&0b00001111,(x_pack&0b11110000)>>4],0)
        # _x = self.dequant(_x_int.half(), self.scale, self.round_zero_point, uint=True)
        # breakpoint()
        return x_pack

    def unpack_tensor(self, x_pack):
        if self.n_bits == 4:
            x_int = torch.cat(
                [x_pack & 0b00001111, (x_pack & 0b11110000) >> 4], self.pack_dim
            )
            x = self.dequant(x_int, self.scale, self.round_zero_point, uint=True)
        else:
            raise NotImplementedError()

        return x
