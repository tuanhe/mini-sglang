from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import divide_even

from .base import BaseOP


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism."""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class LinearColParallelMerged(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [divide_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)


class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()

        GQA_ratio = divide_even(num_qo_heads, num_kv_heads)
        local_num_kv = divide_even(num_kv_heads, tp_info.size)
        full_isize = hidden_size
        full_osize = (GQA_ratio + 2) * num_kv_heads * head_dim
        local_isize = hidden_size
        local_osize = (GQA_ratio + 2) * local_num_kv * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)


class LinearOProj(_LinearTPImpl):
    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = divide_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()
        local_input_size = divide_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearColumnParallel(_LinearTPImpl):
    """列并行线性层：权重按输出维度拆分，前向输出通过all_gather聚合"""
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool = False,
        gather_output: bool = True  # 是否聚合输出（恢复完整维度）
    ):
        tp_info = get_tp_info()
        self.tp_size = tp_info.size
        self.gather_output = gather_output

        # 计算本地权重维度：输入维度不拆分，输出维度按TP大小均分
        local_input_size = input_size  # 列并行输入维度不拆分
        local_output_size = divide_even(output_size, self.tp_size)  # 输出维度拆分到每个TP进程

        # 初始化父类：记录全局维度与本地维度
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=local_input_size,
            local_osize=local_output_size,
            has_bias=has_bias
        )

        # 通信器：用于all_gather聚合输出
        self._comm = DistributedCommunicator()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：本地线性变换 + （可选）all_gather聚合输出"""
        # 1. 本地线性层计算：输入x形状[B, ..., input_size] → 输出y形状[B, ..., local_output_size]
        y = F.linear(x, self.weight, self.bias)

        # 2. 若开启聚合且TP大小>1，通过all_gather合并所有TP进程的输出维度
        if self.gather_output and self.tp_size > 1:
            # all_gather后形状：[B, ..., local_output_size * tp_size]（即full_osize）
            y = self._comm.all_gather(y, dim=-1)  # 按最后一维（输出维度）拼接

        return y
    
class QKVParallelLinear(LinearColumnParallel):
    """张量并行下的QKV合并投影层（继承列并行线性层）
    
    功能：将输入x通过一个合并的线性层投影为Q、K、V，同时支持张量并行（按输出维度拆分权重）
    输入：x [B, N, hidden_size]
    输出：qkv [B, N, 3×hidden_size]（若gather_output=True）或拆分后的局部QKV [B, N, 3×local_hidden_size]
    """
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = True,
    ):
        tp_info = get_tp_info()
        tp_size = tp_info.size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.num_heads = self.total_num_heads // tp_size
        # QKV合并后的总输出维度 = 3 × hidden_size（Q、K、V各占hidden_size）
        full_qkv_output_size = 3 * hidden_size
        
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = tp_size // self.total_num_kv_heads
        else:
            self.num_kv_heads = self.total_num_kv_heads // tp_size
            
        input_size = self.hidden_size
        output_size = ( (self.num_heads + 2 * self.num_kv_heads) * tp_size * self.head_size)
        
        # 调用父类LinearColumnParallel的初始化：
        # - 输入维度：hidden_size（与QKV输入一致）
        # - 输出维度：full_qkv_output_size（合并后的总输出维度）
        # 父类会自动将输出维度按TP大小拆分，得到local_output_size = (3×hidden_size) / tp_size
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            has_bias=bias,
            gather_output=False
        )
        
        # 记录QKV相关维度（用于后续拆分Q/K/V）
        self.hidden_size = hidden_size
        self.full_qkv_output_size = full_qkv_output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：合并投影→（可选聚合）→拆分Q/K/V
        
        Args:
            x: 输入特征 [B, N, hidden_size]（B: batch, N: token数）
        Returns:
            qkv: 合并的QKV投影结果 [B, N, 3×hidden_size]（若gather_output=True）
                 或 [B, N, 3×local_hidden_size]（若gather_output=False）
        """
        # Step 1: 调用父类LinearColumnParallel的forward，完成列并行线性变换+（可选）聚合
        qkv_merged = super().forward(x)  # 形状：[B, N, full_qkv_output_size]或[B, N, local_qkv_output_size]
        
        # Step 2: 将合并的QKV拆分为独立的Q、K、V（按最后一维拆分）
        # 拆分后Q/K/V的维度：每个为 [B, N, hidden_size]（若聚合）或 [B, N, local_hidden_size]（若未聚合）
        q, k, v = qkv_merged.split(self.hidden_size, dim=-1)
        
        return q, k, v

