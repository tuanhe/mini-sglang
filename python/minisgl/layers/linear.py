from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import divide_even

from .base import BaseOP


def divide(total: int, num_partitions: int) -> int:
    """均分整数（确保无余数，TP场景下通常要求可分）"""
    assert total % num_partitions == 0, f"{total} 无法被 {num_partitions} 均分"
    return total // num_partitions
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
    

class QKVParallelLinear(LinearColumnParallel):  # 继承你的列并行基类
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = True,
    ):
        tp_info = get_tp_info()
        self.tp_size = tp_info.size
        
        # 1. 初始化TP基础信息
        self.tp_rank = tp_info.rank
        
        # 2. 计算当前TP进程的Q/KV头数（核心拆分逻辑）
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        
        # Q头：按TP进程均分
        self.num_heads = divide(self.total_num_heads, self.tp_size)
        # KV头：若TP进程数≥总KV头数，则每个进程复制1个KV头；否则均分KV头
        if self.tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, self.tp_size)
        
        # 3. 计算当前进程的QKV输出维度（本地分片大小）
        self.q_shard_dim = self.num_heads * self.head_size       # 本地Q输出维度
        self.k_shard_dim = self.num_kv_heads * self.head_size    # 本地K输出维度
        self.v_shard_dim = self.num_kv_heads * self.head_size    # 本地V输出维度
        
        # 4. 父类（列并行）的总输出维度 = 本地QKV维度之和 × TP进程数？不！
        # 父类的output_size应为【本地QKV总维度】（因为列并行的输出维度是本地分片）
        # 注：原代码的output_size计算有误，正确逻辑是“本地输出维度=Q+K+V的本地分片之和”
        input_size = hidden_size
        local_output_size = self.q_shard_dim + self.k_shard_dim + self.v_shard_dim
        
        # 5. 初始化列并行基类（核心：输入不拆分，输出为本地QKV总维度）
        super().__init__(
            input_size=input_size,
            output_size=local_output_size,  # 父类需要的“全局输出维度”实际应为本地？不！
            # 修正：父类的output_size是【全局输出维度】，本地输出维度=output_size//tp_size
            # 因此正确的全局输出维度 = （Q总维度 + K总维度 + V总维度）
            # Q总维度 = total_num_heads × head_size
            # K/V总维度 = total_num_kv_heads × head_size
            output_size=(self.total_num_heads + 2*self.total_num_kv_heads)*self.head_size,
            has_bias=bias,
            gather_output=False,  # QKV不需要聚合（后续自行拆分）
        )
        
        # 6. 额外参数（与原逻辑对齐）
        self.bias = bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """前向传播：输出拆分后的Q/K/V（本地分片）"""
        # 1. 调用父类列并行计算：输入x → 本地QKV拼接的输出（shape: [B, ..., Q+K+V本地维度]）
        y, bias = super().forward(x)  # 假设父类forward返回(输出, 偏置)；若父类无bias返回则调整
        
        # 2. 拆分本地输出为Q/K/V分片
        q_shard = y[..., :self.q_shard_dim]               # [B, ..., num_heads×head_size]
        k_shard = y[..., self.q_shard_dim:self.q_shard_dim+self.k_shard_dim]  # [B, ..., num_kv_heads×head_size]
        v_shard = y[..., self.q_shard_dim+self.k_shard_dim:]  # [B, ..., num_kv_heads×head_size]
        
        # 3. 若需要跳过偏置加法，返回偏置；否则直接加偏置（与原逻辑对齐）
        if self.bias:
            q_shard += bias[..., :self.q_shard_dim]
            k_shard += bias[..., self.q_shard_dim:self.q_shard_dim+self.k_shard_dim]
            v_shard += bias[..., self.q_shard_dim+self.k_shard_dim:]
        return q_shard, k_shard, v_shard