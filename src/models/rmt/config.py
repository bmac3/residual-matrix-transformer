from dataclasses import dataclass
from typing import Optional

from jax import numpy as jnp

from src.nn import AbstractConfig, AxisType, ParamAxis


@dataclass(frozen=True)
class RMTConfig(AbstractConfig):
    Vocab: ParamAxis
    Layer: ParamAxis
    Rank: ParamAxis
    ResKey: ParamAxis
    ResVal: ParamAxis
    Neuron: ParamAxis
    rmsnorm_eps: float
    init_std: float
    input_multiplier: float
    output_multiplier: float
    attn_multiplier: float
    query_zero_init: bool
    output_zero_init: bool
    half_dtype: jnp.dtype
    full_dtype: jnp.dtype

    @classmethod
    def init(
            cls,
            vocab_size: int,
            rank: int,
            reskey_dim: int,
            resval_dim: int,
            n_layers: int,
            n_neurons: int,
            rmsnorm_eps: float = 1e-6,
            init_std: float = 0.02,
            input_multiplier: float = 1.0,
            output_multiplier: float = 1.0,
            attn_multiplier: float = 1.0,
            base_rank: Optional[int] = None,
            base_reskey_dim: Optional[int] = None,
            base_resval_dim: Optional[int] = None,
            base_n_neurons: Optional[int] = None,
            query_zero_init: Optional[bool] = True,
            output_zero_init: Optional[bool] = True,
            half_dtype: Optional[jnp.dtype] = jnp.bfloat16,
            full_dtype: Optional[jnp.dtype] = jnp.float32
        ):
        return cls(
            Vocab=ParamAxis(
                name='Vocab', 
                size=vocab_size, 
                axis_type=AxisType.FINITE, 
                fan_dim=1
            ),
            Layer=ParamAxis(
                name='Layer', 
                size=n_layers, 
                axis_type=AxisType.FINITE
            ),
            Rank=ParamAxis(
                name='Rank', 
                size=rank, 
                axis_type=AxisType.INFINITE, 
                base_fan_dim=base_rank
            ),
            ResKey=ParamAxis(
                name='ResKey', 
                size=reskey_dim, 
                axis_type=AxisType.INFINITE, 
                base_fan_dim=base_reskey_dim
            ),
            ResVal=ParamAxis(
                name='ResVal', 
                size=resval_dim, 
                axis_type=AxisType.INFINITE, 
                base_fan_dim=base_resval_dim
            ),
            Neuron=ParamAxis(
                name='Neuron', 
                size=n_neurons, 
                axis_type=AxisType.INFINITE, 
                base_fan_dim=base_n_neurons
            ),
            rmsnorm_eps=rmsnorm_eps,
            init_std=init_std,
            input_multiplier=input_multiplier,
            output_multiplier=output_multiplier,
            attn_multiplier=attn_multiplier,
            query_zero_init=query_zero_init,
            output_zero_init=output_zero_init,
            half_dtype=half_dtype,
            full_dtype=full_dtype
        )
