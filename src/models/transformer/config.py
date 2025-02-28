from dataclasses import dataclass
from typing import Optional

from jax import numpy as jnp

from src.nn import AbstractConfig, AxisType, ParamAxis


@dataclass(frozen=True)
class TransformerConfig(AbstractConfig):
    Vocab: ParamAxis
    Embed: ParamAxis
    Layer: ParamAxis
    Head: ParamAxis
    HeadDim: ParamAxis
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
            embed_dim: int,
            n_layers: int,
            n_heads: int,
            head_dim: int,
            n_neurons: int,
            rmsnorm_eps: float = 1e-6,
            init_std: float = 0.02,
            input_multiplier: float = 1.0,
            output_multiplier: float = 1.0,
            attn_multiplier: float = 1.0,
            base_embed_dim: Optional[int] = None,
            base_n_heads: Optional[int] = None,
            base_head_dim: Optional[int] = None,
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
            Embed=ParamAxis(
                name='Embed', 
                size=embed_dim, 
                axis_type=AxisType.INFINITE, 
                base_fan_dim=base_embed_dim
            ),
            Layer=ParamAxis(
                name='Layer', 
                size=n_layers, 
                axis_type=AxisType.FINITE
            ),
            Head=ParamAxis(
                name='Head', 
                size=n_heads, 
                axis_type=AxisType.INFINITE, 
                base_fan_dim=base_n_heads
            ),
            HeadDim=ParamAxis(
                name='HeadDim', 
                size=head_dim, 
                axis_type=AxisType.INFINITE, 
                base_fan_dim=base_head_dim
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
