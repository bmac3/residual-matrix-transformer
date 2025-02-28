import math
from typing import ClassVar

import equinox as eqx
import haliax as hax
import jax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import PRNGKeyArray
import jmp

from .config import TransformerConfig
from src.nn import AbstractResidualBlock, AbstractSowableModule, AxisType, Linear, ParamAxis, RMSNorm, mixed_precision


class SHA(AbstractSowableModule):
    mup_scaling: float = eqx.field(static=True)
    Head: hax.Axis = eqx.field(static=True)
    policy: jmp.Policy = eqx.field(static=True)
    intermediates_cache: dict = eqx.field(default_factory=dict)

    def make_attn_bias(self, mask: hax.NamedArray):
        m = hax.geomspace(self.Head, start=2**(-8/self.Head.size), stop=2**(-8))
        bias = -(hax.cumsum(mask[{'KVPos': slice(None, None, -1)}], axis='KVPos')[{'KVPos': slice(None, None, -1)}] - 1)
        bias = bias.broadcast_axis(self.Head) * m
        big_neg = jnp.finfo(jnp.float32).min
        return hax.where(mask, bias, big_neg)

    @mixed_precision
    def forward(
            self,
            query: hax.NamedArray,
            key: hax.NamedArray,
            value: hax.NamedArray,
            mask: hax.NamedArray,
            layer_idx: int
        ):
        orig_axes = query.axes
        query, key, value = hax.tree_util.tree_map(lambda x: x.rearrange((..., 'Head', 'Pos', 'HeadDim')), (query, key, value))
        key, value = hax.tree_util.tree_map(lambda x: x.rename({'Pos': 'KVPos'}), (key, value))
        attn_logits = hax.dot(query, key, axis='HeadDim', precision=jax.lax.Precision.HIGHEST)
        # not quite sure about order of operations with alibi but it's not supposed to be scaled by sqrt(d_h)
        attn_logits = attn_logits * self.mup_scaling
        # track for coord checking here
        attn_logits = self.sow('attn_logits', attn_logits)
        attn_logits = (attn_logits / (layer_idx + 1)) + self.make_attn_bias(mask)
        attn_weights = hax.nn.softmax(attn_logits, axis='KVPos')
        ret = hax.dot(attn_weights, value, axis='KVPos')
        return ret.rearrange(orig_axes)

    @classmethod
    def init(
            cls,
            config: TransformerConfig
        ):
        return cls(
            mup_scaling=config.attn_multiplier * (math.sqrt(config.HeadDim.base_fan_dim)/config.HeadDim.fan_dim),
            Head=config.Head.axis,
            policy=jmp.Policy(
                compute_dtype=config.full_dtype,
                param_dtype=config.full_dtype,
                output_dtype=config.half_dtype,
            )
        )


class MHA(AbstractSowableModule):
    norm: RMSNorm
    qkv_linear: Linear
    heads: SHA
    out_linear: Linear
    intermediates_cache: dict = eqx.field(default_factory=dict)

    def forward(
            self,
            x: hax.NamedArray,
            mask: hax.NamedArray,
            layer_idx: int
        ):
        x = self.norm(x)
        x = self.sow('attn_in', x)
        q, k, v = self.qkv_linear(x).unbind('QKV')
        q = self.sow('query', q)
        k = self.sow('key', k)
        v = self.sow('value', v)
        attn_out = self.sow_fn(self.heads)(q, k, v, mask, layer_idx)
        ret = self.out_linear(attn_out)
        ret = self.sow('attn_out', ret)
        return ret

    @classmethod
    def init(
            cls,
            config: TransformerConfig,
            key: PRNGKeyArray
        ):
        qkv_key, out_key = jr.split(key)
        QKV = ParamAxis('QKV', 3, AxisType.FINITE)
        qkv_linear = Linear.init(
            config=config,
            In=config.Embed,
            Out=(
                QKV,
                config.Head,
                config.HeadDim
            ),
            key=qkv_key
        )
        # Appendix D.2 of mup paper
        if config.query_zero_init:
            qkv_linear.set(qkv_linear.param.weight.at[{'QKV': 0}].set(0.0))
        return cls(
            norm=RMSNorm.init(
                config=config,
                axis=config.Embed
            ),
            qkv_linear=qkv_linear,
            heads=SHA.init(config),
            out_linear=Linear.init(
                config=config,
                In=(
                    config.Head,
                    config.HeadDim
                ),
                Out=config.Embed,
                key=out_key,
            ).scale(1/math.sqrt(2 * config.Layer.size)),
        )


class MHABlock(AbstractResidualBlock):
    module: MHA
    module_cls: ClassVar[type] = MHA
    intermediates_cache: dict = eqx.field(default_factory=dict)
