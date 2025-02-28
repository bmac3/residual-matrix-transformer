import math
from typing import ClassVar

import equinox as eqx
import haliax as hax
from jax import random as jr
from jaxtyping import PRNGKeyArray

from .config import TransformerConfig
from src.nn import AbstractResidualBlock, AbstractSowableModule, Linear, RMSNorm


class MLP(AbstractSowableModule):
    norm: RMSNorm
    W1: Linear
    W2: Linear
    intermediates_cache: dict = eqx.field(default_factory=dict)

    def forward(self, x: hax.NamedArray):
        x = self.norm(x)
        x = self.sow('mlp_in', x)
        x = self.W1(x)
        x = self.sow('pre_activations', x)
        x = hax.nn.gelu(x)
        x = self.sow('activations', x)
        x = self.W2(x)
        x = self.sow('mlp_out', x)
        return x

    @classmethod
    def init(
            cls,
            config: TransformerConfig,
            key: PRNGKeyArray,
        ):
        w1_key, w2_key = jr.split(key)
        return cls(
            norm=RMSNorm.init(
                config=config,
                axis=config.Embed,
            ),
            W1=Linear.init(
                config=config,
                In=config.Embed,
                Out=config.Neuron,
                key=w1_key,
            ),
            W2=Linear.init(
                config=config,
                In=config.Neuron,
                Out=config.Embed,
                key=w2_key,
            ).scale(1/math.sqrt(2 * config.Layer.size))
        )


class MLPBlock(AbstractResidualBlock):
    module: MLP
    module_cls: ClassVar[type] = MLP
    intermediates_cache: dict = eqx.field(default_factory=dict)
