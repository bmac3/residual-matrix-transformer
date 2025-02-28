import math
from typing import ClassVar

import equinox as eqx
import haliax as hax
from jax import random as jr
from jaxtyping import PRNGKeyArray

from .config import RMTConfig
from src.nn import AbstractResidualBlock, AbstractSowableModule, Linear, RMSNorm


class MLP(AbstractSowableModule):
    norm: RMSNorm
    retrieval_linear: Linear
    W1: Linear
    W2: Linear
    storage_linear: Linear
    intermediates_cache: dict = eqx.field(default_factory=dict)

    def forward(self, x: hax.NamedArray):
        x = self.norm(x)
        x = self.retrieval_linear(x)
        x = self.sow('mlp_in', x)
        x = self.W1(x)
        x = self.sow('pre_activations', x)
        x = hax.nn.gelu(x)
        x = self.sow('activations', x)
        x = self.W2(x)
        x = self.sow('mlp_out', x)
        x = self.storage_linear(x)
        return x

    @classmethod
    def init(
            cls,
            config: RMTConfig,
            key: PRNGKeyArray,
        ):
        r_key, w1_key, w2_key, s_key = jr.split(key, 4)
        return cls(
            norm=RMSNorm.init(
                config=config,
                axis=(
                    config.ResVal,
                    config.ResKey
                )
            ),
            retrieval_linear=Linear.init(
                config=config,
                In=config.ResKey,
                Out=config.Rank,
                key=r_key,
            ),
            W1=Linear.init(
                config=config,
                In=(
                    config.ResVal,
                    config.Rank
                ),
                Out=config.Neuron,
                key=w1_key,
            ),
            W2=Linear.init(
                config=config,
                In=config.Neuron,
                Out=(
                    config.ResVal,
                    config.Rank
                ),
                key=w2_key,
            ),
            storage_linear=Linear.init(
                config=config,
                In=config.Rank,
                Out=config.ResKey,
                key=s_key,
            ).scale(1/math.sqrt(2 * config.Layer.size)),
        )


class MLPBlock(AbstractResidualBlock):
    module: MLP
    module_cls: ClassVar[type] = MLP
    intermediates_cache: dict = eqx.field(default_factory=dict)
