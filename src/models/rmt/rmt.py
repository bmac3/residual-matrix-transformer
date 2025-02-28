import equinox as eqx
import haliax as hax
from jax import random as jr
from jaxtyping import PRNGKeyArray

from .attention import MHABlock
from .config import RMTConfig
from .mlp import MLPBlock
from src.nn import AbstractSowableModule, Embedding, Linear, RMSNorm


class RMTLayer(AbstractSowableModule):
    attention: MHABlock
    mlp: MLPBlock
    intermediates_cache: dict = eqx.field(default_factory=dict)

    def forward(
            self,
            x: hax.NamedArray,
            mask: hax.NamedArray,
            layer_idx: int
        ):
        x = self.sow_fn(self.attention)(x, mask, layer_idx)
        x = self.sow_fn(self.mlp)(x)
        return x
    
    @classmethod
    def init(
            cls,
            config: RMTConfig,
            key: PRNGKeyArray
        ):
        mha_key, mlp_key = jr.split(key)
        return cls(
            attention=MHABlock.init(
                config=config,
                key=mha_key
            ),
            mlp=MLPBlock.init(
                config=config,
                key=mlp_key
            )
        )


class RMT(AbstractSowableModule):
    embedding: Embedding
    storage_linear: Linear
    rmt_stack: hax.nn.Stacked[RMTLayer]
    norm: RMSNorm
    retrieval_linear: Linear
    unembedding: Linear
    Layer: hax.Axis = eqx.field(static=True)
    intermediates_cache: dict = eqx.field(default_factory=dict)

    def forward(
            self, 
            input_ids: hax.NamedArray,
            mask: hax.NamedArray
        ):
        x = self.embedding(input_ids)
        x = self.storage_linear(x)
        x = self.sow('embedding', x)
        x = self.sow_fn(self.rmt_stack.scan)(x, mask, hax.arange(self.Layer))
        x = self.norm(x)
        x = self.retrieval_linear(x)
        return self.unembedding(x)

    @classmethod
    def init(
            cls,
            config: RMTConfig,
            key: PRNGKeyArray
        ):
        e_key, s_key, stack_key, r_key, ue_key = jr.split(key, 5)
        return cls(
            embedding=Embedding.init(
                config=config,
                Vocab=config.Vocab,
                Embed=(
                    config.ResVal,
                    config.Rank
                ),
                key=e_key
            ),
            storage_linear=Linear.init(
                config=config,
                In=config.Rank,
                Out=config.ResKey,
                key=s_key
            ),
            rmt_stack=hax.nn.Stacked.init(
                config.Layer.axis, 
                RMTLayer,
                gradient_checkpointing=True
            )(
                config=config,
                key=jr.split(stack_key, config.Layer.size),
            ),
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
                key=r_key
            ),
            unembedding=Linear.init(
                config=config,
                In=(
                    config.ResVal,
                    config.Rank
                ),
                Out=config.Vocab,
                key=ue_key,
                apply_wd=False
            ),
            Layer=config.Layer.axis
        )
