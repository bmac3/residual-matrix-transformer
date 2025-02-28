import equinox as eqx
import haliax as hax
from jax import random as jr
from jaxtyping import PRNGKeyArray

from .attention import MHABlock
from .config import TransformerConfig
from .mlp import MLPBlock
from src.nn import AbstractSowableModule, Embedding, Linear, RMSNorm


class TransformerLayer(AbstractSowableModule):
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
            config: TransformerConfig,
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


class Transformer(AbstractSowableModule):
    embedding: Embedding
    transformer_stack: hax.nn.Stacked[TransformerLayer]
    norm: RMSNorm
    unembedding: Linear
    Layer: hax.Axis = eqx.field(static=True)
    intermediates_cache: dict = eqx.field(default_factory=dict)

    def forward(
            self, 
            input_ids: hax.NamedArray,
            mask: hax.NamedArray
        ):
        x = self.embedding(input_ids)
        x = self.sow('embedding', x)
        x = self.sow_fn(self.transformer_stack.scan)(x, mask, hax.arange(self.Layer))
        x = self.norm(x)
        return self.unembedding(x)

    @classmethod
    def init(
            cls,
            config: TransformerConfig,
            key: PRNGKeyArray
        ):
        e_key, stack_key, ue_key = jr.split(key, 3)
        return cls(
            embedding=Embedding.init(
                config=config,
                Vocab=config.Vocab,
                Embed=config.Embed,
                key=e_key
            ),
            transformer_stack=hax.nn.Stacked.init(
                config.Layer.axis, 
                TransformerLayer,
                gradient_checkpointing=True
            )(
                config=config,
                key=jr.split(stack_key, config.Layer.size),
            ),
            norm=RMSNorm.init(
                config=config,
                axis=config.Embed
            ),
            unembedding=Linear.init(
                config=config,
                In=config.Embed,
                Out=config.Vocab,
                key=ue_key,
                apply_wd=False
            ),
            Layer=config.Layer.axis
        )
