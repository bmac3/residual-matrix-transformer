from typing import ClassVar

import equinox as eqx
import haliax as hax
import jmp

from .config import TransformerConfig
from .transformer import Transformer
from src.nn import AbstractModel


class TransformerModel(AbstractModel):
    module: Transformer
    config: TransformerConfig = eqx.field(static=True)
    module_cls: ClassVar[type] = Transformer
    config_cls: ClassVar[type] = TransformerConfig

    def __call__(
            self,
            input_ids: hax.NamedArray,
            mask: hax.NamedArray, 
            capture_intermediates: bool = False
        ):
        ret, intermediates = self.module(input_ids, mask)
        if capture_intermediates:
            return ret, intermediates
        else:
            return ret
    
    @property
    def non_embedding_size(self):
        return self.size - self.module.embedding.size - self.module.unembedding.size
