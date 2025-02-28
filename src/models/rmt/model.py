from typing import ClassVar

import equinox as eqx
import haliax as hax
import jmp

from .config import RMTConfig
from .rmt import RMT
from src.nn import AbstractModel


class RMTModel(AbstractModel):
    module: RMT
    config: RMTConfig = eqx.field(static=True)
    module_cls: ClassVar[type] = RMT
    config_cls: ClassVar[type] = RMTConfig

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
