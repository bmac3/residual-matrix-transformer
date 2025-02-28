from abc import abstractmethod
from typing import Any

import equinox as eqx
from jax import numpy as jnp
from jax import tree_util as jtu


class AbstractCountableModule(eqx.Module):

    @property
    def size(self):
        return int(sum(jnp.prod(jnp.array(p.shape)) for p in jtu.tree_leaves(self)))


class AbstractSowableModule(AbstractCountableModule):
    intermediates_cache: eqx.AbstractVar[dict]

    def sow(self, key: str, value: Any):
        self.intermediates_cache[key] = value
        return value

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        try:
            self.intermediates_cache.clear()
            ret = self.forward(*args, **kwargs)
            return ret, self.intermediates_cache.copy()
        finally:
            self.intermediates_cache.clear()
    
    def sow_fn(self, func):
        def wrapped(*args, **kwargs):
            ret, ret_intermediates = func(*args, **kwargs)
            self.intermediates_cache.update(ret_intermediates)
            return ret
        return wrapped
