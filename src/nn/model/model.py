import equinox as eqx
from jax import random as jr
from jaxtyping import PRNGKeyArray

from .config import AbstractConfig
from ..mixed_precision import cast_to_param_dtype
from ..modules import AbstractCountableModule


class AbstractModel(AbstractCountableModule):
    module: eqx.AbstractVar[eqx.Module]
    config: eqx.AbstractVar[AbstractConfig]
    module_cls: eqx.AbstractClassVar[type]
    config_cls: eqx.AbstractClassVar[type]

    @classmethod
    def init(
            cls,
            config: AbstractConfig,
            key: PRNGKeyArray
        ):
        module = cls.module_cls.init(
            config=config,
            key=key
        )
        return cls(
            module=cast_to_param_dtype(module),
            config=config
        )

    def save(self, filename):
        with open(filename, 'wb') as f:
            f.write(self.config.encode())
            f.write(b'\n')
            eqx.tree_serialise_leaves(f, self.model)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            config = cls.config_cls.decode(f.readline())
            module_like = cls.module_cls.init(config, jr.PRNGKey(0))
            module = eqx.tree_deserialise_leaves(f, module_like)
        return cls(module=module, config=config)
