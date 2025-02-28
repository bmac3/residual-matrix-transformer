import equinox as eqx

from .mixins import AbstractSowableModule


class AbstractResidualBlock(AbstractSowableModule):
    module: eqx.AbstractVar[AbstractSowableModule]
    module_cls: eqx.AbstractClassVar[type]
    intermediates_cache: eqx.AbstractVar[dict]

    def forward(self, x, *args, **kwargs):
        x = self.sow_fn(self.module)(x, *args, **kwargs) + x
        x = self.sow(f'{type(self.module).__name__} residual', x)
        return x
    
    @classmethod
    def init(cls, *args, **kwargs):
        return cls(module=cls.module_cls.init(*args, **kwargs))
