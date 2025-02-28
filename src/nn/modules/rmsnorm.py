from typing import Union

import equinox as eqx
import haliax as hax
from jax import random as jr
import jmp

from .mixins import AbstractCountableModule
from ..mixed_precision import mixed_precision
from ..model import AbstractConfig
from ..param import Param, ParamAxis


class RMSNorm(AbstractCountableModule):
    param: Param
    eps: float = eqx.field(static=True)
    policy: jmp.Policy = eqx.field(static=True)

    @mixed_precision
    def __call__(self, x: hax.NamedArray):
        inv_rms = hax.rsqrt((x ** 2).mean(self.param.Out.as_axis_spec()) + self.eps)
        return x * inv_rms * self.param.weight

    @classmethod
    def init(
            cls,
            config: AbstractConfig,
            axis: Union[tuple[ParamAxis], ParamAxis],
            apply_wd: bool = False
        ):
        param = Param.init(
            config=config,
            In=(),
            Out=axis,
            apply_wd=apply_wd,
            key=jr.PRNGKey(0)
        )
        return cls(
            param=param.set(hax.ones_like(param.weight)),
            eps=config.rmsnorm_eps,
            policy=jmp.Policy(
                compute_dtype=config.full_dtype,
                param_dtype=config.full_dtype,
                output_dtype=config.half_dtype,
            )
        )
