from typing import Union

import equinox as eqx
import haliax as hax
from jaxtyping import PRNGKeyArray
import jmp

from .mixins import AbstractCountableModule
from ..mixed_precision import mixed_precision
from ..model import AbstractConfig
from ..param import Param, ParamAxis


class Linear(AbstractCountableModule):
    param: Param
    policy: jmp.Policy = eqx.field(static=True)

    @mixed_precision
    def __call__(self, x: hax.NamedArray):
        return hax.dot(x, self.param.weight, axis=self.param.In.as_axis_spec()) * self.param.multiplier
    
    def scale(self, factor: float):
        return Linear(param=self.param.scale(factor), policy=self.policy)

    def set(self, value: hax.NamedArray):
        return Linear(param=self.param.set(value), policy=self.policy)
        
    @classmethod
    def init(
            cls, 
            config: AbstractConfig,
            In: Union[tuple[ParamAxis], ParamAxis], 
            Out: Union[tuple[ParamAxis], ParamAxis],
            key: PRNGKeyArray, 
            apply_wd: bool = True,
        ):
        return cls(
            param=Param.init(
                config=config,
                In=In,
                Out=Out,
                key=key,
                apply_wd=apply_wd
            ),
            policy=jmp.Policy(
                compute_dtype=config.half_dtype,
                param_dtype=config.full_dtype,
                output_dtype=config.half_dtype,
            )
        )
