from typing import Union

import equinox as eqx
import haliax as hax
from jaxtyping import PRNGKeyArray
import jmp

from .mixins import AbstractCountableModule
from ..mixed_precision import mixed_precision
from ..model import AbstractConfig
from ..param import Param, ParamAxis


class Embedding(AbstractCountableModule):
    param: Param
    policy: jmp.Policy = eqx.field(static=True)

    @mixed_precision
    def __call__(self, x: hax.NamedArray):
        return hax.take(self.param.weight, axis=self.param.In.as_axis_spec()[0], index=x) * self.param.multiplier

    @classmethod
    def init(
            cls, 
            config: AbstractConfig,
            Vocab: ParamAxis, 
            Embed: Union[tuple[ParamAxis], ParamAxis],
            key: PRNGKeyArray, 
            apply_wd: bool = False,
            scale_factor: float = 1.0
        ):
        if not isinstance(Vocab, ParamAxis):
            raise ValueError(f"Expected ParamAxis, got {type(Vocab)}")
        return cls(
            param=Param.init(
                config=config,
                In=Vocab,
                Out=Embed,
                key=key,
                apply_wd=apply_wd
            ).scale(scale_factor),
            policy=jmp.Policy(
                compute_dtype=config.half_dtype,
                param_dtype=config.full_dtype,
                output_dtype=config.half_dtype,
            )
        )
