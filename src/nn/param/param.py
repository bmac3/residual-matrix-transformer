from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Union

import equinox as eqx
import haliax as hax
from jaxtyping import PRNGKeyArray

from .param_axis import AxisType, ParamAxis, ParamAxisSpec
from ..model import AbstractConfig


class ParamType(StrEnum):
    INPUT = 'input'
    OUTPUT = 'output'
    HIDDEN = 'hidden'


@dataclass(frozen=True)
class ParamSpec:
    In: ParamAxisSpec
    Out: ParamAxisSpec
    input_multiplier: float
    output_multiplier: float
    global_init_std: float
    output_zero_init: bool

    @property
    def param_type(self) -> ParamType:
        if self.In.axis_type == AxisType.INFINITE and self.Out.axis_type == AxisType.INFINITE:
            return ParamType.HIDDEN
        elif self.In.axis_type == AxisType.INFINITE and self.Out.axis_type == AxisType.FINITE:
            return ParamType.OUTPUT
        elif self.In.axis_type == AxisType.FINITE and self.Out.axis_type == AxisType.INFINITE:
            return ParamType.INPUT
        else:
            raise ValueError("No infinite axes found")

    @property
    def init_std(self) -> float:
        # table 8 from mup paper
        if self.param_type == ParamType.INPUT:
            return self.global_init_std / math.sqrt(self.In.fan_dim)
        elif self.param_type == ParamType.HIDDEN:
            return self.global_init_std / math.sqrt(self.In.fan_dim)
        else:
            if self.output_zero_init:
                # Appendix D.2 of mup paper
                return 0.0
            else:
                return self.global_init_std
            
    @property
    def multiplier(self) -> float:
        # table 8 from mup paper
        if self.param_type == ParamType.INPUT:
            return self.input_multiplier
        elif self.param_type == ParamType.HIDDEN:
            return 1.0
        else:
            return self.output_multiplier / self.In.width_ratio
        
    def init_weight(self, key: PRNGKeyArray) -> hax.NamedArray:
        joint_spec = (self.In + self.Out).as_axis_spec()
        return hax.random.normal(key, joint_spec) * self.init_std

    @classmethod
    def init(
            cls, 
            config: AbstractConfig,
            In: ParamAxisSpec,
            Out: ParamAxisSpec
        ):
        return cls(
            In=In,
            Out=Out,
            input_multiplier=config.input_multiplier,
            output_multiplier=config.output_multiplier,
            global_init_std=config.init_std,
            output_zero_init=config.output_zero_init
        )


class Param(eqx.Module):
    weight: hax.NamedArray
    In: ParamAxisSpec = eqx.field(static=True)
    Out: ParamAxisSpec = eqx.field(static=True)
    multiplier: float = eqx.field(static=True)
    param_type: ParamType = eqx.field(static=True)
    apply_wd: bool = eqx.field(static=True)

    @property
    def fan_in(self) -> int:
        return self.In.fan_dim
    
    @property
    def base_fan_in(self) -> int:
        return self.In.base_fan_dim

    @property
    def width_ratio(self) -> float:
        return self.In.width_ratio

    def scale(self, x: hax.NamedArray) -> 'Param':
        return Param(
            weight=self.weight * x,
            In=self.In,
            Out=self.Out,
            multiplier=self.multiplier,
            param_type=self.param_type,
            apply_wd=self.apply_wd
        )

    def set(self, weight: hax.NamedArray) -> 'Param':
        if weight.axes != self.weight.axes:
            raise ValueError(f"Invalid weight shape: {weight.shape}, expected {self.weight.shape}")
        return Param(
            weight=weight,
            In=self.In,
            Out=self.Out,
            multiplier=self.multiplier,
            param_type=self.param_type,
            apply_wd=self.apply_wd
        )
        
    @classmethod
    def init(
            cls,
            config: AbstractConfig,
            In: Union[tuple[ParamAxis], ParamAxis, ParamAxisSpec],
            Out: Union[tuple[ParamAxis], ParamAxis, ParamAxisSpec],
            key: PRNGKeyArray,
            apply_wd: bool = True
        ) -> 'Param':
        In = ParamAxisSpec(In)
        Out = ParamAxisSpec(Out)
        param_spec = ParamSpec.init(config, In, Out)
        return cls(
            weight=param_spec.init_weight(key),
            In=In,
            Out=Out,
            multiplier=param_spec.multiplier,
            param_type=param_spec.param_type,
            apply_wd=apply_wd
        )
