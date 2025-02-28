from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Optional

import equinox as eqx
import haliax as hax


class AxisType(StrEnum):
    FINITE = 'finite'
    INFINITE = 'infinite'


@dataclass(frozen=True)
class ParamAxis:
    name: str
    size: int
    axis_type: AxisType
    fan_dim: Optional[int] = eqx.field(default=None)
    base_fan_dim: Optional[int] = eqx.field(default=None)

    def __post_init__(self):
        if self.fan_dim is None:
            object.__setattr__(self, 'fan_dim', self.size)
        if self.base_fan_dim is None:
            object.__setattr__(self, 'base_fan_dim', self.fan_dim)

    @property
    def axis(self) -> hax.Axis:
        return hax.Axis(self.name, self.size)


@dataclass(frozen=True)
class ParamAxisSpec:
    axes: tuple[ParamAxis]

    def __post_init__(self):
        if isinstance(self.axes, ParamAxis):
            object.__setattr__(self, 'axes', (self.axes,))
        elif isinstance(self.axes, tuple):
            for ax in self.axes:
                if not isinstance(ax, ParamAxis):
                    raise ValueError(f"Invalid axes: {self.axes}")
        else:
            raise ValueError(f"Invalid axes: {self.axes}")

    @property
    def fan_dim(self) -> int:
        return math.prod(ax.fan_dim for ax in self.axes)
    
    @property
    def base_fan_dim(self) -> int:
        return math.prod(ax.base_fan_dim for ax in self.axes)
    
    @property
    def width_ratio(self) -> float:
        return self.fan_dim / self.base_fan_dim
    
    @property
    def axis_type(self) -> AxisType:
        for ax in self.axes:
            if ax.axis_type == AxisType.INFINITE:
                return AxisType.INFINITE
        else:
            return AxisType.FINITE
    
    def as_axis_spec(self) -> hax.AxisSpec:
        return tuple(ax.axis for ax in self.axes)
    
    def __add__(self, other: 'ParamAxisSpec') -> 'ParamAxisSpec':
        return ParamAxisSpec(self.axes + other.axes)
