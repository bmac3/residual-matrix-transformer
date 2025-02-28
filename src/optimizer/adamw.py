from operator import attrgetter
from typing import Any, Optional

import equinox as eqx
import jax
import optax

from src.nn import Param, ParamType


def map_params_fn(fn):

    def map_fn(x):
        if isinstance(x, Param):
            return eqx.tree_at(attrgetter('weight'), x, fn(x))
        else:
            return x
        
    def apply_map_fn(model):
        return jax.tree.map(map_fn, model, is_leaf=lambda x: isinstance(x, Param))
    
    return apply_map_fn


def scale_lr_by_mup(lr, x):
    if x.param_type == ParamType.HIDDEN:
        lr /= x.width_ratio
    return lr


def get_mup_learning_rates(global_lr, model):
    lrs = set()
    add_to_lrs = lambda x: lrs.add(scale_lr_by_mup(global_lr, x))
    map_params_fn(add_to_lrs)(model)
    return lrs


def lr_label(lr):
    return f'lr={lr:.6e}'


def get_mup_learning_rate_transforms(global_lr, model):
    lrs = get_mup_learning_rates(global_lr, model)
    return {lr_label(lr): optax.scale_by_learning_rate(lr, flip_sign=False) for lr in lrs}


def scale_by_mup_learning_rate(global_lr, model):
    transforms = get_mup_learning_rate_transforms(global_lr, model)
    label_fn = lambda x: lr_label(scale_lr_by_mup(global_lr, x))
    return optax.multi_transform(transforms, map_params_fn(label_fn))


def adamw(
        model,
        global_lr,
        schedule,
        b1,
        b2,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        mu_dtype: Optional[Any] = None,
        weight_decay: float = 1e-4,
        *,
        nesterov: bool = False,
    ):
    return optax.chain(
        optax.scale_by_adam(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
        ),
        scale_by_mup_learning_rate(global_lr, model),
        optax.add_decayed_weights(weight_decay, map_params_fn(lambda x: x.apply_wd)),
        # use scale_by_learning_rate instead of scale_by_schedule to play nice with inject_hyperparameters
        optax.scale_by_learning_rate(schedule, flip_sign=True),
    )
