from typing import Callable

import equinox as eqx
import jmp
import optax

from .dataset import MiniBatch
from .nn import AbstractModel, cast_to_param_dtype


class TrainStep:

    def __init__(
            self, 
            loss_fn: Callable, 
            tx: optax.GradientTransformation, 
            policy: jmp.Policy
        ):
        self.loss_fn = loss_fn
        self.tx = tx
        self.policy = policy

    def compute_loss(
            self, 
            model: AbstractModel, 
            batch: MiniBatch, 
            loss_scale: jmp.LossScale
        ):
        loss = self.loss_fn(model, batch)
        loss = loss_scale.scale(loss)
        return loss

    def __call__(
            self, 
            model: AbstractModel, 
            batch: MiniBatch, 
            opt_state: optax.OptState, 
            loss_scale: jmp.LossScale
        ):
        loss, grads = eqx.filter_value_and_grad(self.compute_loss)(model, batch, loss_scale)
        grads = self.policy.cast_to_compute(grads)
        loss, grads = loss_scale.unscale((loss, grads))
        grads_finite = jmp.all_finite(grads)
        loss_scale = loss_scale.adjust(grads_finite)
        grads = cast_to_param_dtype(grads)
        updates, new_opt_state = self.tx.update(grads, opt_state, model)
        new_model = optax.apply_updates(model, updates)
        model, opt_state = jmp.select_tree(grads_finite, (new_model, new_opt_state), (model, opt_state))
        return loss, model, opt_state, loss_scale
    
    @classmethod
    def build(cls, loss_fn, tx, mp_policy):
        
        def train_step(model, batch, opt_state, loss_scale):
            return cls(loss_fn, tx, mp_policy)(model, batch, opt_state, loss_scale)
        
        return train_step
