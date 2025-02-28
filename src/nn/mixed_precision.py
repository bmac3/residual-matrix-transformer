import equinox as eqx
from jax import tree_util as jtu


class mixed_precision:

    def __init__(self, method):
        self.method = method

    def __get__(self, instance, owner):
        
        if instance is None:
            return self
        
        def wrapped(*args, **kwargs):
            module, args, kwargs = instance.policy.cast_to_compute((instance, args, kwargs))
            out = self.method(module, *args, **kwargs)
            out = instance.policy.cast_to_output(out)
            return out

        return wrapped


def cast_to_param_dtype(module):

    def is_leaf(x):
        return hasattr(x, 'policy')

    def cast(x):
        if is_leaf(x):
            return x.policy.cast_to_param(x)
        else:
            return x

    return jtu.tree_map(cast, module, is_leaf=is_leaf)
