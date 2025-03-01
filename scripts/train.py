import abc
from typing import Dict, Union

from clu import metric_writers, periodic_actions
import haliax as hax
import jax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import PRNGKeyArray
from jsonargparse import auto_cli
import jmp
import optax
from transformers import AutoTokenizer
from tqdm import tqdm

from src import TrainStep, adamw, load_dataset, z_loss
from src.models.rmt import RMTConfig, RMTModel
from src.models.transformer import TransformerConfig, TransformerModel


def init_dataset(
        tokenizer_name,
        sequence_length,
        batch_size,
        data_path,
        rng_key,
    ):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    Pos = hax.Axis('Pos', sequence_length)
    KVPos = Pos.alias('KVPos')
    Batch = hax.Axis('Batch', batch_size)
    return load_dataset(
        data_path, 
        Pos, 
        KVPos, 
        Batch, 
        bos_id=tokenizer.bos_token_id, 
        key=rng_key
    )


def init_optimizer(
        model,
        lr,
        grad_clip,
        warmup_steps,
        decay_steps,
    ):
    return optax.chain(
        optax.clip_by_global_norm(grad_clip),
        adamw(
            model=model,
            global_lr=lr,
            schedule=optax.schedules.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=1.0,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=0.1
            ),
            b1=0.9,
            b2=0.95,
        )
    )


def loss_fn(model, batch):
    logits = model(batch.input_ids, batch.attention_mask)
    return z_loss(logits, 'Vocab', batch.target_ids).mean().scalar()


class Train(abc.ABC):

    def __init__(
            self,
            logdir: str,
            tokenizer_name: str,
            sequence_length: int,
            batch_size: int,
            data_path: str,
            lr: float,
            grad_clip: float,
            warmup_steps: int,
            total_steps: int,
            full_dtype: str,
            half_dtype: str,
            model_config: Dict[str, Union[int, float]],
            seed: int
        ):
        data_key, model_key = jr.split(jr.PRNGKey(seed), 2)
        self.dataset = init_dataset(tokenizer_name, sequence_length, batch_size, data_path, data_key)
        self.model = self._init_model(tokenizer_name, model_config, full_dtype, half_dtype, model_key)
        self.tx = init_optimizer(self.model, lr, grad_clip, warmup_steps, total_steps)
        self.opt_state = self.tx.init(self.model)
        mp_policy = jmp.Policy(
            param_dtype=full_dtype,
            compute_dtype=half_dtype,
            output_dtype=full_dtype
        )
        self.train_step = jax.jit(TrainStep.build(loss_fn, self.tx, mp_policy))
        self.total_steps = total_steps
        self.writer = metric_writers.create_default_writer(logdir)
        self.hooks = [
            periodic_actions.ReportProgress(
                num_train_steps=total_steps,
                every_steps=10, writer=self.writer),
            periodic_actions.Profile(logdir=logdir)
        ]
        self.loss_scale = jmp.DynamicLossScale(jnp.float32(2**12), min_loss_scale=jnp.float32(1.0))

    @staticmethod
    @abc.abstractmethod
    def _init_model(
            tokenizer_name: str, 
            model_config: dict, 
            full_dtype: str, 
            half_dtype: str, 
            rng_key: PRNGKeyArray
        ):
        raise NotImplementedError

    def __call__(self):
        for step in tqdm(range(self.total_steps)):
            batch = self.dataset[step]
            loss, self.model, self.opt_state, self.loss_scale = self.train_step(self.model, batch, self.opt_state, self.loss_scale)
            self.writer.write_scalars(step, dict(loss=loss))
            for hook in self.hooks:
                hook(step)


class TrainTransformer(Train):

    @staticmethod
    def _init_model(
            tokenizer_name: str, 
            model_config: Dict[str, Union[int, float]],
            full_dtype: str, 
            half_dtype: str, 
            rng_key: PRNGKeyArray
        ):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return TransformerModel.init(
            TransformerConfig.init(
                vocab_size=tokenizer.vocab_size,
                embed_dim=model_config['embed_dim'],
                n_heads=model_config['n_heads'],
                head_dim=model_config['embed_dim'] // model_config['n_heads'],
                n_layers=model_config['n_layers'],
                n_neurons=model_config['n_neurons'],
                rmsnorm_eps=model_config['rmsnorm_eps'],
                full_dtype=full_dtype,
                half_dtype=half_dtype,
            ),
            key=rng_key
        )
    

class TrainRMT(Train):

    @staticmethod
    def _init_model(
            tokenizer_name: str, 
            model_config: Dict[str, Union[int, float]],
            full_dtype: str, 
            half_dtype: str, 
            rng_key: PRNGKeyArray
        ):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return RMTModel.init(
            RMTConfig.init(
                vocab_size=tokenizer.vocab_size,
                reskey_dim=model_config['reskey_dim'],
                resval_dim=model_config['resval_dim'],
                rank=model_config['rank'],
                n_layers=model_config['n_layers'],
                n_neurons=model_config['n_neurons'],
                rmsnorm_eps=model_config['rmsnorm_eps'],
                full_dtype=full_dtype,
                half_dtype=half_dtype,
            ),
            key=rng_key
        )


if __name__ == '__main__':
    auto_cli(
        {
            'transformer': TrainTransformer,
            'rmt': TrainRMT
        },
        as_positional=False
    )()
