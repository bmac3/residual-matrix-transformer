from absl import app
from clu import metric_writers, periodic_actions
from fire import Fire
import haliax as hax
import jax
from jax import numpy as jnp
from jax import random as jr
import jmp
import optax
from transformers import AutoTokenizer
from tqdm import tqdm
import yaml

from src import TrainStep, adamw, load_dataset, z_loss
from src.models.rmt import RMTConfig, RMTModel
from src.models.transformer import TransformerConfig, TransformerModel


dtypes = {
    'f32': jnp.float32,
    'f16': jnp.float16,
    'bf16': jnp.bfloat16
}


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


def init_transformer(
        tokenizer_name,
        model_config,
        full_dtype,
        half_dtype,
        rng_key
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
        key=rng_key,
    )


def init_rmt(
        tokenizer_name,
        model_config,
        full_dtype,
        half_dtype,
        rng_key
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
        key=rng_key,
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


def train_model(init_model):

    def train(
            *,
            logdir,
            tokenizer_name,
            sequence_length,
            batch_size,
            data_path,
            seed,
            lr,
            grad_clip,
            warmup_steps,
            total_steps,
            full_dtype,
            half_dtype,
            **model_config
        ):
        full = dtypes[full_dtype]
        half = dtypes[half_dtype]
        data_key, model_key = jr.split(jr.PRNGKey(seed), 2)
        dataset = init_dataset(tokenizer_name, sequence_length, batch_size, data_path, data_key)
        model = init_model(tokenizer_name, model_config, full, half, model_key)
        tx = init_optimizer(model, lr, grad_clip, warmup_steps, total_steps)
        opt_state = tx.init(model)
        mp_policy = jmp.Policy(
            param_dtype=full,
            compute_dtype=half,
            output_dtype=half
        )
        train_step = jax.jit(TrainStep.build(loss_fn, tx, mp_policy))
        writer = metric_writers.create_default_writer(logdir)
        hooks = [
            periodic_actions.ReportProgress(
                num_train_steps=total_steps,
                every_steps=10, writer=writer),
            periodic_actions.Profile(logdir=logdir)
        ]
        loss_scale = jmp.DynamicLossScale(jnp.float32(2**12), min_loss_scale=jnp.float32(1.0))
        for step in tqdm(range(total_steps)):
            batch = dataset[step]
            loss, model, opt_state, loss_scale = train_step(model, batch, opt_state, loss_scale)
            writer.write_scalars(step, dict(loss=loss))
            for hook in hooks:
                hook(step)
        
    return train


if __name__ == '__main__':
    Fire({
        'transformer': train_model(init_transformer),
        'rmt': train_model(init_rmt)
    })
