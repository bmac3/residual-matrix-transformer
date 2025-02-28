from collections import namedtuple

import grain.python as grain
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import PRNGKeyArray
import haliax as hax
import numpy as np


def create_position_ids(arr, bos_id=0, start=0):
    zero_mask = arr == bos_id
    indices = np.arange(len(arr))
    group_starts = np.maximum.accumulate(np.where(zero_mask, indices, -1))
    prev_offset = np.where(group_starts == group_starts[0], start, 0)
    return indices - group_starts + prev_offset


def create_attention_mask(position_ids):
    seq_length = len(position_ids)
    pos_ids_unsqueeze = position_ids[:, None]
    pos_ids_expand = np.broadcast_to(pos_ids_unsqueeze, (seq_length, seq_length))
    causal_mask = pos_ids_expand >= pos_ids_expand.transpose()
    sequence_starts = np.concatenate(
        [np.ones((1,)), (position_ids[1:] < position_ids[:-1])]
    )
    sequence_ids = np.cumsum(sequence_starts) - 1
    seq_ids_unsqueeze = sequence_ids[:, None]
    seq_ids_expand = np.broadcast_to(seq_ids_unsqueeze, (seq_length, seq_length))
    sequence_mask = seq_ids_expand == seq_ids_expand.transpose()
    attention_mask = causal_mask & sequence_mask
    return attention_mask


MiniBatch = namedtuple('MiniBatch', 'input_ids attention_mask target_ids')


def preprocess(token_ids, bos_id=0):
    input_ids = token_ids[:-1]
    target_ids = token_ids[1:]
    position_ids = create_position_ids(input_ids, bos_id=bos_id)
    attention_mask = create_attention_mask(position_ids)
    return MiniBatch(input_ids, attention_mask, target_ids)


def to_hax(batch, Pos, KVPos, Batch):
    return MiniBatch(
        hax.NamedArray(batch.input_ids, (Batch, Pos)),
        hax.NamedArray(batch.attention_mask, (Batch, Pos, KVPos)),
        hax.NamedArray(batch.target_ids, (Batch, Pos))
    )


def load_dataset(
        filename: str,
        Pos: hax.Axis,
        KVPos: hax.Axis,
        Batch: hax.Axis,
        key: PRNGKeyArray,
        dtype: np.dtype = np.int32,
        bos_id: int = 0,
    ):
    arr = np.memmap(filename, dtype=dtype, mode='r')
    sequence_length_plus = Pos.size + 1
    seed = int(jr.randint(key, (), 0, jnp.iinfo(jnp.int32).max))
    return (
        grain.MapDataset
            .source(arr)
            .batch(sequence_length_plus, drop_remainder=True)
            .map(lambda x: preprocess(x, bos_id=bos_id))
            .shuffle(seed=seed)
            .batch(Batch.size, drop_remainder=True)
            .map(lambda x: to_hax(x, Pos, KVPos, Batch))
    )
