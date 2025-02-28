import os

from datasets import load_dataset
from fire import Fire
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def download_and_tokenize(
        dataset_name, 
        tokenizer_name, 
        save_dir, 
        batch_size=1024,
        n_shards=1024,
        test_data_ratio=0.0,
        subset=None
    ):
    os.makedirs(save_dir, exist_ok=True)
    ds = load_dataset(dataset_name, subset, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if test_data_ratio > 0:
        assert len(ds) == 1, 'Dataset already has multiple splits'
        ds = ds[next(iter(ds.keys()))].train_test_split(test_size=test_data_ratio, shuffle=True)

    def encode(batch):
        ret = tokenizer(batch['text'])
        ret['input_ids'] = [[tokenizer.bos_token_id] + x for x in ret['input_ids']]
        ret['len'] = [len(x) for x in ret['input_ids']]
        return ret

    ds = ds.map(encode, batched=True, batch_size=batch_size, desc="tokenizing...")

    dtype = np.int32
    assert np.iinfo(dtype).max >= tokenizer.vocab_size, f'vocab size {tokenizer.vocab_size} is too large for dtype {dtype}'

    for split, dset in ds.items():
        save_path = f'{save_dir}/{split}.bin'
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        arr = np.memmap(save_path, dtype=dtype, mode='w+', shape=(arr_len,))
        n_shards = min(n_shards, len(dset))
        idx = 0
        for batch_idx in tqdm(range(n_shards), desc=f'saving {split} data to disk...'):
            batch = dset.shard(num_shards=n_shards, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['input_ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()


if __name__ == '__main__':
    Fire(download_and_tokenize)
