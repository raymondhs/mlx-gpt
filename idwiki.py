"""
Indonesian Wikipedia dataset
https://huggingface.co/datasets/wikimedia/wikipedia
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python idwiki.py
Will save shards to the local directory "idwiki".
"""

import os
import multiprocessing as mp
import numpy as np
from transformers import AutoTokenizer # use custom tokenizer
from transformers.utils import logging
logging.set_verbosity(40) # turn off long sequence warning
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
local_dir = "idwiki"
shard_size = int(5e6) # 5M tokens per shard, total of 45 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname('.'), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("wikimedia/wikipedia", "20231101.id", split="train")

# init the tokenizer
enc = AutoTokenizer.from_pretrained('./indonesian-wikipedia-tokenizer')
eot = enc.eos_token_id
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    text = doc['title'] + '\n' + doc['text']
    text = text.replace('\n\n', '\n')
    tokens.extend(enc.encode(text, add_special_tokens=False))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

if __name__ == '__main__':
    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count()//2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        total_tokens = 0
        for tokens in pool.imap(tokenize, fw, chunksize=16):

            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"idwiki_{split}_{shard_index:06d}")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                total_tokens += token_count + remainder
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"idwiki_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])
            total_tokens += token_count
        print(f"Total tokens: {total_tokens}")
