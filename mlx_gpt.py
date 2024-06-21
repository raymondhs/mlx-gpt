import math
import os
import time
from dataclasses import dataclass
from functools import partial
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_reduce

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # scale for attention queries
        self.scale = (config.n_embd // config.n_head)**-0.5
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def __call__(self, x, mask=None):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        k = k.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        q = q.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        v = v.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approx='precise')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def __call__(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)        
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def __call__(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd)

    def __call__(self, idx):
        # idx is of shape (B, T)
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = mx.array(np.arange(T)) # shape (T)
        pos_emb = self.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # create additive causal mask
        inds = mx.arange(T)
        mask = (inds[:, None] < inds) * -1e9
        mask = mask.astype(x.dtype)
        # forward the blocks of the transformer
        for block in self.h:
            x = block(x, mask)
        # forward the final layernorm and the classifier
        x = self.ln_f(x)
        # weight sharing scheme
        logits = self.wte.as_linear(x) # (B, T, vocab_size)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = dict(tree_flatten(model.parameters()))
        sd_keys = sd.keys()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        sd_keys_hf = [k for k in sd_keys_hf if not k.startswith('lm_head')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k_hf in sd_keys_hf:
            k = k_hf.replace("transformer.", "")
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k_hf].shape[::-1] == sd[k].shape
                sd[k] = mx.array(sd_hf[k_hf].t().numpy())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k_hf].shape == sd[k].shape
                sd[k] = mx.array(sd_hf[k_hf].numpy())
        model.load_weights(list(sd.items()))
        return model

    def generate(self, x, max_length=30, k=50):
        while x.shape[1] < max_length:
            # forward the model to get the logits
            logits = self(x) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = nn.softmax(logits, axis=-1)
            sorted_indices = mx.argsort(probs, axis=-1)
            sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (B, 50)
            topk_probs = mx.zeros_like(sorted_probs)
            topk_probs[:, -k:] = sorted_probs[:, -k:]
            # select a token from the top-k probabilities
            # note: categorical does not demand the input to sum to 1
            ix = mx.random.categorical(mx.log(topk_probs))[:, None] # (B, 1)
             # gather the corresponding indices
            xcol = mx.take_along_axis(sorted_indices, ix, axis=1)
            # append to the sequence
            x = mx.concatenate([x, xcol], axis=1)
        return x

# -----------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = mx.array(npt, dtype=mx.uint32)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "idwiki"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).reshape(B, T) # inputs
        y = (buf[1:]).reshape(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T
        return x, y

if __name__ == '__main__':
    mx.random.seed(1234)

    enc = tiktoken.get_encoding('gpt2')

    total_batch_size = 4096
    B = 4 # micro batch size
    T = 512 # sequence length
    assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
    grad_accum_steps = total_batch_size // (B * T)
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B=B, T=T, split="train")
    val_loader = DataLoaderLite(B=B, T=T, split="val")

    model = GPT.from_pretrained('gpt2')
    nparams = sum(
        x.size for k, x in tree_flatten(model.parameters()) if "embedding" not in k
    )
    print(f"number of params: {nparams / 1000**2:.3f} M")
    # use bfloat16 to save memory
    model.apply(lambda x: x.astype(mx.bfloat16))

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 200
    max_steps = 20000
    eval_steps = 250
    save_steps = 5000
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)

    def loss_fn(model, x, y, reduce=True):
        logits = model(x)
        losses = nn.losses.cross_entropy(logits, y)
        return mx.mean(losses) if reduce else mx.mean(losses, axis=(-1, -2))

    # optimize!
    optimizer = optim.AdamW(learning_rate=3e-4, betas=[0.9, 0.95], eps=1e-8)

    state = [model.state, optimizer.state]

    # create the log directory we will write checkpoints to and log to
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

    def eval_fn():
        val_loader.reset()
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            loss = loss_fn(model, x, y, reduce=False)
            loss = loss / val_loss_steps
            val_loss_accum += loss
        return val_loss_accum

    for step in range(max_steps):
        last_step = (step == max_steps - 1)

        # once in a while evaluate our validation loss
        if step % eval_steps == 0 or last_step:
            val_loss_accum = eval_fn()
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % save_steps == 0 or last_step):
                # optionally write model checkpoints
                model_path = os.path.join(log_dir, f"model_{step:05d}.safetensors")
                weights = tree_flatten(model.trainable_parameters())
                mx.save_safetensors(str(model_path), dict(weights))

        # once in a while generate from the model (except step 0, which is noise)
        if ((step > 0 and step % eval_steps == 0) or last_step):
            num_return_sequences = 4
            max_length = 32
            k = 50
            tokens = enc.encode("Jakarta adalah")
            x = mx.array(tokens)
            x = mx.repeat(x[None,:], repeats=num_return_sequences, axis=0)
            x = model.generate(x, max_length)
            for i in range(num_return_sequences):
                tokens = x[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(">", decoded)

        t0 = time.time()
        lr = get_lr(step)
        optimizer.learning_rate = lr
        accum_grads = tree_map(
            lambda x: mx.zeros_like(x), model.parameters()
        )
        accum_loss = 0
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            loss, grads = loss_and_grad_fn(model, x, y)
            accum_grads = tree_map(
                lambda acc, new: acc + new * (1.0 / grad_accum_steps),
                accum_grads,
                grads,
            )
            tree_map(
                lambda grad: mx.eval(grad),
                accum_grads,
            )
            accum_loss += loss
        loss = accum_loss / grad_accum_steps
        clipped_grads, _ = optim.clip_grad_norm(accum_grads, 1.0)
        optimizer.update(model, clipped_grads)
        mx.eval(state, loss)
        t1 = time.time()
        dt = t1 - t0 # time difference in seconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
        tokens_per_sec = tokens_processed / dt
        peak_mem = mx.metal.get_peak_memory() / 2**30
        print(f"step {step:5d} | loss: {loss.item():.6f} | lr {lr:.4e} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | peak mem: {peak_mem:.3f} GB")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss.item():.6f}\n")
