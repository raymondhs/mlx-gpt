from dataclasses import dataclass
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


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
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def __call__(self, idx, targets=None):
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
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

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

    def generate(self, x, max_length=30):
        while x.shape[1] < max_length:
            # forward the model to get the logits
            logits, _ = model(x) # (B, T, vocab_size)
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


if __name__ == '__main__':
    import tiktoken
    mx.random.seed(1234)
    model = GPT.from_pretrained('gpt2')
    num_return_sequences = 5
    max_length = 30
    k = 50
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("I'm a language model,")
    x = mx.array(tokens)
    x = mx.repeat(x[None,:], repeats=num_return_sequences, axis=0)
    x = model.generate(x)
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)
