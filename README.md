# MLX-GPT

In this repo, I ported Andrej Karpathy's [NanoGPT](https://github.com/karpathy/build-nanogpt) implementation of GPT-2 into [MLX](https://github.com/ml-explore/mlx). This was done both for personal learning and to explore the feasibility of running (and training) large language models (LLMs) on consumer hardware like a MacBook.

## Requirements

To install MLX using pip, these are [needed](https://github.com/ml-explore/mlx/issues/10#issuecomment-1843415420):

- Machine with M-series chip
- MacOS >= 13.0
- Python between 3.8-3.11

Additionally, these packages are needed:

- mlx
- tiktoken (for using GPT-2's tokenizer)
- transformers, torch (for loading HuggingFace's GPT-2 models)

## Usage

To run the script and generate text, use the following command:

```bash
python mlx_gpt.py
```

By default, this will load the smallest (124M parameters) GPT-2 model from [HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/gpt2), and run some generation:

Here's some example output I got:

```
> I'm a language model, we're trying to solve the problem like a real human and we're trying to understand the human anatomy and know. And
> I'm a language model, but I like the way that people define what they write or do. And as a translator I write it myself.
> I'm a language model, not some advanced computer, and I'm talking about more than just writing numbers. I'm talking about the development of an
> I'm a language model, not a language model," says the study's lead author, Dr. Robert W. Smith of Yale University. The scientists
> I'm a language model, so you can see in the examples what the type of the current data is. That makes it harder to learn something of
```

You can modify the script to customize the input prompt and generation parameters such as model type and maximum length.

## Training

WIP

## References

- [build-nanogpt](https://github.com/karpathy/build-nanogpt)
- [Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU) (Youtube)
