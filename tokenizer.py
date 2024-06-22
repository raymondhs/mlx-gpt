from datasets import load_dataset
from transformers import AutoTokenizer

data = load_dataset("wikimedia/wikipedia", "20231101.id", split="train")

def get_training_corpus():
    batch_size = 1000
    for i in range(0, len(data), batch_size):
        batch = data[i: i + batch_size]
        yield [title + "\n" + text for title, text in zip(batch["title"], batch["text"])]
training_corpus = get_training_corpus()

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 50257)
tokenizer.save_pretrained("indonesian-wikipedia-tokenizer")
