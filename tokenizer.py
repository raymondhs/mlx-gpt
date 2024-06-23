from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

wiki = load_dataset("wikimedia/wikipedia", "20231101.id", split="train")
wiki = wiki.remove_columns(["id", "url", "title"])
c4 = load_dataset("indonesian-nlp/mc4-id", "tiny", split="train", trust_remote_code=True)
c4 = c4.remove_columns(["url", "timestamp"])
data = concatenate_datasets([wiki, c4])

def get_training_corpus():
    batch_size = 1000
    batch = []
    for example in data:
        batch.append(example["text"])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
training_corpus = get_training_corpus()

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 50257)
tokenizer.save_pretrained("indonesian-wikipedia-tokenizer")
