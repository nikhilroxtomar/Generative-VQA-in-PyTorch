import os
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize

class Vocab:
    def __init__(self):
        self.vocab = None
        self.vocab_size = None
        self.word2idx = None
        self.idx2word = None

        # special tokens
        self.pad = '<pad>'
        self.bos = '<bos>'
        self.eos = '<eos>'
        self.unk = '<unk>'

    def build_vocab(self, df, min_freq=1):
        counter = Counter()
        for ans in df['answer']:
            tokens = word_tokenize(ans.lower())
            counter.update(tokens)

        # Build vocabulary
        vocab = sorted([word for word, freq in counter.items() if freq >= min_freq])
        vocab = [self.pad, self.bos, self.eos, self.unk] + vocab
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for word, idx in word2idx.items()}
        
        self.vocab = vocab
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.vocab_size = len(vocab)

        # Special token IDs for easy reference
        self.pad_token_id = self.word2idx["<pad>"]
        self.bos_token_id = self.word2idx["<bos>"]
        self.eos_token_id = self.word2idx["<eos>"]
        self.unk_token_id = self.word2idx["<unk>"]

    def encoder(self, text, max_len):
        # Tokenize
        tokens = word_tokenize(text.lower())

        # Convert to IDs
        token_ids = [self.word2idx.get(token, self.unk_token_id) for token in tokens]
        token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        # Pad or truncate
        if len(token_ids) < max_len:
            token_ids += [self.pad_token_id] * (max_len - len(token_ids))
        else:
            token_ids = token_ids[:max_len]
        
        return token_ids

    def decoder(self, token_ids):
        tokens = []
        for idx in token_ids:
            if idx == self.eos_token_id:
                break
            if idx in (self.pad_token_id, self.bos_token_id):
                continue
            tokens.append(self.idx2word.get(idx, "<unk>"))
        return ' '.join(tokens).strip()
    

if __name__ == "__main__":
    # Config
    DATA_DIR = "./mini_vqa_v2"
    CSV_PATH = os.path.join(DATA_DIR, "metadata.csv")
    answer_max_len = 10

    # Load CSV
    metadata = pd.read_csv(CSV_PATH)

    # Answer vocabulary
    vocab = Vocab()
    vocab.build_vocab(metadata, min_freq=2)
    answer_vocab_size = len(vocab.vocab)
    print(f"Answer Vocab Size: {answer_vocab_size}")

    # Example encoding/decoding
    sample_answer = metadata['answer'].values
    text = sample_answer[0]
    print("")

    encoded = vocab.encoder(text, answer_max_len)
    decoded = vocab.decoder(encoded)
    print(f"Sample Answer: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
