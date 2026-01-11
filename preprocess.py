import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from model import VQAModel


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
    
class VQADataset(Dataset):
    def __init__(self, df, img_dir, question_tokenizer, text_processor, clip_processor, 
                 question_max_len=32, answer_max_len=16):  # Different lengths
        self.df = df
        self.img_dir = img_dir
        self.question_tokenizer = question_tokenizer
        self.text_processor = text_processor
        self.clip_processor = clip_processor
        self.question_max_len = question_max_len
        self.answer_max_len = answer_max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_path'])
        image = Image.open(img_path).convert('RGB')
        question = row['question']
        answer = row['answer']

        # Tokenize question with question max length
        question_tokenized = self.question_tokenizer(
            question, 
            padding='max_length', 
            truncation=True,
            max_length=self.question_max_len, 
            return_tensors='pt'
        )
        
        # Tokenize answer with answer max length
        answer_ids = self.text_processor.encoder(answer, max_len=self.answer_max_len)

        # Process image
        image = self.clip_processor(image)

        return {
            'image_path': img_path,
            'image': image,
            'question_ids': question_tokenized['input_ids'].squeeze(0),
            'question_mask': question_tokenized['attention_mask'].squeeze(0),
            'answer_ids': torch.tensor(answer_ids, dtype=torch.long)
        }

    
if __name__ == "__main__":
    # Paths
    DATA_DIR = "./mini_vqa_v2"
    CSV_PATH = os.path.join(DATA_DIR, "metadata.csv")

    # Hyperparameters
    batch_size = 16
    question_max_len = 16
    answer_max_len = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load CSV
    metadata = pd.read_csv(CSV_PATH)

    # Answer vocabulary
    vocab = Vocab()
    vocab.build_vocab(metadata, min_freq=2)
    answer_vocab_size = len(vocab.vocab)
    print(f"Answer Vocab Size: {answer_vocab_size}")

    # Split: 80% train, 10% val, 10% test
    train_df, test_df = train_test_split(metadata, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    print()

    # Model with separate lengths
    model = VQAModel(
        vocab_size=answer_vocab_size, 
        device=device,
        question_max_len=question_max_len,
        answer_max_len=answer_max_len,
        pad_token_id=vocab.pad_token_id,
        bos_token_id=vocab.bos_token_id,
        eos_token_id=vocab.eos_token_id,
        unk_token_id=vocab.unk_token_id
    ).to(device)

    # Tokenizers and processors
    clip_processor = model.clip_preprocess
    question_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

    # Add PAD token if missing
    if question_tokenizer.pad_token is None:
        question_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.gpt2_model.resize_token_embeddings(len(question_tokenizer))

    # Datasets with separate lengths
    train_dataset = VQADataset(
        train_df, DATA_DIR, question_tokenizer, vocab, 
        clip_processor=clip_processor, 
        question_max_len=question_max_len, 
        answer_max_len=answer_max_len
    )
    val_dataset = VQADataset(
        val_df, DATA_DIR, question_tokenizer, vocab,
        clip_processor=clip_processor,
        question_max_len=question_max_len,
        answer_max_len=answer_max_len
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for batch in train_loader:
        images = batch['image']
        ques_ids = batch['question_ids']
        attn_mask = batch["question_mask"]
        answers = batch['answer_ids']

        print(f"Image: {images.shape}")
        print(f"Question Ids: {ques_ids.shape}")
        print(f"Attention Mask: {attn_mask.shape}")
        print(f"Answer Ids: {answers.shape}")

        print(answers[0])

        break
