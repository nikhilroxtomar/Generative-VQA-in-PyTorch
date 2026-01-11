import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from model import VQAModel
from train import VQADataset, Vocab, save_checkpoint, plot_losses


# ============================================================
# Optimizer: Different learning rates for model components
# ============================================================
def create_optimizer_with_differential_lr(model, clip_lr=1e-6, gpt_lr=1e-6, other_lr=5e-6):
    clip_params, gpt_params, other_params = [], [], []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'clip_model' in name:
                clip_params.append(param)
            elif 'gpt2_model' in name:
                gpt_params.append(param)
            else:
                other_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': clip_params, 'lr': clip_lr},
        {'params': gpt_params, 'lr': gpt_lr},
        {'params': other_params, 'lr': other_lr}
    ], weight_decay=1e-4)

    print(f"Optimizer: CLIP params: {len(clip_params)}, GPT-2 params: {len(gpt_params)}, Other params: {len(other_params)}")
    return optimizer


# ============================================================
# Training and Validation
# ============================================================
def train_one_epoch(model, dataloader, optimizer, device, vocab, scaler):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_token_id, label_smoothing=0.1)

    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        images = batch['image'].to(device)
        questions = {
            'input_ids': batch['question_ids'].to(device),
            'attention_mask': batch['question_mask'].to(device)
        }
        answers = batch['answer_ids'].to(device)

        # Mixed precision forward
        with torch.amp.autocast(device):
            logits = model(images, questions, answer_input_ids=answers)

            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_answers = answers[:, 1:].contiguous()

            loss = criterion(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_answers.view(-1)
            )

        # NaN protection
        if torch.isnan(loss):
            print("NaN loss detected, skipping batch.")
            continue

        # Scaled backprop
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_one_epoch(model, dataloader, device, vocab):
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_token_id)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            questions = {
                'input_ids': batch['question_ids'].to(device),
                'attention_mask': batch['question_mask'].to(device)
            }
            answers = batch['answer_ids'].to(device)

            with torch.amp.autocast("cuda"):
                logits = model(images, questions, answer_input_ids=answers)

                shifted_logits = logits[:, :-1, :].contiguous()
                shifted_answers = answers[:, 1:].contiguous()

                loss = criterion(
                    shifted_logits.view(-1, shifted_logits.size(-1)),
                    shifted_answers.view(-1)
                )

            total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================================================
# Main Fine-Tuning Routine
# ============================================================
def main():
    print("\n# =================================================")
    print("# VQA: Fine-Tuning")
    print("# =================================================\n")

    # Seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Paths
    DATA_DIR = "./mini_vqa_v2"
    CSV_PATH = os.path.join(DATA_DIR, "metadata.csv")
    PRETRAINED_CHECKPOINT = "./output/feature_extraction/vqa_checkpoint.pt"
    OUTPUT_DIR = "./output/fine_tuning"
    FINE_TUNED_CHECKPOINT = os.path.join(OUTPUT_DIR, "vqa_checkpoint.pt")
    LOG_CSV = os.path.join(OUTPUT_DIR, "train_log.csv")
    LOSS_GRAPH_PATH = os.path.join(OUTPUT_DIR, "loss_plot.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Hyperparameters
    batch_size = 8
    num_epochs = 10
    patience = 5
    clip_layers_to_unfreeze = 4
    gpt_layers_to_unfreeze = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint
    checkpoint = torch.load(PRETRAINED_CHECKPOINT, map_location=device)
    metadata = pd.read_csv(CSV_PATH)

    # Rebuild answer processor
    vocab = Vocab()
    
    vocab.vocab = checkpoint['vocab']
    vocab.vocab_size = len(checkpoint['vocab'])
    vocab.word2idx = checkpoint['word2idx']
    vocab.idx2word = checkpoint['idx2word']

    vocab.pad_token_id = checkpoint['pad_token_id']
    vocab.bos_token_id = checkpoint['bos_token_id']
    vocab.eos_token_id = checkpoint['eos_token_id']
    vocab.unk_token_id = checkpoint['unk_token_id']

    print(f"Answer vocabulary size: {len(vocab.vocab)}")

    # Model setup
    model = VQAModel(
        vocab_size=len(checkpoint['vocab']),
        device=device,
        question_max_len=checkpoint.get('question_max_len', 16),
        answer_max_len=checkpoint.get('answer_max_len', 10),
        pad_token_id=checkpoint['pad_token_id'],
        bos_token_id=checkpoint['bos_token_id'],
        eos_token_id=checkpoint['eos_token_id'],
        unk_token_id=checkpoint['unk_token_id']
    ).to(device)

    # Tokenizer
    question_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    if question_tokenizer.pad_token is None:
        question_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.gpt2_model.resize_token_embeddings(len(question_tokenizer))

    # Load pretrained weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("Pretrained model loaded successfully!\n")

    # Unfreeze last layers
    print("=== UNFREEZING LAYERS FOR FINE-TUNING ===")
    model.unfreeze_clip_layers(num_layers=clip_layers_to_unfreeze)
    model.unfreeze_gpt2_layers(num_layers=gpt_layers_to_unfreeze)

    # Split dataset
    train_df, test_df = train_test_split(metadata, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}\n")

    train_dataset = VQADataset(train_df, DATA_DIR, question_tokenizer, vocab,
                               clip_processor=model.clip_preprocess)
    val_dataset = VQADataset(val_df, DATA_DIR, question_tokenizer, vocab,
                             clip_processor=model.clip_preprocess)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer and scheduler
    optimizer = create_optimizer_with_differential_lr(
        model,
        clip_lr=1e-6,
        gpt_lr=1e-6,
        other_lr=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler(device)

    # Fine-tuning loop
    print("\n=== STARTING FINE-TUNING ===")
    best_val_loss = np.inf
    logs = []
    counter = 0

    for epoch in range(num_epochs):
        print(f"\nFine-tuning Epoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, vocab, scaler)
        val_loss = validate_one_epoch(model, val_loader, device, vocab)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']}")

        scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, vocab, FINE_TUNED_CHECKPOINT)
            print("Checkpoint saved!")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break

        logs.append([epoch + 1, train_loss, val_loss, optimizer.param_groups[0]['lr']])

    # Save logs and plot
    pd.DataFrame(logs, columns=["epoch", "train_loss", "val_loss", "lr"]).to_csv(LOG_CSV, index=False)
    plot_losses([x[1] for x in logs], [x[2] for x in logs], save_path=LOSS_GRAPH_PATH)

    print(f"\n=== FINE-TUNING COMPLETE ===")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Fine-tuned model saved to: {FINE_TUNED_CHECKPOINT}")


if __name__ == "__main__":
    main()
