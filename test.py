import os, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from model import VQAModel
from train import VQADataset, Vocab
import sacrebleu
import matplotlib.pyplot as plt
from PIL import Image
import textwrap


# Generate answers
def evaluate(model, dataloader, question_tokenizer, vocab, device):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            questions = {
                'input_ids': batch['question_ids'].to(device),
                'attention_mask': batch['question_mask'].to(device)
            }
            answers = batch['answer_ids']  # Keep on CPU

            # Prediction
            pred_ids = model(images, questions, answer_input_ids=None)  # (B, answer_max_len)

            for i in range(len(answers)):
                question_ids = questions['input_ids'][i].cpu().tolist()
                answer_ids = answers[i].cpu().tolist()
                pred_token_ids = pred_ids[i].cpu().tolist()

                # Decode up to <eos> token
                if vocab.eos_token_id in pred_token_ids:
                    eos_idx = pred_token_ids.index(vocab.eos_token_id)
                    pred_token_ids = pred_token_ids[:eos_idx+1]

                # Decode question and answers
                question_text = question_tokenizer.decode(question_ids, skip_special_tokens=True)
                answer_text = vocab.decoder(answer_ids)
                pred_text = vocab.decoder(pred_token_ids)

                results.append({
                    'image_path': batch['image_path'][i],
                    'question': question_text,
                    'answer': answer_text,
                    'pred': pred_text
                })

    return results

# Compute Evaluation Metrics
def compute_metrics(results):
    """
    Compute BLEU score for VQA predictions.
    Uses sacrebleu for reliability on Python 3.12+.
    """
    bleu_scores = []

    for r in results:
        ref = r['answer'].strip()
        hyp = r['pred'].strip()

        # Skip empty predictions
        if len(hyp) == 0 or len(ref) == 0:
            bleu_scores.append(0.0)
            continue

        try:
            bleu = sacrebleu.sentence_bleu(hyp, [ref]).score / 100.0
        except Exception:
            bleu = 0.0

        bleu_scores.append(bleu)

    metrics = {
        "Mean BLEU": np.mean(bleu_scores)
    }

    return metrics

# Visualization
def visualize_predictions(results, save_path, num_samples=3):
    """
    Display image with question, ground-truth answer, and predicted answer.
    """
    os.makedirs(save_path, exist_ok=True)

    samples = results[:num_samples]

    for i, r in enumerate(samples):
        img = Image.open(r['image_path']).convert("RGB")

        question = textwrap.fill(f"Q: {r['question']}", width=80)
        answer = textwrap.fill(f"GT: {r['answer']}", width=80)
        pred = textwrap.fill(f"Pred: {r['pred']}", width=80)

        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis("off")

        full_text = f"{question}\n\n{answer}\n\n{pred}"
        plt.figtext(
            0.5, 0.02,
            full_text,
            wrap=True,
            horizontalalignment='center',
            fontsize=10
        )

        plt.tight_layout()
        plt.savefig(f"{save_path}/{i}.png")



# Main Testing Loop
def main():
    print()
    print("# =================================================")
    print("# VQA: Testing")
    print("# =================================================")
    print()

    # Seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Paths
    DATA_DIR = "./mini_vqa_v2"
    CSV_PATH = os.path.join(DATA_DIR, "metadata.csv")
    OUTPUT_DIR = "./output"
    CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "fine_tuning", "vqa_checkpoint.pt")
    OUTPUT_CSV = os.path.join(OUTPUT_DIR, "vqa_test_results.csv")
    SAVE_PATH = "results"

    os.makedirs(SAVE_PATH, exist_ok=True)

    # Hyperparameters
    batch_size = 2

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = "cpu"

    # Load CSV
    metadata = pd.read_csv(CSV_PATH)

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    # Get lengths from checkpoint or use defaults
    question_max_len = checkpoint.get('question_max_len', 32)
    answer_max_len = checkpoint.get('answer_max_len', 16)
    
    print(f"Loading model with: question_max_len={question_max_len}, answer_max_len={answer_max_len}")

    # Reconstruct answer processor from checkpoint
    vocab = Vocab()
    vocab.vocab = checkpoint['vocab']
    vocab.vocab_size = len(checkpoint['vocab'])
    vocab.word2idx = checkpoint['word2idx']
    vocab.idx2word = checkpoint['idx2word']

    vocab.pad_token_id = checkpoint['pad_token_id']
    vocab.bos_token_id = checkpoint['bos_token_id']
    vocab.eos_token_id = checkpoint['eos_token_id']
    vocab.unk_token_id = checkpoint['unk_token_id']

    print(f"Answer Vocab Size: {len(vocab.vocab)}")

    # Initialize model
    model = VQAModel(
        vocab_size=len(checkpoint['vocab']),
        device=device,
        question_max_len=question_max_len,
        answer_max_len=answer_max_len,
        pad_token_id=checkpoint['pad_token_id'],
        bos_token_id=checkpoint['bos_token_id'],
        eos_token_id=checkpoint['eos_token_id'],
        unk_token_id=checkpoint['unk_token_id']
    ).to(device)

    # Initialize tokenizer
    question_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    if question_tokenizer.pad_token is None:
        question_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.gpt2_model.resize_token_embeddings(len(question_tokenizer))

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Split data
    train_df, test_df = train_test_split(metadata, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
    print(f"Test size: {len(test_df)}")
    print()

    # Dataset with correct lengths
    test_dataset = VQADataset(
        test_df, DATA_DIR, question_tokenizer, vocab, 
        clip_processor=model.clip_preprocess,
        question_max_len=question_max_len,
        answer_max_len=answer_max_len
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate
    results = evaluate(model, test_loader, question_tokenizer, vocab, device)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Test results saved to {OUTPUT_CSV}")

    # Compute metrics
    print("\n# ===============================")
    print("#   Evaluation Metrics")
    print("# ===============================")
    metrics = compute_metrics(results)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Print samples
    print("\n=== SAMPLE PREDICTIONS ===")
    for i in range(min(5, len(results))):
        print(f"Sample {i+1}:")
        print(f"  Question: {results[i]['question']}")
        print(f"  Answer: {results[i]['answer']}")
        print(f"  Prediction: {results[i]['pred']}")
        print()

    # Visualize predictions
    print("\n=== VISUALIZING PREDICTIONS ===")
    visualize_predictions(results, SAVE_PATH, num_samples=10)


if __name__ == "__main__":
    main()