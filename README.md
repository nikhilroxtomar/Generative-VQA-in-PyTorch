# 🧠 Generative Visual Question Answering (VQA) using CLIP + GPT-2

This repository contains an **end-to-end Generative Visual Question Answering (VQA) pipeline** built using **CLIP (ViT-B/32)** for image understanding and **GPT-2** for question encoding and answer generation.

The project follows a **two-stage training strategy**:
1. **Feature Extraction (Frozen Encoders)**
2. **Selective Fine-Tuning (Unfreezing Last Layers)**

This implementation is designed for **research, learning, and reproducibility**, and is intentionally written in **clean PyTorch** without excessive abstractions.

---

## 🚀 Key Features

- 🔍 **CLIP-based image encoding**
- 📝 **GPT-2 based question encoding**
- 🔗 **Gated multimodal fusion**
- 🧠 **GRU-based generative answer decoder**
- 🎯 **Teacher forcing during training**
- ⚡ **Mixed precision training (AMP)**
- 📊 **BLEU score evaluation**
- 🖼️ **Prediction visualization**
- 🧪 **Mini VQA v2 dataset creation script**

---

🎥 **YouTube Playlist (Walkthrough & Demo):**  
👉 https://www.youtube.com/playlist?list=PLHYn9gDxQOphmmPsayzbdKnWdEupt8gHG

Watch the videos to understand architecture, training, and code walkthrough step-by-step!

---

## 📁 Repository Structure
```
.
├── mini_vqa_v2.py # Dataset preparation from VQA v2
├── model.py # CLIP + GPT-2 VQA model
├── vocab.py # Answer vocabulary handling
├── preprocess.py # Dataset & vocab utilities
├── train.py # Feature extraction training
├── fine_tune.py # Selective fine-tuning
├── test.py # Testing, BLEU score & visualization
├── output/
│ ├── feature_extraction/
│ ├── fine_tuning/
│ └── vqa_test_results.csv
└── results/ # Prediction visualizations
```

---

## 🧠 Model Architecture Overview

### 🔹 Image Encoder
- **CLIP ViT-B/32**
- Frozen during feature extraction
- Last *N* layers unfrozen during fine-tuning

### 🔹 Question Encoder
- **GPT-2 (distilgpt2)**
- Mean-pooled token embeddings
- Last *N* transformer blocks unfrozen during fine-tuning

### 🔹 Fusion
- Gated fusion mechanism
- Followed by refinement MLP

### 🔹 Answer Decoder
- GRU-based auto-regressive decoder
- Word-level answer generation

---

## 🧪 Dataset

This project uses a **mini version of VQA v2**, created using:

- MS COCO 2014 images
- VQA v2 Open-Ended questions
- Most frequent human answer as ground truth
- Filtered short/ambiguous answers

### Create Dataset
```bash
python mini_vqa_v2.py
```
This generates:
```
mini_vqa_v2/
├── images/
├── metadata.csv
└── qa_pairs.json
```

## Outputs:
- 📄 vqa_test_results.csv
- 📊 Mean BLEU score (via sacrebleu)
- 🖼️ Saved qualitative visualisations

## Metrics
- BLEU Score (Generative evaluation)
- Qualitative inspection via image + text overlays

> ⚠️ Note: VQA accuracy is not ideal for generative settings, hence BLEU is used.

---

License

This project is released under the MIT License.
Feel free to use, modify, and cite for academic or educational purposes.

⭐ If you found this project useful, consider starring the repository!



