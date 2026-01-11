import os
import json
import random
import shutil
import pandas as pd
from tqdm import tqdm

# ==================================================
# CONFIGURATION
# ==================================================
IMAGES_DIR = "/media/nikhil/12234499 dp/R/ML_DATASET/coco2014/train2014"
VQA_DIR = "/media/nikhil/12234499 dp/R/ML_DATASET/VQAv2"
QUESTIONS_PATH = os.path.join(VQA_DIR, "v2_OpenEnded_mscoco_train2014_questions.json")
ANNOTATIONS_PATH = os.path.join(VQA_DIR, "v2_mscoco_train2014_annotations.json")
OUTPUT_DIR = "mini_vqa_v2"

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

# ==================================================
# LOAD ORIGINAL DATA
# ==================================================
print("Loading VQA v2 data...")
with open(QUESTIONS_PATH, "r") as f:
    questions = json.load(f)["questions"]
with open(ANNOTATIONS_PATH, "r") as f:
    annotations = json.load(f)["annotations"]

# Build mapping from question_id to annotation
qid_to_ann = {ann["question_id"]: ann for ann in annotations}

# ==================================================
# MERGE QUESTIONS + ANSWERS
# ==================================================
print("Merging questions and answers...")
merged_data = []
for q in tqdm(questions, total=len(questions)):
    ann = qid_to_ann.get(q["question_id"])
    if not ann:
        continue

    # Take most frequent answer
    answers = [a["answer"] for a in ann["answers"] if a["answer"].strip()]
    if not answers:
        continue

    main_answer = max(set(answers), key=answers.count)
    if len(main_answer.split()) >= 3: # Filter out very short answers
        merged_data.append({
            "image_id": q["image_id"],
            "question_id": q["question_id"],
            "question": q["question"],
            "answer": main_answer
        })

print(f"Total valid Q-A pairs: {len(merged_data)}")

# ==================================================
# COPY IMAGES + SAVE OUTPUT
# ==================================================
print("Copying selected images and saving data...")
final_data = []
for item in tqdm(merged_data):
    img_name = f"COCO_train2014_{item['image_id']:012d}.jpg"
    src_path = os.path.join(IMAGES_DIR, img_name)
    dst_path = os.path.join(OUTPUT_DIR, "images", img_name)

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        item["image_path"] = f"images/{img_name}"
        final_data.append(item)

# Save JSON + CSV
with open(os.path.join(OUTPUT_DIR, "qa_pairs.json"), "w") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

pd.DataFrame(final_data).to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)
print("Data preparation complete.")
