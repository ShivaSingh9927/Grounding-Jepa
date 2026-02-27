import os
import json
import math
import random
from typing import Dict, Any
import pandas as pd
import pydicom
import torch
from transformers import pipeline
from PIL import Image
import json
import re

# ============================================================
# CONFIGURATION
# ============================================================

ANNOT_CSV = "/weka/kanpur/data_radiovision/paediatric_xray_dataset/physionet.org/files/vindr-pcxr/1.0.0/annotations_train.csv"

DICOM_ROOT = "/weka/kanpur/data_radiovision/paediatric_xray_dataset/physionet.org/files/vindr-pcxr/1.0.0/train"

PNG_ROOT = "/weka/kanpur/data_radiovision/paediatric_xray_dataset/physionet.org/png_images/train"

STEP1_OUT = "/nuvodata/User_data/shiva/Grounding-Jepa/data/step1_image_text.jsonl"
STEP2_OUT = "/nuvodata/User_data/shiva/Grounding-Jepa/data/step2_image_text.jsonl"
DEBUG_OUT = "/nuvodata/User_data/shiva/Grounding-Jepa/data/debug_samples_image_text.jsonl"
MEDGEMMA_OUT = "/nuvodata/User_data/shiva/Grounding-Jepa/data/medgemma_outputs_image_text.jsonl"


# ============================================================
# STEP 1 — DICOM METADATA EXTRACTION
# ============================================================

def extract_dicom_metadata(dcm_path: str) -> Dict[str, Any]:
    try:
        dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    except Exception:
        return {"age_years": None, "gender": None}

    age_raw = getattr(dcm, "PatientAge", None)
    age_years = None
    if age_raw and age_raw.endswith("Y"):
        try:
            age_years = float(age_raw[:-1])
        except ValueError:
            pass

    gender = getattr(dcm, "PatientSex", None)

    return {
        "age_years": age_years,
        "gender": gender
    }


def step1_extract_metadata(annotation_csv, dicom_root, png_root, output_jsonl):
    df = pd.read_csv(annotation_csv)
    records_written = 0

    with open(output_jsonl, "w") as fout:
        for _, row in df.iterrows():
            image_id = row["image_id"]

            dcm_path = os.path.join(dicom_root, f"{image_id}.dicom")
            png_path = os.path.join(png_root, f"{image_id}.png")

            if not os.path.exists(dcm_path):
                continue
            if not os.path.exists(png_path):
                continue

            meta = extract_dicom_metadata(dcm_path)

            record = {
                "image_id": image_id,
                "image_path": png_path,
                "rad_id": row["rad_ID"],
                "abnormality": row["class_name"],
                "class_id": int(row["class_id"]),
                "bbox": [
                    float(row["x_min"]),
                    float(row["y_min"]),
                    float(row["x_max"]),
                    float(row["y_max"]),
                ],
                "age_years": meta["age_years"],
                "gender": meta["gender"],
                "dataset": "vindr-pcxr",
            }

            fout.write(json.dumps(record) + "\n")
            records_written += 1

    print(f"[Step 1] Records written: {records_written}")


# ============================================================
# STEP 2 — AGE STRATIFICATION
# ============================================================

AGE_BINS = [
    ("neonate", 0.0, 28 / 365),
    ("infant", 28 / 365, 1.0),
    ("early_childhood", 1.0, 5.0),
    ("middle_childhood", 5.0, 12.0),
    ("adolescent", 12.0, 18.0),
]

AGE_TOKENS = {
    "neonate": "<AGE_0_28D>",
    "infant": "<AGE_0_1>",
    "early_childhood": "<AGE_1_5>",
    "middle_childhood": "<AGE_6_12>",
    "adolescent": "<AGE_13_18>",
    "unknown": "<AGE_UNKNOWN>",
}


def assign_age_group(age_years):
    if age_years is None:
        return "unknown"
    try:
        age = float(age_years)
    except (ValueError, TypeError):
        return "unknown"
    if age < 0 or math.isnan(age):
        return "unknown"

    for group, low, high in AGE_BINS:
        if low <= age < high:
            return group
    return "unknown"


def step2_age_stratification(input_jsonl, output_jsonl):
    with open(input_jsonl) as fin, open(output_jsonl, "w") as fout:
        for line in fin:
            sample = json.loads(line)
            age_group = assign_age_group(sample.get("age_years"))

            sample["age_group"] = age_group
            sample["age_group_token"] = AGE_TOKENS[age_group]
            sample["is_pediatric"] = True

            fout.write(json.dumps(sample) + "\n")

    print(f"[Step 2] Output written: {output_jsonl}")


# ============================================================
# STEP 3 — DEBUG SAMPLING
# ============================================================

def step3_debug_sample(input_jsonl, output_jsonl, n=50):
    with open(input_jsonl) as fin:
        lines = fin.readlines()

    samples = random.sample(lines, min(n, len(lines)))

    with open(output_jsonl, "w") as fout:
        for l in samples:
            fout.write(l)

    print(f"[Step 3] Debug samples saved: {len(samples)}")


# ============================================================
# STEP 4 — MEDGEMMA INFERENCE
# ============================================================

def build_prompt(sample):
    bbox = f"[{sample['bbox'][0]}, {sample['bbox'][1]}, {sample['bbox'][2]}, {sample['bbox'][3]}]"

    return f"""
        You are a board-certified pediatric radiologist. You are helping the user to create a real looking dataset for user to create a medical grounding model. 

        You are given:
        1) A chest X-ray image.
        2) Structured ground-truth metadata.

        Your task is to generate 3–5 clinically grounded, linguistically diverse question–answer pairs.

        ====================================================
        STRICT GROUNDING RULES (MANDATORY)
        ====================================================

        1. The abnormality label below is the ONLY abnormality allowed.
        2. Do NOT introduce additional findings.
        3. Do NOT speculate beyond the label.
        4. Describe findings ONLY inside the bounding box.
        5. Questions: Must sound like a clinician asking about the image. Do NOT mention "bounding box," "box," "coordinates," or "specified region" in the question.
        6. EVERY answer must explicitly include the bounding box: {bbox}
        7. Use cautious pediatric radiology phrasing.
        8. At least one question must contain an incorrect assumption that you correct.
        9. Do NOT reuse example coordinates or invent new ones.

        ====================================================
        ANTI-REPETITION REQUIREMENTS (VERY IMPORTANT)
        ====================================================

        • Each question must start differently.
        • Vary tone and framing:
        - Observational
        - Differential-style
        - Educational
        - Clinical reasoning
        - Skeptical review
        • Avoid repeating phrases.
        • Make questions sound like they come from different clinicians.

        ====================================================
        PATIENT CONTEXT
        ====================================================

        Age group: {sample['age_group']} ({sample['age_group_token']})
        Age (years): {sample['age_years']}
        Modality: Pediatric Chest X-ray

        ====================================================
        GROUND TRUTH (AUTHORITATIVE)
        ====================================================

        Abnormality label:
        {sample['abnormality']}

        Bounding box:
        {bbox}

        Only this region may be described.

        ====================================================
        QUESTION STYLE GUIDE
        ====================================================

        Create 3–5 questions using DIFFERENT styles such as:

        • Presence of abnormality(must)
        • Localization of abnormality(must)
        • Teaching-style question
        • Incorrect premise challenge
        • Anatomical orientation question

        Do NOT make them formulaic.

        ====================================================
        OUTPUT FORMAT (STRICT JSON)
        ====================================================

        {{
        "qa_pairs": [
            {{
            "type": "...",
            "question": "...",
            "answer": "..."
            }}
        ]
        }}

        Constraints:
        - 3–5 total QA pairs
        - Each must have a distinct tone
        - Each answer must explicitly include {bbox}
        - The abnormality name must appear exactly as written above
        - No markdown
        - No explanations
        - Valid JSON only
        """





def step4_medgemma_inference(input_jsonl, output_jsonl):
    pipe = pipeline(
        "image-text-to-text",
        model="google/medgemma-27b-it",
        torch_dtype=torch.bfloat16,
        device="cuda:1",
        
    )

    with open(input_jsonl) as fin, open(output_jsonl, "w") as fout:
        for line in fin:
            sample = json.loads(line)
            prompt = build_prompt(sample)

            image = Image.open(sample["image_path"]).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        # {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            output = pipe(text=messages, max_new_tokens=3000)
            generated = output[0]["generated_text"][-1]["content"]

            fout.write(json.dumps({
                "image_id": sample["image_id"],
                "generated_text": generated
            }) + "\n")

    print("[Step 4] MedGemma inference completed")


# ============================================================
# STEP 5 — CLEAN MEDGEMMA OUTPUT
# ============================================================

def step5_clean_output(input_file, output_file):

    def extract_clean_json(raw_text):
        # Remove reasoning tokens
        clean_text = re.sub(
            r'<unused94>.*?<unused95>',
            '',
            raw_text,
            flags=re.DOTALL
        ).strip()

        # 1️⃣ Try extracting fenced JSON blocks first
        json_blocks = re.findall(
            r'```json\s*(.*?)\s*```',
            clean_text,
            re.DOTALL
        )

        # 2️⃣ If none found, try parsing entire text directly
        if not json_blocks:
            json_blocks = [clean_text]

        qa_list = []

        for block in json_blocks:
            try:
                data = json.loads(block)

                # Case A: model outputs a list directly
                if isinstance(data, list):
                    qa_list.extend(data)

                # Case B: model outputs {"qa_pairs": [...]}
                elif isinstance(data, dict) and "qa_pairs" in data:
                    qa_list.extend(data["qa_pairs"])

                # Case C: single QA dict
                elif isinstance(data, dict):
                    qa_list.append(data)

            except json.JSONDecodeError:
                continue

        return {"qa_pairs": qa_list} if qa_list else None

    # ==============================
    # Main cleaning loop
    # ==============================

    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:

        for line in f_in:
            try:
                sample = json.loads(line)
                raw_content = sample.get("generated_text", "")

                clean_data = extract_clean_json(raw_content)

                if not clean_data:
                    print(f"Warning: No valid QA found for {sample.get('image_id')}")
                    continue

                entry = {
                    "image_id": sample["image_id"],
                    "structured_output": clean_data
                }

                f_out.write(json.dumps(entry) + "\n")

            except Exception as e:
                print(f"Error processing image {sample.get('image_id')}: {e}")

    print(f"[Step 5] Clean results saved to: {output_file}")



CLEAN_OUTPUT_FILE = "/nuvodata/User_data/shiva/Grounding-Jepa/data/medgemma_outputs_image_text_clean.jsonl"
# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    step1_extract_metadata(
        ANNOT_CSV,
        DICOM_ROOT,
        PNG_ROOT,
        STEP1_OUT
    )

    step2_age_stratification(
        STEP1_OUT,
        STEP2_OUT
    )

    # step3_debug_sample(
    #     STEP2_OUT,
    #     DEBUG_OUT,
    #     n=15
    # )

    step4_medgemma_inference(
        STEP2_OUT,
        MEDGEMMA_OUT
    )

    step5_clean_output(
        MEDGEMMA_OUT,
        CLEAN_OUTPUT_FILE
    )
if __name__ == "__main__":
    main()
