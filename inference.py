import torch
from model import GroundingJepa
from PIL import Image
import os
import json

def run_test_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Initialize Model
    # We use the same configuration as training
    model = GroundingJepa(
        ijepa_model_id='jmtzt/ijepa_vitg16_22k',
        device=device,
        freeze_observer=True,
        use_lora=True
    ).to(device, dtype=torch.bfloat16)

    # 2. Load Trained Weights
    print("Loading trained weights...")
    bridge_path = "checkpoints/bridge_final.pt"
    lora_path = "checkpoints/reasoner_lora"

    if os.path.exists(bridge_path):
        model.bridge.load_state_dict(torch.load(bridge_path, map_location=device))
        print(f"Loaded bridge from {bridge_path}")
    
    if os.path.exists(lora_path):
        from peft import PeftModel
        # Since model.reasoner.model is already a PeftModel if use_lora=True,
        # we can target the underlying base model and load the trained adapter
        # Or more simply, if we want to BE SURE, we can reload it:
        model.reasoner.model = PeftModel.from_pretrained(
            model.reasoner.model.get_base_model(), 
            lora_path
        ).to(device, dtype=torch.bfloat16)
        print(f"Loaded LoRA adapter from {lora_path}")
    
    # CRITICAL: Re-integrate JEPA into the newly loaded PEFT model
    model.reasoner.integrate_jepa(model.observer, model.bridge)

    model.eval()

    # 3. Select a Sample Image and Question
    # Let's take the first one from your jsonl
    jsonl_path = "/nuvodata/User_data/shiva/Grounding-Jepa/Sample_Data/structured_medgemma_results.jsonl"
    image_dir = "/nuvodata/User_data/shiva/Grounding-Jepa/Sample_Data/images"
    
    with open(jsonl_path, 'r') as f:
        first_item = json.loads(f.readline())
    
    img_id = first_item['image_id']
    question = first_item['structured_output'].get('REC_QUESTION', "What is seen on the X-ray?")
    img_path = os.path.join(image_dir, f"{img_id}.png")

    print(f"\n--- Inference Test ---")
    print(f"Image ID: {img_id}")
    print(f"Question: {question}")
    print(f"Expected Answer: {first_item['structured_output'].get('REC_ANSWER', 'N/A')}")

    # 4. Run Prediction
    image = Image.open(img_path).convert('RGB')
    pixel_values = model.observer.processor(image, return_tensors="pt").pixel_values.to(device, dtype=torch.bfloat16)

    prediction = model.predict_grounding(pixel_values.squeeze(0), question)
    print(f"\nModel Output:\n{prediction}")

if __name__ == "__main__":
    run_test_inference()
