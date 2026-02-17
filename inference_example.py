import torch
from model import GroundingJepa
from PIL import Image
import requests
from modules.reasoner import AutoProcessor

def run_grounding_demo():
    # 1. Setup Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Using vit_huge as per config and 7B Reasoner
    model = GroundingJepa(
        ijepa_variant='vit_huge', 
        patch_size=7,
        device=device
    ).to(device)
    
    # 2. Prepare Sample Input
    # Using a dummy image for demo purposes
    image = torch.randn(1, 3, 224, 224).to(device, dtype=torch.bfloat16)
    
    # MS-CXR style prompt
    question = "Identify pediatric lung field."
    question_template = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
    full_prompt = question_template.format(Question=question)
    
    # 3. Tokenization (Using Qwen Reasoner's components)
    inputs = model.reasoner.tokenizer(full_prompt, return_tensors="pt").to(device)
    
    # 4. Inference
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(
            images=image,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )
    
    print("Inference successful!")
    # In a real scenario, you'd use model.reasoner.model.generate() 
    # with the injected visual tokens.

if __name__ == "__main__":
    run_grounding_demo()
