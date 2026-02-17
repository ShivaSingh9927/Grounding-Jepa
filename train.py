import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from model import GroundingJepa
from tqdm import tqdm

class GroundingDataset(Dataset):
    def __init__(self, jsonl_path, image_dir, tokenizer, processor, image_processor):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_processor = image_processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_id = item['image_id']
        img_path = os.path.join(self.image_dir, f"{img_id}.png")
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        # Use observer's image processor
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)

        # Robust extraction of question and answer
        out = item.get('structured_output', {})
        question = ""
        answer = ""

        # Priority 1: REC_QUESTION / REC_ANSWER
        if 'REC_QUESTION' in out:
            question = out['REC_QUESTION']
            answer = out.get('REC_ANSWER', "")
        # Priority 2: REC QUESTION / REC ANSWER (space instead of underscore)
        elif 'REC QUESTION' in out:
            question = out['REC QUESTION']
            answer = out.get('REC ANSWER', out.get('ANSWER', ""))
        # Priority 3: REC as dict or list
        elif 'REC' in out:
            rec = out['REC']
            if isinstance(rec, list) and len(rec) > 0:
                question = rec[0].get('question', "")
                answer = rec[0].get('answer', "")
            elif isinstance(rec, dict):
                question = rec.get('question', "")
                answer = rec.get('answer', "")
            elif isinstance(rec, str):
                # Handle "Question: ... Answer: ..." string if present
                question = rec
                answer = out.get('ANSWER', "")
        # Priority 4: REC QUESTION AND ANSWER
        elif 'REC QUESTION AND ANSWER' in out:
            qa = out['REC QUESTION AND ANSWER']
            question = qa.get('question', "")
            answer = qa.get('answer', "")
        # Priority 5: REPORT_QUESTION_ANSWER
        elif 'REPORT_QUESTION_ANSWER' in out:
            qa = out['REPORT_QUESTION_ANSWER']
            question = qa.get('question', "")
            answer = qa.get('answer', "")
        # Fallback
        else:
            question = out.get('QUESTION', "Describe the chest X-ray.")
            answer = out.get('ANSWER', "The chest X-ray appears normal.")

        # Clean strings (remove prefixes like "Question: " if they exist)
        question = question.replace("Question: ", "").strip()
        answer = answer.replace("Answer: ", "").strip()
        
        # Use processor to format prompt correctly for both user and assistant parts
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            }
        ]
        
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # 10. Processor needs a 224x224 image to generate exactly 64 vision tokens (8x8 grid)
        # to match our JEPA+Bridge output.
        image_for_processor = image.resize((224, 224))
        
        # Use processor instead of tokenizer to properly expand vision pads (64 for 224x224)
        inputs = self.processor(
            text=[prompt],
            images=[image_for_processor],
            return_tensors="pt",
            padding="max_length",
            truncation=False, # Disable truncation here to avoid token count errors
            max_length=768
        )

        # Truncate manually if necessary to avoid the error
        if inputs.input_ids.shape[1] > 768:
            inputs.input_ids = inputs.input_ids[:, :768]
            inputs.attention_mask = inputs.attention_mask[:, :768]

        labels = inputs.input_ids.clone()
        
        return {
            "images": pixel_values,
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "image_grid_thw": inputs.image_grid_thw.squeeze(0),
            "labels": labels.squeeze(0)
        }

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Initialize Model
    model = GroundingJepa(
        ijepa_model_id='jmtzt/ijepa_vitg16_22k',
        device=device,
        freeze_observer=True,
        use_lora=True
    ).to(device)

    # Convert model to bfloat16 for efficiency
    model.to(dtype=torch.bfloat16)

    # 2. Setup Dataset
    dataset = GroundingDataset(
        jsonl_path="/nuvodata/User_data/shiva/Grounding-Jepa/Sample_Data/structured_medgemma_results.jsonl",
        image_dir="/nuvodata/User_data/shiva/Grounding-Jepa/Sample_Data/images",
        tokenizer=model.reasoner.tokenizer,
        processor=model.reasoner.processor,
        image_processor=model.observer.processor
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 3. Optimizer
    # Only optimize Bridge and LoRA parameters
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=1e-4)

    # 4. Training Loop
    model.train()
    num_epochs = 15
    save_every = 5 # Save every 5 epochs
    
    # Create a unique run directory
    base_checkpoints_dir = "/nuvodata/User_data/shiva/Grounding-Jepa/checkpoints"
    os.makedirs(base_checkpoints_dir, exist_ok=True)
    run_idx = 1
    while os.path.exists(os.path.join(base_checkpoints_dir, f"train{run_idx}")):
        run_idx += 1
    run_dir = os.path.join(base_checkpoints_dir, f"train{run_idx}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Starting training run: {run_dir}")

    for epoch in range(num_epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        epoch_loss = 0
        for batch in loop:
            optimizer.zero_grad()
            
            images = batch['images'].to(device, dtype=torch.bfloat16)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image_grid_thw = batch['image_grid_thw'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                images=images,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss/len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss}")

        # Periodic Saving
        if (epoch + 1) % save_every == 0:
            epoch_save_dir = os.path.join(run_dir, f"epoch_{epoch+1}")
            os.makedirs(epoch_save_dir, exist_ok=True)
            torch.save(model.bridge.state_dict(), os.path.join(epoch_save_dir, "bridge.pt"))
            model.reasoner.model.save_pretrained(os.path.join(epoch_save_dir, "reasoner_lora"))
            print(f"Saved periodic checkpoint to {epoch_save_dir}")

    # 5. Save final results
    final_dir = os.path.join(run_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    # Save the bridge separately
    torch.save(model.bridge.state_dict(), os.path.join(final_dir, "bridge_final.pt"))
    # Save the LoRA adapter
    model.reasoner.model.save_pretrained(os.path.join(final_dir, "reasoner_lora"))
    print(f"Training complete. Models saved in {final_dir}")

if __name__ == "__main__":
    train()
