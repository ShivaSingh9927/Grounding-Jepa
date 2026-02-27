import os
import json
import torch
import random
import numpy as np
import wandb
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from model import GroundingJepa
from tqdm import tqdm
from eval_utils import get_clinical_metrics

class GroundingDataset(Dataset):
    def __init__(self, data_list, image_dir, tokenizer, processor, image_processor):
        self.data = data_list
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

        # Extraction logic
        question = item.get('question', "")
        answer = item.get('answer', "")

        if not question or not answer:
            out = item.get('structured_output', {})
            if 'REC_QUESTION' in out:
                question = out['REC_QUESTION']
                answer = out.get('REC_ANSWER', "")
            elif 'REC QUESTION' in out:
                question = out['REC QUESTION']
                answer = out.get('REC ANSWER', out.get('ANSWER', ""))
            else:
                question = out.get('QUESTION', "Describe the chest X-ray.")
                answer = out.get('ANSWER', "The chest X-ray appears normal.")

        question = question.replace("Question: ", "").strip()
        answer = answer.replace("Answer: ", "").strip()
        
        # Training format (Chat template)
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        image_for_processor = image.resize((224, 224))
        
        inputs = self.processor(
            text=[prompt],
            images=[image_for_processor],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=768
        )

        labels = inputs.input_ids.clone()
        
        return {
            "images": pixel_values,
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "image_grid_thw": inputs.image_grid_thw.squeeze(0),
            "labels": labels.squeeze(0),
            "raw_image": image_for_processor, # Keep for visualization if needed
            "raw_question": question,
            "raw_answer": answer
        }

def custom_collate(batch):
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        if key in ["images", "input_ids", "attention_mask", "image_grid_thw", "labels"]:
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            collated[key] = [item[key] for item in batch]
    return collated

def evaluate(model, dataloader, device, max_samples=100):
    model.eval()
    predictions = []
    ground_truths = []
    
    # We limit validation samples for speed during training if desired
    # For a full epoch end, we use more.
    count = 0
    print("Running Validation...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if count >= max_samples:
                break
                
            images = batch['images'].to(device, dtype=torch.bfloat16)
            image_grid_thw = batch['image_grid_thw'].to(device)
            # For generation, we use the reasoning prompt (User only)
            
            for i in range(len(batch['raw_question'])):
                if count >= max_samples: break
                
                # Single sample inference
                img = batch['raw_image'][i]
                prompt = batch['raw_question'][i]
                gt = batch['raw_answer'][i]
                
                # We can call model.predict_grounding directly but we already have pixels
                # Let's use a optimized version here
                pixel_values = images[i].unsqueeze(0)
                grid_thw = image_grid_thw[i].unsqueeze(0)
                
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                gen_prompt = model.reasoner.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                inputs = model.reasoner.processor(text=[gen_prompt], images=[img], return_tensors="pt").to(device)
                
                outputs = model.reasoner.generate_answer(
                    input_ids=inputs.input_ids,
                    images=pixel_values,
                    image_grid_thw=grid_thw,
                    attention_mask=inputs.attention_mask
                )
                
                pred_text = model.reasoner.tokenizer.decode(outputs[0], skip_special_tokens=True)
                predictions.append(pred_text)
                ground_truths.append(gt)
                count += 1
                
    metrics = get_clinical_metrics(predictions, ground_truths)
    model.train()
    return metrics

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize W&B
    wandb.init(project="grounding-jepa-xray", name="clinical-metrics-run-1")

    # 1. Initialize Model
    model = GroundingJepa(
        ijepa_model_id='jmtzt/ijepa_vitg16_22k',
        device=device,
        freeze_observer=True,
        use_lora=True,
        jepa_checkpoint_path='/nuvodata/User_data/shiva/Grounding-Jepa/weights/jepa_xray_vitg16.pth.tar'
    ).to(device)
    model.to(dtype=torch.bfloat16)

    # 2. Setup Data Split (Image-ID aware)
    jsonl_path = "/nuvodata/User_data/shiva/Grounding-Jepa/data/medgemma_clean_qa.jsonl"
    all_data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
            
    # Unique IDs
    unique_ids = list(set([item['image_id'] for item in all_data]))
    random.seed(42)
    random.shuffle(unique_ids)
    
    split_idx = int(0.8 * len(unique_ids))
    train_ids = set(unique_ids[:split_idx])
    val_ids = set(unique_ids[split_idx:])
    
    train_data = [d for d in all_data if d['image_id'] in train_ids]
    val_data = [d for d in all_data if d['image_id'] in val_ids]
    
    print(f"Dataset Split: {len(train_data)} train, {len(val_data)} val ({len(unique_ids)} unique images)")

    train_dataset = GroundingDataset(
        data_list=train_data,
        image_dir="/weka/kanpur/data_radiovision/paediatric_xray_dataset/physionet.org/png_images/train",
        tokenizer=model.reasoner.tokenizer,
        processor=model.reasoner.processor,
        image_processor=model.observer.processor
    )
    
    val_dataset = GroundingDataset(
        data_list=val_data,
        image_dir="/weka/kanpur/data_radiovision/paediatric_xray_dataset/physionet.org/png_images/train",
        tokenizer=model.reasoner.tokenizer,
        processor=model.reasoner.processor,
        image_processor=model.observer.processor
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate)

    # 3. Optimizer
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=1e-4)

    # 4. Training Loop
    num_epochs = 3
    best_iou = 0.0
    
    run_dir = "/nuvodata/User_data/shiva/Grounding-Jepa/checkpoints/clinical_run_1"
    os.makedirs(run_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        epoch_loss = 0
        
        for batch in loop:
            optimizer.zero_grad()
            images = batch['images'].to(device, dtype=torch.bfloat16)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image_grid_thw = batch['image_grid_thw'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(images=images, input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            wandb.log({"train_loss": loss.item()})

        avg_loss = epoch_loss/len(train_loader)
        
        # Validation
        val_metrics = evaluate(model, val_loader, device, max_samples=250)
        print(f"Epoch {epoch+1} Val Metrics: {val_metrics}")
        wandb.log({
            "val_avg_loss": avg_loss,
            "val_mean_iou": val_metrics['mean_iou'],
            "val_sensitivity": val_metrics['sensitivity'],
            "val_specificity": val_metrics['specificity'],
            "val_tp": val_metrics['tp'],
            "val_tn": val_metrics['tn'],
            "epoch": epoch + 1
        })

        # Save Last
        last_dir = os.path.join(run_dir, "last")
        os.makedirs(last_dir, exist_ok=True)
        torch.save(model.bridge.state_dict(), os.path.join(last_dir, "bridge.pt"))
        model.reasoner.model.save_pretrained(os.path.join(last_dir, "reasoner_lora"))

        # Save Best
        if val_metrics['mean_iou'] > best_iou:
            best_iou = val_metrics['mean_iou']
            best_dir = os.path.join(run_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            torch.save(model.bridge.state_dict(), os.path.join(best_dir, "bridge.pt"))
            model.reasoner.model.save_pretrained(os.path.join(best_dir, "reasoner_lora"))
            print(f"New Best mIOU: {best_iou}. Saved to {best_dir}")

    wandb.finish()

if __name__ == "__main__":
    train()
