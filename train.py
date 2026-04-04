import os
import json
import torch
import random
import numpy as np
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import datetime
import argparse
from model import GroundingJepa
from tqdm import tqdm
from eval_utils import get_clinical_metrics

# --- Dataset and Collate (Keep your existing classes) ---
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
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
        question = item.get('question', "").replace("Question: ", "").strip()
        answer = item.get('answer', "").replace("Answer: ", "").strip()
        if not question: question = "Describe the chest X-ray."
        
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=[prompt], images=[image.resize((224, 224))], return_tensors="pt", 
                                padding="max_length", truncation=True, max_length=768)
        return {
            "images": pixel_values,
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "image_grid_thw": inputs.image_grid_thw.squeeze(0),
            "labels": inputs.input_ids.clone().squeeze(0),
            "raw_image": image.resize((224, 224)),
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

def setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    else:
        torch.cuda.set_device(0)
        return False

def cleanup(is_distributed):
    if is_distributed:
        dist.destroy_process_group()

def evaluate(model, dataloader, device, max_samples=None):
    model.eval()
    predictions, ground_truths = [], []
    is_ddp = isinstance(model, DDP)
    m_base = model.module if is_ddp else model
    
    count = 0
    total_samples = max_samples if max_samples else len(dataloader.dataset)
    print(f"Evaluating {total_samples} samples...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if max_samples and count >= max_samples:
                break
            
            images = batch['images'].to(device, dtype=torch.bfloat16)
            for i in range(len(batch['raw_question'])):
                if max_samples and count >= max_samples: break
                
                prompt = batch['raw_question'][i]
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                gen_prompt = m_base.reasoner.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = m_base.reasoner.processor(text=[gen_prompt], images=[batch['raw_image'][i]], return_tensors="pt").to(device)
                outputs = m_base.reasoner.generate_answer(input_ids=inputs.input_ids, images=images[i].unsqueeze(0), 
                                                        image_grid_thw=batch['image_grid_thw'][i].to(device).unsqueeze(0), 
                                                        attention_mask=inputs.attention_mask)
                predictions.append(m_base.reasoner.tokenizer.decode(outputs[0], skip_special_tokens=True))
                ground_truths.append(batch['raw_answer'][i])
                count += 1
                
    metrics = get_clinical_metrics(predictions, ground_truths)
    model.train()
    return metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Train Grounding-JEPA")
    parser.add_argument("--use_rope", action="store_true", help="Enable 2D RoPE in C-Abstractor")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta value")
    parser.add_argument("--run_name", type=str, default="ddp-clinical", help="Wandb run name")
    return parser.parse_args()

def train():
    args = parse_args()
    is_distributed = setup()
    local_rank = 0 if not is_distributed else int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    
    # 1. Model Initialization
    model = GroundingJepa(
        ijepa_model_id='jmtzt/ijepa_vitg16_22k', device=device, freeze_observer=True, use_lora=True,
        use_rope=args.use_rope, rope_theta=args.rope_theta,
        jepa_checkpoint_path='/nuvodata/User_data/shiva/Grounding-Jepa/weights/jepa_xray_vitg16.pth.tar'
    ).to(device).to(dtype=torch.bfloat16)

    # Gradient Checkpointing fixes
    model.reasoner.model.gradient_checkpointing_enable()
    model.reasoner.model.config.use_cache = False # Required for checkpointing

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        m_base = model.module
    else:
        m_base = model

    # 2. Data Setup
    jsonl_path = "/nuvodata/User_data/shiva/Grounding-Jepa/data/medgemma_clean_qa.jsonl"
    all_data = [json.loads(line) for line in open(jsonl_path, 'r')]
    unique_ids = list(set([item['image_id'] for item in all_data]))
    random.seed(42)
    random.shuffle(unique_ids)
    split_idx = int(0.8 * len(unique_ids))
    train_data = [d for d in all_data if d['image_id'] in set(unique_ids[:split_idx])]
    val_data = [d for d in all_data if d['image_id'] in set(unique_ids[split_idx:])]

    train_ds = GroundingDataset(train_data, "/weka/kanpur/data_radiovision/paediatric_xray_dataset/physionet.org/png_images/train", 
                                m_base.reasoner.tokenizer, m_base.reasoner.processor, m_base.observer.processor)
    val_ds = GroundingDataset(val_data, "/weka/kanpur/data_radiovision/paediatric_xray_dataset/physionet.org/png_images/train", 
                             m_base.reasoner.tokenizer, m_base.reasoner.processor, m_base.observer.processor)

    if is_distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=6, sampler=train_sampler, collate_fn=custom_collate, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=6, shuffle=True, collate_fn=custom_collate, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=6, shuffle=True, collate_fn=custom_collate)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    best_iou = 0.0
    if local_rank == 0:
        run_suffix = "_rope" if args.use_rope else "_baseline"
        wandb.init(project="grounding-jepa-xray", name=args.run_name + run_suffix)
        run_dir = "/nuvodata/User_data/shiva/Grounding-Jepa/checkpoints/clinical_run_final"
        os.makedirs(run_dir, exist_ok=True)

    # 3. Training Loop
    accumulation_steps = 4
    for epoch in range(15):
        if is_distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        
        for i, batch in enumerate(tqdm(train_loader, disable=(is_distributed and local_rank != 0))):
            images = batch['images'].to(device, dtype=torch.bfloat16)
            images.requires_grad_(True) 
            
            outputs = model(
                images=images, 
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                image_grid_thw=batch['image_grid_thw'].to(device),
                labels=batch['labels'].to(device)
            )
            
            loss = outputs.loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                if (not is_distributed or local_rank == 0): wandb.log({"train_loss": loss.item() * accumulation_steps})

        if local_rank == 0:
            # Clear cache before evaluation to maximize memory for generation
            torch.cuda.empty_cache()
            
            # We evaluate on a 500-sample subset to keep training efficient
            metrics = evaluate(model, val_loader, device, max_samples=500)
            wandb.log({**metrics, "epoch": epoch + 1})
            
            # Save Last
            last_dir = os.path.join(run_dir, "last")
            os.makedirs(last_dir, exist_ok=True)
            torch.save(m_base.bridge.state_dict(), os.path.join(last_dir, "bridge.pt"))
            m_base.reasoner.model.save_pretrained(os.path.join(last_dir, "reasoner_lora"))

            # Save Best (Based on mean_iou)
            if metrics['mean_iou'] > best_iou:
                best_iou = metrics['mean_iou']
                best_dir = os.path.join(run_dir, "best")
                os.makedirs(best_dir, exist_ok=True)
                torch.save(m_base.bridge.state_dict(), os.path.join(best_dir, "bridge.pt"))
                m_base.reasoner.model.save_pretrained(os.path.join(best_dir, "best_reasoner_lora"))
                print(f"New Best mIOU: {best_iou:.4f} - Saved to {best_dir}")

        if is_distributed:
            dist.barrier()

    cleanup(is_distributed)

if __name__ == "__main__":
    train()