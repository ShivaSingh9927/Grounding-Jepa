import torch
import torch.nn as nn
from modules.observer import IJepaObserver
from modules.bridge import CAbstractor, CAbstractorWithFinalProj
from modules.reasoner import QwenReasoner

class GroundingJepa(nn.Module):
    """
    Grounding-JEPA Architecture
    
    1. The Observer (Vision): I-JEPA Backbone (Patch-level features)
    2. The Bridge (Alignment): C-Abstractor (Spatial reduction + MLP)
    3. The Reasoner (Language): Qwen2.5-VL-7B (Decision Maker)
    """
    def __init__(self, 
                 ijepa_model_id='jmtzt/ijepa_vitg16_22k', 
                 qwen_model_id='Qwen/Qwen2.5-VL-7B-Instruct',
                 device='cuda',
                 freeze_observer=True,
                 use_lora=True,
                 use_rope=False,
                 rope_theta=10000.0,
                 jepa_checkpoint_path=None):
        super().__init__()
        
        # 1. The Observer
        self.observer = IJepaObserver(model_id=ijepa_model_id, device=device, checkpoint_path=jepa_checkpoint_path)
        if freeze_observer:
            print("Freezing Observer (I-JEPA)")
            for param in self.observer.parameters():
                param.requires_grad = False
        
        # 2. The Bridge
        in_dim = self.observer.embed_dim
        out_dim = 3584 # Default for Qwen2.5-7B
        self.bridge = CAbstractorWithFinalProj(
            in_channels=in_dim,
            out_channels=out_dim,
            use_rope=use_rope,
            rope_theta=rope_theta
        )
        if use_rope:
            print(f"RoPE enabled in Bridge (theta={rope_theta})")
        
        # 3. The Reasoner
        self.reasoner = QwenReasoner(model_id=qwen_model_id, device=device)
        self.reasoner.integrate_jepa(self.observer, self.bridge)
        
        if use_lora:
            print("Applying LoRA to Reasoner")
            from peft import LoraConfig, get_peft_model
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.reasoner.model = get_peft_model(self.reasoner.model, peft_config)
            self.reasoner.model.print_trainable_parameters()
        
    def forward(self, images, input_ids, image_grid_thw=None, attention_mask=None, labels=None):
        """
        End-to-end forward pass.
        """
        # We now pass images and text directly to the reasoner 
        # which uses the integrated JEPA-Vision tower.
        outputs = self.reasoner(
            input_ids=input_ids,
            images=images,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs

    def predict_grounding(self, image, prompt):
        """
        Inference method using integrated vision tower and processor-formatted prompt.
        """
        self.eval()
        device = next(self.parameters()).device
        
        # 1. Process image for the JEPA Observer
        pixel_values = self.observer.processor(image, return_tensors="pt").pixel_values.to(device, dtype=torch.bfloat16)

        # 2. Use Reasoner processor to format prompt and get image_grid_thw
        # Note: We resize to 224x224 to match our fixed 64 token bridge output
        image_for_qwen = image.resize((224, 224))
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # This gives us correctly expanded vision pads and the 1x16x16 grid thw
        qwen_inputs = self.reasoner.processor(
            text=self.reasoner.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            images=[image_for_qwen],
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            # Generate
            outputs = self.reasoner.generate_answer(
                input_ids=qwen_inputs.input_ids,
                images=pixel_values, # JEPA-formatted pixels
                image_grid_thw=qwen_inputs.image_grid_thw,
                attention_mask=qwen_inputs.attention_mask
            )
            
            return self.reasoner.tokenizer.decode(outputs[0], skip_special_tokens=True)
