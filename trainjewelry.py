"""
Phi-4 Multimodal Jewelry Dataset Trainer for Modal
==================================================
A cloud-optimized script to fine-tune Microsoft's Phi-4 multimodal model
on a jewelry dataset using Modal's GPU infrastructure.
"""

import os
import json
import hashlib
import re
import torch
import csv
import modal
import logging
from pathlib import Path
from PIL import Image, ImageFile
from datetime import datetime
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# Configure logging and image handling
ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Modal app
app = modal.App("jewelry-classifier")

# Define Modal volume for persistent storage
jewelry_volume = modal.Volume.from_name("jewelry-dataset", create_if_missing=True)

# =====================================================================
# CONFIGURATION
# =====================================================================

class Config:
    # Model settings
    MODEL_NAME = "microsoft/Phi-4-multimodal-instruct"
    
    # Training settings
    BATCH_SIZE = 4
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 100
    MAX_LENGTH = 2048
    
    # Paths (adjusted for Modal)
    MOUNT_POINT = "/jewelry"
    DATA_DIR = f"{MOUNT_POINT}/images"
    OUTPUT_DIR = f"{MOUNT_POINT}/models"
    CHECKPOINTS_DIR = f"{MOUNT_POINT}/checkpoints"
    
    # GPU and Memory management
    GPU_TYPE = "A10G"  # Need A10G or better for training
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINTS_DIR, exist_ok=True)

# =====================================================================
# IMAGE DATASET PREPARATION
# =====================================================================

class JewelryImageDataset(torch.utils.data.Dataset):
    """Dataset for jewelry images with descriptions and categories"""
    
    def __init__(self, image_dir, csv_file=None, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.entries = []
        
        # If CSV file with descriptions is provided, load it
        if csv_file and os.path.exists(csv_file):
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.entries.append({
                        'image_path': os.path.join(image_dir, row['image_path']),
                        'description': row['description'],
                        'category': row['primary_category']
                    })
        else:
            # Otherwise, just collect all images
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                for img_path in self.image_dir.glob(f"**/*{ext}"):
                    self.entries.append({
                        'image_path': str(img_path),
                        'description': '',
                        'category': self._extract_category(img_path.name)
                    })
    
    def _extract_category(self, filename):
        """Extract category from filename if possible (e.g., 'ring_001.jpg' -> 'ring')"""
        if '_' in filename:
            return filename.split('_')[0].lower()
        return "unknown"
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        try:
            image = Image.open(entry['image_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
                
            return {
                'image': image,
                'description': entry['description'],
                'category': entry['category'],
                'image_path': entry['image_path']
            }
        except Exception as e:
            logger.error(f"Error loading image {entry['image_path']}: {e}")
            # Return a placeholder in case of error
            return {
                'image': Image.new('RGB', (224, 224), color=(128, 128, 128)),
                'description': "Error loading image",
                'category': "error",
                'image_path': entry['image_path']
            }

# =====================================================================
# MODEL DEFINITION
# =====================================================================

# Define the Modal image with all dependencies
def setup_environment():
    """Setup function for the Modal environment"""
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Define our Docker image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch==2.2.0", 
        "transformers==4.47.0", 
        "accelerate==0.30.0",
        "peft==0.13.0",
        "pillow",
        "tqdm",
        "scikit-learn"
    )
    .run_function(setup_environment)
)

# =====================================================================
# DATA PREPARATION
# =====================================================================

@app.function(
    image=image,
    volumes={Config.MOUNT_POINT: jewelry_volume},
)
def prepare_dataset(csv_path=None):
    """Prepare the jewelry dataset for training"""
    Config.create_dirs()
    
    # Get all image files in the dataset
    image_dir = Config.DATA_DIR
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    image_files = []
    for ext in extensions:
        image_files.extend(list(Path(image_dir).glob(f"**/*{ext}")))
    
    if not image_files:
        logger.error(f"No images found in {image_dir}")
        return {"status": "error", "message": "No images found"}
    
    # If CSV path is provided, use it
    if csv_path and os.path.exists(csv_path):
        logger.info(f"Using description CSV file: {csv_path}")
        return {
            "status": "success", 
            "message": f"Found {len(image_files)} images with descriptions",
            "image_count": len(image_files),
            "csv_path": csv_path
        }
    
    # Otherwise, just report the images found
    logger.info(f"Found {len(image_files)} images without descriptions")
    return {
        "status": "success", 
        "message": f"Found {len(image_files)} images without descriptions",
        "image_count": len(image_files),
        "csv_path": None
    }

# =====================================================================
# TRAINING FUNCTIONS
# =====================================================================

@app.function(
    image=image,
    gpu=Config.GPU_TYPE,
    timeout=14400,  # 4 hours max for training
    volumes={Config.MOUNT_POINT: jewelry_volume},
    memory=32768,  # 32GB RAM
)
def train_model(csv_path=None, epochs=None, batch_size=None, learning_rate=None):
    """Fine-tune the Phi-4 model on jewelry images"""
    from transformers import (
        AutoProcessor, 
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer
    )
    from peft import LoraConfig, get_peft_model
    import torch
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    
    # Use provided parameters or defaults from Config
    epochs = epochs or Config.EPOCHS
    batch_size = batch_size or Config.BATCH_SIZE
    learning_rate = learning_rate or Config.LEARNING_RATE
    
    Config.create_dirs()
    
    logger.info("Loading Phi-4 model and processor...")
    
    # Load processor and tokenizer
    processor = AutoProcessor.from_pretrained(
        Config.MODEL_NAME,
        trust_remote_code=True
    )
    
    # Load model with optimizations for training
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Configure LoRA for efficient fine-tuning
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    logger.info("Applying LoRA adapters for fine-tuning...")
    model = get_peft_model(model, lora_config)
    
    # Set vision encoder to trainable and activate LoRA
    logger.info("Setting up trainable parameters...")
    model.set_lora_adapter('vision')
    for param in model.model.embed_tokens_extend.image_embed.parameters():
        param.requires_grad = True
    
    # Create dataset and data loader
    dataset = JewelryImageDataset(Config.DATA_DIR, csv_path)
    logger.info(f"Dataset contains {len(dataset)} images")
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_steps=Config.WARMUP_STEPS,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,
        bf16=True,  # Use bfloat16 for more stable training
        logging_steps=10,
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda x: x,  # We'll handle batching in dataset
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # Save the model
    logger.info("Saving model...")
    trainer.save_model(Config.OUTPUT_DIR)
    
    return {
        "status": "success",
        "message": "Model training completed",
        "model_path": Config.OUTPUT_DIR
    }

# =====================================================================
# INFERENCE AND EVALUATION
# =====================================================================

@app.function(
    image=image,
    gpu=Config.GPU_TYPE,
    timeout=3600,  # 1 hour
    volumes={Config.MOUNT_POINT: jewelry_volume},
    memory=16384,  # 16GB RAM
)
def generate_descriptions(test_image_dir):
    """Generate descriptions for test jewelry images using the fine-tuned model"""
    from transformers import AutoProcessor, AutoModelForCausalLM
    import torch
    from PIL import Image
    
    # Load the fine-tuned model
    logger.info("Loading fine-tuned model...")
    
    processor = AutoProcessor.from_pretrained(
        Config.OUTPUT_DIR,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        Config.OUTPUT_DIR,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Get test images
    test_images = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        test_images.extend(list(Path(test_image_dir).glob(f"*{ext}")))
    
    if not test_images:
        logger.error(f"No test images found in {test_image_dir}")
        return {"status": "error", "message": "No test images found"}
    
    # Process test images and generate descriptions
    results = []
    
    for img_path in tqdm(test_images[:20], desc="Generating descriptions"):  # Limit to 20 for testing
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Create a user message with the image
            user_message = {
                'role': 'user',
                'content': '<|image_1|>Describe this jewelry piece in detail, including materials, style, and approximate value.',
            }
            
            # Format using the chat template
            prompt = processor.tokenizer.apply_chat_template(
                [user_message], tokenize=False, add_generation_prompt=True
            )
            
            # Process the input
            inputs = processor(
                text=prompt,
                images=[image],
                return_tensors="pt"
            ).to("cuda")
            
            # Generate description
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.2
                )
            
            # Decode the generated text
            decoded_output = processor.decode(output[0], skip_special_tokens=True)
            
            # Extract just the model's response (remove the prompt)
            response = decoded_output.split('<|assistant|>')[-1].strip()
            
            results.append({
                "image_path": str(img_path),
                "description": response
            })
            
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
            results.append({
                "image_path": str(img_path),
                "description": f"Error: {str(e)}"
            })
    
    # Save results to JSON
    output_file = os.path.join(Config.OUTPUT_DIR, "test_descriptions.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return {
        "status": "success",
        "message": f"Generated descriptions for {len(results)} images",
        "output_file": output_file
    }

# =====================================================================
# MAIN EXECUTION
# =====================================================================

@app.local_entrypoint()
def main(
    action: str = "all",  # Options: "prepare", "train", "generate", "all"
    csv_path: str = None,  # Path to CSV with image descriptions
    test_dir: str = None,  # Path to test images for generation
    epochs: int = Config.EPOCHS,
    batch_size: int = Config.BATCH_SIZE,
    learning_rate: float = Config.LEARNING_RATE
):
    """Main entry point for the jewelry classifier"""
    print("\n" + "="*70)
    print(" Phi-4 Multimodal Jewelry Model Trainer ".center(70, "="))
    print("="*70)
    
    # Execute the requested action
    if action == "prepare" or action == "all":
        print("\nPreparing dataset...")
        result = prepare_dataset.remote(csv_path)
        print(f"Dataset preparation: {result['message']}")
        
        # Save the CSV path for the next steps
        if result['csv_path']:
            csv_path = result['csv_path']
    
    if action == "train" or action == "all":
        print("\nTraining model...")
        print("This may take several hours depending on dataset size and GPU availability.")
        result = train_model.remote(
            csv_path=csv_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        print(f"Training completed: {result['message']}")
    
    if action == "generate" or action == "all":
        if not test_dir:
            test_dir = Config.DATA_DIR  # Use training data if no test dir specified
            
        print(f"\nGenerating descriptions for images in {test_dir}...")
        result = generate_descriptions.remote(test_dir)
        print(f"Generation completed: {result['message']}")
        
        if result.get('output_file'):
            print(f"Results saved to: {result['output_file']}")
    
    print("\n" + "="*70)
    print(" Process Complete ".center(70, "="))
    print("="*70)
    
    # Instructions for retrieving results
    print("\nTo retrieve your trained model and results, run:")
    print(f"modal volume get jewelry-dataset {Config.OUTPUT_DIR} ./local_models")

if __name__ == "__main__":
    main()