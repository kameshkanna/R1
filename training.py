#!/usr/bin/env python
# ADLF-based RLHF/SFT Training Script with Command Line Arguments

import os
import json
import torch
import numpy as np
import pandas as pd
import argparse
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from trl import PPOTrainer, PPOConfig, SFTTrainer, create_reference_model
from tqdm import tqdm
import wandb
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RLHF/SFT Training Script with ADLF")
    
    # Required arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the dataset file (JSONL format)")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                        help="Base model name or path")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Tokenizer name or path (if different from model)")
    parser.add_argument("--output_dir", type=str, default="./finetuned_model",
                        help="Directory to save the fine-tuned model")
    
    # SFT settings
    parser.add_argument("--run_sft", action="store_true", default=True,
                        help="Whether to run supervised fine-tuning")
    parser.add_argument("--sft_epochs", type=int, default=3,
                        help="Number of epochs for supervised fine-tuning")
    parser.add_argument("--sft_batch_size", type=int, default=4,
                        help="Batch size for supervised fine-tuning")
    parser.add_argument("--sft_learning_rate", type=float, default=2e-5,
                        help="Learning rate for supervised fine-tuning")
    parser.add_argument("--sft_checkpoint", type=str, default=None,
                        help="Path to SFT checkpoint to resume from")
    
    # RLHF settings
    parser.add_argument("--run_rlhf", action="store_true", default=True,
                        help="Whether to run RLHF after SFT")
    parser.add_argument("--ppo_epochs", type=int, default=1,
                        help="Number of epochs for PPO training")
    parser.add_argument("--ppo_batch_size", type=int, default=4,
                        help="Batch size for PPO training")
    parser.add_argument("--ppo_learning_rate", type=float, default=1e-5,
                        help="Learning rate for PPO training")
    
    # ADLF settings
    parser.add_argument("--adlf_alpha", type=float, default=0.2,
                        help="Weight for ADLF reward component")
    parser.add_argument("--use_coherence_reward", action="store_true", default=True,
                        help="Whether to use coherence in internal rewards")
    parser.add_argument("--use_length_penalty", action="store_true", default=True,
                        help="Whether to penalize very short responses")
    
    # Training settings
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Number of steps between logging updates")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Number of steps between evaluations")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Number of steps between model saves")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Whether to use mixed precision training")
    
    # Experiment tracking
    parser.add_argument("--wandb_project", type=str, default="model-finetune-adlf",
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (defaults to timestamp)")
    parser.add_argument("--use_wandb", action="store_true", default=True,
                        help="Whether to use Weights & Biases for tracking")
    
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (defaults to cuda if available)")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Run in debug mode with limited data")
    
    args = parser.parse_args()
    
    # Default tokenizer to model name if not specified
    if args.tokenizer_name is None:
        args.tokenizer_name = args.model_name
    
    # Default device to cuda if available
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Default W&B run name to timestamp
    if args.wandb_run_name is None:
        args.wandb_run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    return args

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_dataset(data_path, debug=False):
    """Load and preprocess the dataset with Q&A conversations and scores."""
    logger.info(f"Loading dataset from {data_path}")
    
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    if debug:
        logger.info("Debug mode: Using limited dataset")
        data = data[:100]  # Limit data in debug mode
    
    processed_data = []
    for item in data:
        # Extract conversation
        conversation = item.get("conversation", [])
        
        # Extract evaluation data
        score = item.get("meta_eval_score", 0)
        reason = item.get("score_reason", "")
        
        # Process each conversation into the desired format
        processed_item = {
            "conversation": conversation,
            "score": score,
            "reason": reason,
            # Convert the conversation to a formatted string for training
            "formatted_text": format_conversation(conversation),
        }
        processed_data.append(processed_item)
    
    logger.info(f"Processed {len(processed_data)} conversations")
    
    # Convert to Hugging Face Dataset
    return Dataset.from_pandas(pd.DataFrame(processed_data))

def format_conversation(conversation):
    """Format the conversation for model training."""
    formatted = ""
    for turn in conversation:
        if turn.get("role") == "human":
            formatted += f"Human: {turn.get('content', '')}\n\n"
        elif turn.get("role") == "assistant":
            formatted += f"Assistant: {turn.get('content', '')}\n\n"
    return formatted.strip()

def prepare_sft_dataset(dataset, tokenizer, max_seq_length):
    """Prepare dataset for supervised fine-tuning."""
    def tokenize_function(examples):
        return tokenizer(
            examples["formatted_text"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors="pt"
        )
    
    logger.info("Tokenizing dataset for SFT")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def compute_adlf_rewards(model_outputs, scores, tokenizer, args):
    """
    Compute rewards using ADLF (Agent Debate and Learning from Feedback).
    Incorporates both external reward signals (meta eval scores) and
    internal reward signals (model's confidence, coherence)
    """
    rewards = []
    
    for output, score in zip(model_outputs, scores):
        # Parse the output text
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
        
        # 1. External reward component: Meta evaluation score (normalized to [-1, 1])
        external_reward = (score / 5.0) * 2 - 1  # Assuming score is 0-5
        
        # 2. Internal reward components
        internal_reward = 0.0
        internal_components = 0
        
        # Coherence score (simplified placeholder)
        if args.use_coherence_reward:
            coherence_score = 0.5  # Placeholder for a real coherence metric
            internal_reward += coherence_score
            internal_components += 1
        
        # Response quality based on length
        if args.use_length_penalty:
            response_length = len(decoded_output.split())
            length_penalty = min(1.0, response_length / 100)  # Penalize very short responses
            internal_reward += length_penalty
            internal_components += 1
        
        # Average internal rewards if any components were used
        if internal_components > 0:
            internal_reward /= internal_components
        
        # Combine rewards using ADLF approach
        combined_reward = (
            (1 - args.adlf_alpha) * external_reward + 
            args.adlf_alpha * internal_reward
        )
        
        rewards.append(combined_reward)
    
    return torch.tensor(rewards, device=args.device)

def run_supervised_finetuning(model, tokenizer, dataset, args):
    """Run supervised fine-tuning on the dataset."""
    logger.info("Starting supervised fine-tuning")
    
    # Create output directory if it doesn't exist
    sft_output_dir = os.path.join(args.output_dir, "sft")
    os.makedirs(sft_output_dir, exist_ok=True)
    
    # Prepare dataset for SFT
    sft_dataset = prepare_sft_dataset(dataset, tokenizer, args.max_seq_length)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=sft_output_dir,
        per_device_train_batch_size=args.sft_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.sft_learning_rate,
        num_train_epochs=args.sft_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        fp16=args.fp16,
        report_to="wandb" if args.use_wandb else "none",
        seed=args.seed,
    )
    
    # Set up SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=sft_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Resume from checkpoint if specified
    if args.sft_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.sft_checkpoint}")
        trainer.train(resume_from_checkpoint=args.sft_checkpoint)
    else:
        trainer.train()
    
    # Save the fine-tuned model
    final_model_path = os.path.join(args.output_dir, "sft_final")
    trainer.save_model(final_model_path)
    logger.info(f"SFT model saved to {final_model_path}")
    
    return model

def run_rlhf_training(sft_model, tokenizer, dataset, args):
    """Run RLHF training using the dataset scores."""
    logger.info("Starting RLHF training")
    
    # Create output directory if it doesn't exist
    rlhf_output_dir = os.path.join(args.output_dir, "rlhf")
    os.makedirs(rlhf_output_dir, exist_ok=True)
    
    # Create a reference model (frozen copy of the fine-tuned model)
    logger.info("Creating reference model")
    ref_model = create_reference_model(sft_model)
    
    # PPO configuration
    ppo_config = PPOConfig(
        batch_size=args.ppo_batch_size,
        learning_rate=args.ppo_learning_rate,
        ppo_epochs=args.ppo_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        model_name=args.model_name,
    )
    
    # Initialize PPO trainer
    logger.info("Initializing PPO trainer")
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=sft_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )
    
    # Prepare data for RLHF
    logger.info("Preparing data for RLHF")
    queries = [item["formatted_text"].split("Assistant:")[0].strip() for item in dataset]
    scores = [item["score"] for item in dataset]
    
    # Training loop
    for epoch in range(args.ppo_epochs):
        logger.info(f"Starting PPO epoch {epoch+1}/{args.ppo_epochs}")
        
        # Process batches
        for i in tqdm(range(0, len(queries), args.ppo_batch_size)):
            batch_queries = queries[i:i+args.ppo_batch_size]
            batch_scores = scores[i:i+args.ppo_batch_size]
            
            # Tokenize inputs
            batch_inputs = tokenizer(
                batch_queries,
                padding=True,
                truncation=True,
                max_length=args.max_seq_length // 2,  # Leave room for responses
                return_tensors="pt"
            ).to(args.device)
            
            # Generate responses
            response_tensors = ppo_trainer.generate(
                batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                max_new_tokens=args.max_seq_length // 2,
            )
            
            # Compute rewards using ADLF approach
            rewards = compute_adlf_rewards(response_tensors, batch_scores, tokenizer, args)
            
            # Run PPO step
            stats = ppo_trainer.step(batch_inputs["input_ids"], response_tensors, rewards)
            
            # Log metrics
            if args.use_wandb:
                wandb.log(stats)