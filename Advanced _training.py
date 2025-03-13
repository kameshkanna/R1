#!/usr/bin/env python
# Enhanced RLHF/SFT Training Script with Advanced Techniques

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
    DataCollatorForLanguageModeling,
    get_scheduler,
    get_cosine_schedule_with_warmup
)
from trl import PPOTrainer, PPOConfig, SFTTrainer, create_reference_model
from tqdm import tqdm
import wandb
import logging
from datetime import datetime
import random
from torch.nn import functional as F

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
    parser = argparse.ArgumentParser(description="Enhanced RLHF/SFT Training Script")
    
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
    
    # ===== INNOVATIVE TECHNIQUES =====
    
    # Curriculum Learning
    parser.add_argument("--use_curriculum", action="store_true", default=False,
                        help="Whether to use curriculum learning (train on easier examples first)")
    
    # Multi-Objective Optimization
    parser.add_argument("--use_multi_objective", action="store_true", default=False,
                        help="Whether to use multi-objective optimization")
    parser.add_argument("--fluency_weight", type=float, default=0.3,
                        help="Weight for fluency objective")
    parser.add_argument("--faithfulness_weight", type=float, default=0.3,
                        help="Weight for faithfulness objective")
    parser.add_argument("--quality_weight", type=float, default=0.4,
                        help="Weight for overall quality objective")
    
    # Contrastive Learning
    parser.add_argument("--use_contrastive", action="store_true", default=False,
                        help="Whether to use contrastive learning")
    parser.add_argument("--contrastive_temp", type=float, default=0.07,
                        help="Temperature for contrastive learning")
    
    # Advanced Scheduling
    parser.add_argument("--scheduler", type=str, default="linear",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"],
                        help="Learning rate scheduler")
    parser.add_argument("--cosine_cycles", type=int, default=1,
                        help="Number of cycles for cosine scheduler with restarts")
    
    # Dynamic KL Penalty
    parser.add_argument("--use_dynamic_kl", action="store_true", default=False,
                        help="Whether to use dynamic KL penalty in RLHF")
    parser.add_argument("--kl_target", type=float, default=0.1,
                        help="Target KL divergence for dynamic penalty")
    
    # Custom Tokenization
    parser.add_argument("--token_format", type=str, default="standard", 
                        choices=["standard", "role_markers", "turn_markers"],
                        help="Format for tokenizing conversations")
    
    # Self-Critique Training
    parser.add_argument("--use_self_critique", action="store_true", default=False,
                        help="Use self-critique training loop")
    parser.add_argument("--critique_weight", type=float, default=0.5,
                        help="Weight for self-critique loss")
                        
    # Decision Transformer Approach
    parser.add_argument("--use_decision_transformer", action="store_true", default=False,
                        help="Use Decision Transformer approach (condition on desired score)")
                        
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
    random.seed(seed)
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
            "formatted_text": format_conversation(conversation),
        }
        processed_data.append(processed_item)
    
    logger.info(f"Processed {len(processed_data)} conversations")
    
    # Convert to Hugging Face Dataset
    return Dataset.from_pandas(pd.DataFrame(processed_data))

def format_conversation(conversation, token_format="standard"):
    """Format the conversation based on specified format."""
    if token_format == "standard":
        formatted = ""
        for turn in conversation:
            if turn.get("role") == "human":
                formatted += f"Human: {turn.get('content', '')}\n\n"
            elif turn.get("role") == "assistant":
                formatted += f"Assistant: {turn.get('content', '')}\n\n"
        return formatted.strip()
    
    elif token_format == "role_markers":
        formatted = ""
        for turn in conversation:
            if turn.get("role") == "human":
                formatted += f"<human>{turn.get('content', '')}</human>"
            elif turn.get("role") == "assistant":
                formatted += f"<assistant>{turn.get('content', '')}</assistant>"
        return formatted.strip()
    
    elif token_format == "turn_markers":
        formatted = ""
        for i, turn in enumerate(conversation):
            if turn.get("role") == "human":
                formatted += f"<turn{i}><human>{turn.get('content', '')}</human></turn{i}>"
            elif turn.get("role") == "assistant":
                formatted += f"<turn{i}><assistant>{turn.get('content', '')}</assistant></turn{i}>"
        return formatted.strip()
    
    else:
        raise ValueError(f"Unknown token format: {token_format}")

def prepare_sft_dataset(dataset, tokenizer, max_seq_length, args):
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
    
    # Apply curriculum learning if enabled
    if args.use_curriculum:
        logger.info("Applying curriculum learning - sorting by complexity")
        # Sort by conversation length as a simple proxy for complexity
        tokenized_dataset = tokenized_dataset.sort("length")
    
    return tokenized_dataset

def compute_adlf_rewards(model_outputs, scores, tokenizer, args, ref_model=None):
    """
    Enhanced reward computation using ADLF and additional metrics
    """
    rewards = []
    
    for output, score in zip(model_outputs, scores):
        # Parse the output text
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
        
        # 1. External reward component: Meta evaluation score (normalized to [-1, 1])
        external_reward = (score / 5.0) * 2 - 1  # Assuming score is 0-5
        
        # 2. Internal reward components
        internal_reward = 0.0
        weights = []
        component_rewards = []
        
        # Multi-objective rewards if enabled
        if args.use_multi_objective:
            # Fluency reward (placeholder - implement with actual fluency metrics)
            fluency_score = min(1.0, len(set(decoded_output.split())) / 100)  # Lexical diversity
            component_rewards.append(fluency_score)
            weights.append(args.fluency_weight)
            
            # Faithfulness reward (if reference model provided)
            if ref_model is not None:
                # Simple cosine similarity to reference model output as faithfulness metric
                # Placeholder - implement with better faithfulness metrics in production
                faithfulness_score = 0.7  # Simplified placeholder
                component_rewards.append(faithfulness_score)
                weights.append(args.faithfulness_weight)
            
            # Quality reward (derived from meta_eval_score)
            quality_score = score / 5.0
            component_rewards.append(quality_score)
            weights.append(args.quality_weight)
            
            # Compute weighted average
            if sum(weights) > 0:
                internal_reward = sum(w * r for w, r in zip(weights, component_rewards)) / sum(weights)
        else:
            # Coherence score (simplified placeholder)
            if args.use_coherence_reward:
                coherence_score = 0.5  # Placeholder for a real coherence metric
                component_rewards.append(coherence_score)
                weights.append(1.0)
            
            # Response quality based on length
            if args.use_length_penalty:
                response_length = len(decoded_output.split())
                length_penalty = min(1.0, response_length / 100)  # Penalize very short responses
                component_rewards.append(length_penalty)
                weights.append(1.0)
            
            # Average rewards if any components were used
            if len(weights) > 0:
                internal_reward = sum(component_rewards) / len(weights)
        
        # Combine rewards using ADLF approach
        combined_reward = (
            (1 - args.adlf_alpha) * external_reward + 
            args.adlf_alpha * internal_reward
        )
        
        rewards.append(combined_reward)
    
    return torch.tensor(rewards, device=args.device)

def create_scheduler(optimizer, num_training_steps, args):
    """Create a learning rate scheduler based on args."""
    if args.scheduler == "linear":
        return get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps
        )
    elif args.scheduler == "cosine":
        return get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps
        )
    elif args.scheduler == "cosine_with_restarts":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=args.cosine_cycles
        )
    elif args.scheduler == "polynomial":
        return get_scheduler(
            "polynomial",
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps
        )
    elif args.scheduler == "constant":
        return get_scheduler(
            "constant",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

def contrastive_loss(embeddings, labels, temperature=0.07):
    """
    Compute contrastive loss (SimCLR approach)
    
    Args:
        embeddings: tensor of shape (batch_size, embedding_dim)
        labels: tensor of shape (batch_size) with labels (same label = similar)
        temperature: temperature parameter for softmax
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(embeddings, embeddings.transpose(0, 1)) / temperature
    
    # Create masks for positive and negative pairs
    batch_size = embeddings.shape[0]
    labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
    eye_mask = torch.eye(batch_size, device=embeddings.device).bool()
    
    # Remove self-similarity
    labels_matrix = labels_matrix.masked_fill(eye_mask, False)
    
    # Compute loss
    positives = similarity_matrix[labels_matrix].reshape(batch_size, -1)
    negatives = similarity_matrix[~labels_matrix].reshape(batch_size, -1)
    
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(batch_size, device=embeddings.device).long()
    
    return F.cross_entropy(logits, labels)

def run_supervised_finetuning(model, tokenizer, dataset, args):
    """Run supervised fine-tuning with advanced techniques."""
    logger.info("Starting supervised fine-tuning with advanced techniques")
    
    # Create output directory if it doesn't exist
    sft_output_dir = os.path.join(args.output_dir, "sft")
    os.makedirs(sft_output_dir, exist_ok=True)
    
    # Prepare dataset for SFT
    sft_dataset = prepare_sft_dataset(dataset, tokenizer, args.max_seq_length, args)
    
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
        # Add scheduler if not using default
        lr_scheduler_type=args.scheduler if args.scheduler != "cosine_with_restarts" else "cosine",
    )
    
    # Custom training loop if using specialized techniques
    if args.use_decision_transformer or args.use_self_critique or args.use_contrastive:
        logger.info("Using custom training loop for advanced techniques")
        
        # TODO: Implement custom training loop here
        # This is a placeholder for the implementation
        # For now, we'll fall back to the standard SFT trainer
        
        logger.warning("Custom training loop not yet implemented, falling back to standard SFT")
        
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
    """Run RLHF training with advanced techniques."""
    logger.info("Starting RLHF training with advanced techniques")
    
    # Create output directory if it doesn't exist
    rlhf_output_dir = os.path.join(args.output_dir, "rlhf")
    os.makedirs(rlhf_output_dir, exist_ok=True)
    
    # Create a reference model (frozen copy of the fine-tuned model)
    logger.info("Creating reference model")
    ref_model = create_reference_model(sft_model)
    
    # PPO configuration with advanced settings
    ppo_config = PPOConfig(
        batch_size=args.ppo_batch_size,
        learning_rate=args.ppo_learning_rate,
        ppo_epochs=args.ppo_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        model_name=args.model_name,
        # Add dynamic KL penalty if enabled
        kl_penalty="adaptive" if args.use_dynamic_kl else "none",
        kl_target=args.kl_target if args.use_dynamic_kl else None,
        # Other PPO settings
        adap_kl_ctrl=args.use_dynamic_kl,
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
    
    # Decision Transformer formatting: prepend target score if enabled
    if args.use_decision_transformer:
        logger.info("Using Decision Transformer approach - prepending target scores")
        # For Decision Transformer, we prepend the target score to guide generation
        # This conditions the model on the desired performance level
        queries = [f"[Target Score: {item['score']:.1f}] " + 
                  item["formatted_text"].split("Assistant:")[0].strip() 
                  for item in dataset]
    else:
        queries = [item["formatted_text"].split("Assistant:")[0].strip() for item in dataset]
    
    scores = [item["score"] for item in dataset]
    
    # Apply curriculum learning if enabled
    if args.use_curriculum:
        logger.info("Applying curriculum learning for RLHF")
        # Sort by scores (easier examples first, then harder)
        sort_indices = np.argsort(scores)
        queries = [queries[i] for i in sort_indices]
        scores = [scores[i] for i in sort_indices]
    
    # Training loop
    for epoch in range(args.ppo_epochs):
        logger.info(f"Starting PPO epoch {epoch+1}/{args.ppo_epochs}")
        
        # Dynamic KL coefficient adjustment if enabled
        if args.use_dynamic_kl:
            logger.info(f"Current KL penalty coefficient: {ppo_trainer.kl_ctl.value}")
        
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
            
            # Self-critique loop if enabled
            if args.use_self_critique:
                logger.info("Generating self-critiques for responses")
                
                # Decode responses
                responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
                
                # Generate critiques (in a real implementation, this would be a separate model or prompt)
                critique_prompts = [f"Critique the following response: {r}" for r in responses]
                critique_inputs = tokenizer(
                    critique_prompts,
                    padding=True,
                    truncation=True,
                    max_length=args.max_seq_length // 2,
                    return_tensors="pt"
                ).to(args.device)
                
                # Generate critiques
                critique_tensors = ppo_trainer.generate(
                    critique_inputs["input_ids"],
                    attention_mask=critique_inputs["attention_mask"],
                    max_new_tokens=args.max_seq_length // 4,
                )
                
                # Use critiques to adjust rewards
                critiques = [tokenizer.decode(c, skip_special_tokens=True) for c in critique_tensors]
                
                # Compute rewards with critiques
                rewards = compute_adlf_rewards(response_tensors, batch_scores, tokenizer, args, ref_model)
                
                # Generate improved responses based on critiques
                improved_prompts = [f"{q}\nCritique: {c}" for q, c in zip(batch_queries, critiques)]
                improved_inputs = tokenizer(
                    improved_prompts,
                    padding=True,
                    truncation=True,
                    max_length=args.max_seq_length // 2,
                    return_tensors="pt"
                ).to(args.device)
                
                # Generate improved responses
                improved_tensors = ppo_trainer.generate(
                    improved_inputs["input_ids"],
                    attention_mask=improved_inputs["attention_mask"],
                    max_new_tokens=args.max_seq_length // 2,
                )
                
                # Compute rewards for improved responses
                improved_rewards = compute_adlf_rewards(improved_tensors, batch_scores, tokenizer, args, ref_model)
                
                # Run PPO step on both original and improved responses
                stats1 = ppo_trainer.step(batch_inputs["input_ids"], response_tensors, rewards)
                stats2 = ppo_trainer.step(improved_inputs["input_ids"], improved_tensors, improved_rewards)
                
                # Combine stats
                stats = {k: (stats1.get(k, 0) + stats2.get(k, 0)) / 2 for k in set(stats1) | set(stats2)}
                
            else:
                # Standard reward computation
                rewards = compute_adlf_rewards(response_tensors, batch_scores, tokenizer, args, ref_model)
                
                # Run PPO step
                stats = ppo_trainer.step(batch_inputs["input_ids"], response_tensors, rewards)
            
            # Log metrics
            if args.use_wandb:
                wandb.log(stats)
            
            # Save checkpoint periodically
            if i % (args.save_steps * args.ppo_batch_size) == 0 and i > 0:
                checkpoint_path = os.path.join(rlhf_output_dir, f"checkpoint-{epoch}-{i}")
                ppo_trainer.save_pretrained(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save the final RLHF model
    final_model_path = os.path.join(args.output_dir, "rlhf_final")
    ppo_trainer.save_pretrained(final_model_path)
    logger.info(f"Final RLHF model saved to {final_model_path}")
    
    return ppo_trainer.model

def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize tracking
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    # Set random seed
    set_seed(args.seed)
    
    logger.info(f"Using device: {args.device}")
    logger.info(f"Using innovative techniques: " + 
                f"Curriculum Learning: {args.use_curriculum}, " +
                f"Multi-Objective: {args.use_multi_objective}, " +
                f"Contrastive Learning: {args.use_contrastive}, " +
                f"Self-Critique: {args.use_self_critique}, " +
                f"Decision Transformer: {args.use_decision_transformer}, " +
                f"Dynamic KL: {args.use_dynamic_kl}")
    
    # Load tokenizer and model
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(args.device)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens for role-based formatting if needed
    if args.token_format in ["role_markers", "turn_markers"]:
        logger.info(f"Adding special tokens for {args.token_format}")
        special_tokens = []
        if args.token_format == "role_markers":
            special_tokens = ["<human>", "</human>", "<assistant>", "</assistant>"]
        elif args.token_format == "turn_markers":
            # Add turn markers for a reasonable number of turns
            # Add turn markers for a reasonable number of turns
            for i in range(20):  # Support up to 20 turns
                special_tokens.extend([f"<turn{i}>", f"</turn{i}>"])
            special_tokens.extend(["<human>", "</human>", "<assistant>", "</assistant>"])
        
        special_tokens_dict = {"additional_special_tokens": special_tokens}
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
    
    # Load dataset
    logger.info(f"Loading dataset from: {args.data_path}")
    dataset = load_dataset(args.data_path, debug=args.debug)
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Format dataset with specified token format
    if args.token_format != "standard":
        logger.info(f"Reformatting dataset with {args.token_format} token format")
        dataset = dataset.map(
            lambda x: {"formatted_text": format_conversation(x["conversation"], args.token_format)}
        )
    
    # Run supervised fine-tuning if enabled
    if args.run_sft:
        model = run_supervised_finetuning(model, tokenizer, dataset, args)
    
    # Run RLHF training if enabled
    if args.run_rlhf:
        model = run_rlhf_training(model, tokenizer, dataset, args)
    
    logger.info(f"Training complete. Model saved to: {args.output_dir}")
    
    # Finish tracking
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()