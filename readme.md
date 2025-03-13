# Advanced LLM Fine-tuning Framework

This repository contains scripts for fine-tuning large language models (LLMs) using Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF) with state-of-the-art techniques including Agent Debate and Learning from Feedback (ADLF).

## 📋 Overview

Two training scripts are provided:

1. `training.py` - A baseline implementation with core SFT and RLHF+ADLF capabilities
2. `Advanced_training.py` - An enhanced version with additional innovative techniques such as:
   - Curriculum Learning
   - Multi-Objective Optimization
   - Contrastive Learning
   - Dynamic KL Penalty
   - Self-Critique Training
   - Decision Transformer approach
   - Custom Tokenization formats

## 🔧 Requirements

```bash
pip install torch datasets transformers trl tqdm wandb pandas numpy
```

## 📊 Data Format

Both scripts expect data in JSONL format with each line containing a JSON object with:

```json
{
  "conversation": [
    {"role": "human", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ],
  "meta_eval_score": 4.5,
  "score_reason": "Explanation of the score"
}
```

Where:
- `conversation` is a list of turns between human and assistant
- `meta_eval_score` is a numerical evaluation score (typically 0-5)
- `score_reason` (optional) provides an explanation for the score

## 🚀 Basic Usage

### Base Training Script (`training.py`)

This script implements the core SFT and RLHF training pipeline with ADLF rewards.

```bash
python training.py \
  --data_path path/to/your/data.jsonl \
  --model_name gpt2 \
  --run_sft \
  --run_rlhf \
  --output_dir ./my_finetuned_model
```

### Advanced Training Script (`Advanced_training.py`)

This script extends the base functionality with innovative techniques.

```bash
python Advanced_training.py \
  --data_path path/to/your/data.jsonl \
  --model_name gpt2 \
  --use_curriculum \
  --use_multi_objective \
  --use_dynamic_kl \
  --token_format role_markers \
  --output_dir ./my_advanced_model
```

## 🔍 Key Features

### Core Features (Both Scripts)

- **Supervised Fine-Tuning (SFT)**: Traditional instruction fine-tuning
- **RLHF with ADLF**: Combines human feedback with internally generated rewards
- **W&B Integration**: Track experiments with Weights & Biases
- **Mixed Precision Training**: FP16 acceleration support
- **Checkpointing**: Resume training from saved checkpoints

### Advanced Features (Advanced_training.py only)

- **Curriculum Learning**: Train on easier examples first
  ```bash
  --use_curriculum
  ```

- **Multi-Objective Optimization**: Balance multiple training objectives
  ```bash
  --use_multi_objective \
  --fluency_weight 0.3 \
  --faithfulness_weight 0.3 \
  --quality_weight 0.4
  ```

- **Contrastive Learning**: Learn better response representations
  ```bash
  --use_contrastive \
  --contrastive_temp 0.07
  ```

- **Dynamic KL Penalty**: Adaptive KL divergence management
  ```bash
  --use_dynamic_kl \
  --kl_target 0.1
  ```

- **Custom Tokenization**: Different formats for conversation tokenization
  ```bash
  --token_format role_markers  # Options: standard, role_markers, turn_markers
  ```

- **Self-Critique Training**: Models learn to critique and improve their own outputs
  ```bash
  --use_self_critique \
  --critique_weight 0.5
  ```

- **Decision Transformer Approach**: Condition generation on target scores
  ```bash
  --use_decision_transformer
  ```

## 📚 Complete Example

### Basic Training Example

```bash
python training.py \
  --data_path data/conversations.jsonl \
  --model_name EleutherAI/pythia-1.4b \
  --run_sft \
  --sft_epochs 3 \
  --sft_batch_size 8 \
  --run_rlhf \
  --ppo_epochs 1 \
  --ppo_batch_size 4 \
  --adlf_alpha 0.3 \
  --use_coherence_reward \
  --use_length_penalty \
  --max_seq_length 1024 \
  --fp16 \
  --use_wandb \
  --wandb_project "my-llm-finetuning" \
  --output_dir ./models/my_model_v1
```

### Advanced Training Example

```bash
python Advanced_training.py \
  --data_path data/conversations.jsonl \
  --model_name meta-llama/Llama-2-7b-hf \
  --run_sft \
  --sft_epochs 2 \
  --sft_batch_size 4 \
  --run_rlhf \
  --ppo_epochs 1 \
  --ppo_batch_size 2 \
  --use_curriculum \
  --use_multi_objective \
  --fluency_weight 0.25 \
  --faithfulness_weight 0.25 \
  --quality_weight 0.5 \
  --use_dynamic_kl \
  --kl_target 0.05 \
  --token_format role_markers \
  --scheduler cosine_with_restarts \
  --cosine_cycles 2 \
  --fp16 \
  --use_wandb \
  --wandb_project "advanced-llm-finetune" \
  --output_dir ./models/advanced_model_v1
```

## 🔄 Training Pipeline

Both scripts follow a similar workflow:

1. **Data Loading**: Parse JSONL dataset and preprocess conversations
2. **Format Conversations**: Convert dialogues to the desired string format
3. **SFT Stage**: Perform supervised fine-tuning to learn from examples
4. **RLHF Stage**: Further optimize the model using reinforcement learning with:
   - PPO optimization
   - ADLF reward calculation
   - KL divergence management

## 🧠 ADLF Reward Components

The ADLF approach combines:

1. **External Rewards**: Human evaluation scores
2. **Internal Rewards**: Model-generated reward signals including:
   - Coherence metrics
   - Response length penalties
   - Fluency scores (in advanced version)
   - Faithfulness measures (in advanced version)

Combined using the formula:
```
reward = (1 - alpha) * external_reward + alpha * internal_reward
```

## 🔧 Customizing the Scripts

### Adding New Reward Signals

Modify the `compute_adlf_rewards` function to include additional reward components.

## 🚫 Common Issues and Solutions

- **Out of Memory Errors**: Reduce batch size or use gradient accumulation
  ```bash
  --sft_batch_size 2 --gradient_accumulation_steps 8
  ```

- **Slow Training**: Enable mixed precision training
  ```bash
  --fp16
  ```

- **Poor Convergence**: Try different learning rates and schedulers
  ```bash
  --sft_learning_rate 1e-5 --scheduler cosine
  ```

- **Overfitting**: Reduce training epochs or adjust hyperparameters
  ```bash
  --sft_epochs 1 --use_dynamic_kl
  ```

## 📖 References

- [RLHF: Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)
- [PPO: Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [Decision Transformer](https://arxiv.org/abs/2106.01345)
- [Self-Critique and Reflect](https://arxiv.org/abs/2303.17651)

## 📄 License

MIT

## 👥 Contributing

Contributions welcome! Please feel free to submit a Pull Request.
