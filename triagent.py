import requests
import csv
import re
import json
import time
import os
import argparse
import logging
from typing import List, Dict, Tuple, Any, Optional
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tri_agent_process.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API configuration - initialized in main function
URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = None

# Rate limiting configuration
MAX_REQUESTS_PER_MINUTE = 25
MAX_TOTAL_REQUESTS = 1000
request_count = 0
start_time = time.time()

def make_api_request(payload: Dict[str, Any], max_retries: int = 3) -> Optional[str]:
    """
    Make API request with retry logic and error handling.
    """
    global request_count, start_time

    if HEADERS is None:
        return "API key not set. Please provide an API key."

    # Check if we have reached the total request limit
    if request_count >= MAX_TOTAL_REQUESTS:
        logger.error("Maximum total request limit reached. Exiting.")
        exit(1)

    for attempt in range(max_retries):
        # Rate limiting logic
        elapsed_time = time.time() - start_time
        if request_count >= MAX_REQUESTS_PER_MINUTE and elapsed_time < 60:
            wait_time = 60 - elapsed_time
            logger.info(f"Rate limit reached. Waiting for {wait_time:.2f} seconds.")
            time.sleep(wait_time)
            start_time = time.time()
            request_count = 0

        try:
            response = requests.post(URL, headers=HEADERS, json=payload, timeout=60)
            response.raise_for_status()  # Raise exception for non-200 responses
            request_count += 1

            result = response.json()
            if "choices" in result and result["choices"]:
                content = result["choices"][0].get("message", {}).get("content")
                if content:
                    return content

            # If we reach here, content was empty or missing
            logger.warning(f"Empty response received (attempt {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retrying

        except requests.exceptions.RequestException as e:
            logger.error(f"API request error (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Longer wait on network errors

            # Check if it's a rate limit error
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                logger.warning("Rate limit reached. Waiting longer before retry...")
                time.sleep(15)  # Wait longer for rate limit errors

    # If all retries failed, return a fallback message
    return "No response could be generated due to API errors."

def answer_agent(question: str) -> str:
    """Generates an initial response using Groq API with better error handling."""
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a highly accurate reasoning assistant. Answer concisely and logically."},
            {"role": "user", "content": question}
        ],
        "temperature": 0.7,
        "max_tokens": 2500
    }
    
    response = make_api_request(payload)
    if not response or response == "No response could be generated due to API errors.":
        # Provide a fallback answer if API fails
        return "I couldn't generate a response to this question due to technical difficulties."
    
    return response

def refined_answer_agent(question: str, previous_answer: str, hint: str) -> str:
    """Generates a refined response based on critic's hint with better error handling."""
    prompt = (
        f"Question: {question}\n\n"
        f"Your previous answer: {previous_answer}\n\n"
        f"Critic's hint: {hint}\n\n"
        f"Please provide an improved answer based on the critic's feedback. "
        f"Include your final answer between <answer> and </answer> tags."
    )
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a highly accurate reasoning assistant. Incorporate feedback to improve your answers."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2500
    }
    
    response = make_api_request(payload)
    if not response or response == "No response could be generated due to API errors.":
        # If API fails, return slightly modified previous answer
        return f"{previous_answer}\n\nNote: I attempted to refine this answer based on the hint: '{hint}' but was unable to."
    
    return response

def critic_agent(question: str, answer: str, ground_truth: str = None) -> str:
    """
    Evaluates the answer and provides hints with better error handling.
    Now can use ground_truth when available for better critique.
    """
    if ground_truth:
        prompt = (
            f"Given the question: '{question}', evaluate this answer: '{answer}'. "
            f"The correct answer is: '{ground_truth}'. "
            "Based on this information, provide specific, constructive hints to improve the answer if needed. "
            "If the answer is already correct and complete, explicitly state 'This answer is correct and complete.' "
            "Be concise but informative in your feedback."
        )
    else:
        prompt = (
            f"Given the question: '{question}', evaluate this answer: '{answer}'. "
            "If the answer is incorrect or incomplete, provide specific, constructive hints to improve it. "
            "If the answer is already correct and complete, explicitly state 'This answer is correct and complete.' "
            "Be concise but informative in your feedback."
        )
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a reasoning critic who provides clear, specific feedback to improve answers."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 500
    }
    
    response = make_api_request(payload)
    if not response or response == "No response could be generated due to API errors.":
        # Provide generic feedback if API fails
        return "The answer could be improved by adding more specific details and checking for any logical inconsistencies."
    
    return response

def meta_evaluator(conversation: List[Tuple[str, str]], question: str = None, ground_truth: str = None) -> Tuple[float, str]:
    """
    Scores reasoning quality with robust error handling.
    Now can use ground_truth when available for better evaluation.
    """
    # Format conversation for better readability
    conversation_text = "\n\n".join([f"[{role}]\n{text}" for role, text in conversation])
    
    # Add ground truth information if available
    ground_truth_text = f"\n\nCorrect answer: {ground_truth}" if ground_truth else ""
    
    prompt = (
        f"Evaluate the full reasoning conversation below and assign a score from 0 to 1. "
        f"Do NOT score based only on the final answer. Instead, analyze the entire conversation, including the question, the answer agent's responses, the critic's feedback, and the overall logical flow. "
        f"Penalize answers that are verbose but lack depth, contain logical inconsistencies, or fail to address key aspects of the question. "
        f"Use the following scoring scale: Perfect reasoning (0.95-1.00), Strong reasoning (0.85-0.94), Good reasoning with minor flaws (0.70-0.84), "
        f"Acceptable but needs improvement (0.50-0.69), and Poor reasoning (below 0.50). "
        f"Start your response with ONLY the numerical score (e.g., '0.78'), followed by a structured justification. "
        f"Your justification should cover: (1) Logical consistency, (2) Depth of explanation, (3) Coherence across multiple responses, "
        f"(4) The effectiveness of the critic's feedback in refining the answer, and (5) Whether the conversation truly converges towards a better answer. "
        f"Critically assess whether the reasoning process demonstrates improvement or simply restates information. "
        f"Additionally, if the answer lacks key insights expected for a high-level question, reflect that in the score."
        f"{ground_truth_text}"
        f"\n\n=== CONVERSATION ===\n{conversation_text}\n=== END ==="
    )
    
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a meta-evaluator. YOUR RESPONSE MUST START WITH ONLY A NUMERICAL SCORE FROM 0 TO 1 ON THE FIRST LINE. Then provide explanation."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 300
    }
    
    response = make_api_request(payload)
    if not response or response == "No response could be generated due to API errors.":
        # Return a default score and explanation if API fails
        return 0.5, "Unable to evaluate due to technical difficulties."
    
    # Extract score and explanation with robust parsing
    lines = response.strip().split('\n')
    score_text = lines[0].strip() if lines else "0.5"
    
    # Find the first number in the score text
    score_matches = re.findall(r'0\.\d+|\d+\.\d+|\d+', score_text)
    
    if score_matches:
        try:
            score = float(score_matches[0])
            # Ensure score is within [0,1] range
            score = min(max(score, 0), 1)
        except ValueError:
            score = 0.5
    else:
        score = 0.5
    
    # The rest is the explanation
    explanation = '\n'.join(lines[1:]) if len(lines) > 1 else "No detailed explanation provided."
    
    return score, explanation

def format_conversation_for_csv(conversation: List[Tuple[str, str]]) -> str:
    """Formats the conversation history into a string suitable for CSV."""
    formatted = []
    for role, text in conversation:
        formatted.append(f"=== {role} ===\n{text}")
    return "\n\n".join(formatted)

def extract_final_answer(response: str) -> str:
    """Extracts the final answer from the model output using <answer> tags."""
    answer_pattern = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_pattern:
        return answer_pattern.group(1).strip()
    else:
        # If no tags are found, return the entire response or a reasonable subset
        if len(response) > 500:
            return response[:500] + "... (truncated)"
        return response

def adaptive_dialectic_learning(question: str, ground_truth: str = None, max_iterations: int = 5, min_score_threshold: float = 0.95) -> Dict[str, Any]:
    """
    Executes the tri-agent refinement process with improved structure and error handling.
    Now can use ground_truth when available for better evaluation.
    """
    logger.info(f"Processing question: {question}")
    if ground_truth:
        logger.info(f"Ground truth available: {ground_truth[:50]}...")
    
    # Initial answer
    logger.info("  Generating initial answer...")
    initial_answer = answer_agent(question)
    logger.info("  Initial answer generated.")
    
    history = [initial_answer]
    
    # Start conversation with question and initial answer
    conversation = [
        ("Question", question),
        ("Answer Agent", initial_answer)
    ]
    
    current_answer = initial_answer
    best_score = 0.0
    best_answer = initial_answer
    best_explanation = ""
    
    iteration = 0
    for iteration in range(max_iterations):
        logger.info(f"  Iteration {iteration+1}/{max_iterations}:")
        
        # Get feedback from critic
        logger.info("    Getting critic feedback...")
        hint = critic_agent(question, current_answer, ground_truth)
        logger.info(f"    Critic feedback received. Length: {len(hint)}")
        
        conversation.append(("Critic Agent", hint))
        
        # Check if critic thinks answer is already correct
        if "correct and complete" in hint.lower():
            logger.info("    Critic indicates answer is correct. Performing final evaluation.")
            score, explanation = meta_evaluator(conversation, question, ground_truth)
            logger.info(f"    Final score: {score}")
            
            if score > best_score:
                best_score = score
                best_answer = current_answer
                best_explanation = explanation
            break
        
        # Generate refined answer
        logger.info("    Generating refined answer...")
        refined_answer = refined_answer_agent(question, current_answer, hint)
        logger.info(f"    Refined answer generated. Length: {len(refined_answer)}")
        
        current_answer = refined_answer
        history.append(refined_answer)
        conversation.append(("Answer Agent", refined_answer))
        
        # Evaluate current state with the FULL conversation history
        logger.info("    Evaluating current response...")
        score, explanation = meta_evaluator(conversation, question, ground_truth)
        logger.info(f"    Evaluation score: {score}")
        
        conversation.append(("Meta Evaluator", f"Score: {score}\n{explanation}"))
        
        # Track best answer
        if score > best_score:
            best_score = score
            best_answer = current_answer
            best_explanation = explanation
            logger.info("    New best answer found.")
        
        # Check if we've reached the quality threshold
        if score >= min_score_threshold:
            logger.info(f"    Quality threshold reached ({score} >= {min_score_threshold}). Stopping iterations.")
            break
    
    # Extract final answer
    final_answer = extract_final_answer(best_answer)
    
    # Format conversation for CSV
    formatted_conversation = format_conversation_for_csv(conversation)
    
    logger.info(f"  Process completed in {iteration+1} iterations with final score: {best_score:.2f}")
    
    # Return comprehensive results
    return {
        "question": question,
        "ground_truth": ground_truth,
        "initial_answer": initial_answer,
        "conversation": conversation,
        "formatted_conversation": formatted_conversation,
        "answer_history": history,
        "meta_score": best_score,
        "meta_explanation": best_explanation,
        "final_answer": final_answer,
        "iterations": iteration + 1
    }

def save_results_to_csv(filename: str, results_list: List[Dict[str, Any]]) -> None:
    """Saves results to a CSV file including the full conversation."""
    if not results_list:
        logger.warning("No results to save.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filename)) or '.', exist_ok=True)
    
    # Define fields we want in the CSV
    fieldnames = [
        "question", 
        "ground_truth",
        "initial_answer", 
        "final_answer", 
        "meta_score", 
        "meta_explanation", 
        "iterations",
        "formatted_conversation"
    ]
    
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results_list:
                # Extract only the fields we want
                row = {field: result.get(field, "") for field in fieldnames}
                writer.writerow(row)
        
        logger.info(f"Results successfully saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving CSV: {str(e)}")
    
    # Additionally save the full results as JSON for complete data
    try:
        json_filename = f"{filename.split('.')[0]}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            # Convert conversation tuples to lists for JSON serialization
            serializable_results = []
            for result in results_list:
                result_copy = result.copy()
                if 'conversation' in result_copy:
                    result_copy['conversation'] = [[role, text] for role, text in result_copy['conversation']]
                serializable_results.append(result_copy)
            
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results successfully saved to {json_filename}")
    except Exception as e:
        logger.error(f"Error saving JSON: {str(e)}")

def create_checkpoint(checkpoint_file: str, results: List[Dict[str, Any]], processed_indices: List[int]) -> None:
    """Create a checkpoint file to resume processing later."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_file)) or '.', exist_ok=True)
    
    checkpoint_data = {
        "processed_indices": processed_indices,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)
        logger.info(f"Checkpoint saved to {checkpoint_file}")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")

def load_checkpoint(checkpoint_file: str) -> List[int]:
    """Load checkpoint data to resume processing."""
    if not os.path.exists(checkpoint_file):
        return []
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        processed_indices = checkpoint_data.get("processed_indices", [])
        timestamp = checkpoint_data.get("timestamp", "unknown time")
        logger.info(f"Loaded checkpoint from {timestamp} with {len(processed_indices)} processed examples")
        return processed_indices
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        return []

def process_gsm8k_dataset(
    output_dir: str = "results",
    output_file: str = "gsm8k_adlf_results.csv",
    checkpoint_file: str = "gsm8k_checkpoint.json",
    max_examples: int = None,
    start_idx: int = 0,
    max_iterations: int = 3
) -> List[Dict[str, Any]]:
    """Process the GSM8K Socratic dataset with checkpointing."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    full_output_path = os.path.join(output_dir, output_file)
    full_checkpoint_path = os.path.join(output_dir, checkpoint_file)
    
    # Load dataset
    logger.info("Loading GSM8K Socratic dataset...")
    try:
        dataset = load_dataset("openai/gsm8k", "socratic")
        logger.info(f"Dataset loaded: {len(dataset['train'])} examples in train split")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return []
    
    # Load checkpoint if exists
    processed_indices = load_checkpoint(full_checkpoint_path)
    
    # Setup for processing
    results = []
    
    # Determine which examples to process
    examples_to_process = []
    for i in range(start_idx, len(dataset['train'])):
        if max_examples is not None and len(examples_to_process) >= max_examples:
            break
        if i not in processed_indices:
            examples_to_process.append(i)
    
    logger.info(f"Will process {len(examples_to_process)} examples from index {start_idx}")
    
    # Process each example
    try:
        for count, idx in enumerate(examples_to_process):
            example = dataset['train'][idx]
            question = example['question']
            ground_truth = example['answer']
            
            logger.info(f"\nProcessing example {count+1}/{len(examples_to_process)} (dataset index {idx})")
            
            try:
                result = adaptive_dialectic_learning(question, ground_truth, max_iterations=max_iterations)
                results.append(result)
                processed_indices.append(idx)
                
                # Save intermediate results every 5 examples
                if (count + 1) % 5 == 0 or count == len(examples_to_process) - 1:
                    logger.info(f"Saving intermediate results after {count+1} examples...")
                    save_results_to_csv(full_output_path, results)
                    create_checkpoint(full_checkpoint_path, results, processed_indices)
                
            except Exception as e:
                logger.error(f"Error processing example {idx}: {str(e)}")
                # Create a minimal result record for failed questions
                results.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "initial_answer": "Processing error occurred",
                    "final_answer": "Processing error occurred",
                    "meta_score": 0.0,
                    "meta_explanation": f"Error: {str(e)}",
                    "iterations": 0,
                    "formatted_conversation": f"Error occurred while processing: {str(e)}"
                })
                
                # Still mark as processed to avoid getting stuck
                processed_indices.append(idx)
                create_checkpoint(full_checkpoint_path, results, processed_indices)
    
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user. Saving checkpoint...")
        save_results_to_csv(full_output_path, results)
        create_checkpoint(full_checkpoint_path, results, processed_indices)
        logger.info("Checkpoint saved. You can resume later.")
    
    except Exception as e:
        logger.error(f"\nUnexpected error: {str(e)}")
        logger.info("Saving checkpoint...")
        save_results_to_csv(full_output_path, results)
        create_checkpoint(full_checkpoint_path, results, processed_indices)
        logger.info("Checkpoint saved. You can resume later.")
    
    return results

def test_api_connection(api_key: str) -> bool:
    """Test API connection with the provided key."""
    global HEADERS
    test_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    test_payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 10
    }
    
    try:
        response = requests.post(URL, headers=test_headers, json=test_payload, timeout=30)
        response.raise_for_status()
        logger.info("API connection successful")
        return True
    except Exception as e:
        logger.error(f"API connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GSM8K dataset with tri-agent learning")
    parser.add_argument("--api_key", type=str, required=True, help="Groq API key (REQUIRED)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results and logs")
    parser.add_argument("--output", type=str, default="gsm8k_adlf_results.csv", help="Output CSV filename")
    parser.add_argument("--checkpoint", type=str, default="gsm8k_checkpoint.json", help="Checkpoint filename")
    parser.add_argument("--examples", type=int, default=None, help="Maximum number of examples to process")
    parser.add_argument("--start", type=int, default=0, help="Starting index in the dataset")
    parser.add_argument("--iterations", type=int, default=3, help="Maximum iterations per example")
    
    args = parser.parse_args()
    
    # Setup API connection with provided key
    HEADERS = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json"
    }
    
    # Test API connection
    if not test_api_connection(args.api_key):
        logger.error("Failed to connect to API. Please check your API key.")
        exit(1)
    
    # Setup log file in the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "tri_agent_process.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting GSM8K processing with API key: {args.api_key[:5]}***")
    
    results = process_gsm8k_dataset(
        output_dir=args.output_dir,
        output_file=args.output,
        checkpoint_file=args.checkpoint,
        max_examples=args.examples,
        start_idx=args.start,
        max_iterations=args.iterations
    )
    
    # Display summary of results
    logger.info("\nResults Summary:")
    if results:
        total_score = sum(result.get('meta_score', 0) for result in results)
        avg_score = total_score / len(results) if results else 0
        avg_iterations = sum(result.get('iterations', 0) for result in results) / len(results) if results else 0
        
        logger.info(f"Processed {len(results)} examples")
        logger.info(f"Average score: {avg_score:.2f}")
        logger.info(f"Average iterations: {avg_iterations:.2f}")
        
        # Save summary to a dedicated file
        summary_path = os.path.join(args.output_dir, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Processed {len(results)} examples\n")
            f.write(f"Average score: {avg_score:.2f}\n")
            f.write(f"Average iterations: {avg_iterations:.2f}\n")
            
            # Add more detailed statistics
            f.write("\nDetailed Statistics:\n")
            score_distribution = {
                "Excellent (0.9-1.0)": len([r for r in results if r.get('meta_score', 0) >= 0.9]),
                "Good (0.8-0.9)": len([r for r in results if 0.8 <= r.get('meta_score', 0) < 0.9]),
                "Average (0.7-0.8)": len([r for r in results if 0.7 <= r.get('meta_score', 0) < 0.8]),
                "Fair (0.6-0.7)": len([r for r in results if 0.6 <= r.get('meta_score', 0) < 0.7]),
                "Poor (<0.6)": len([r for r in results if r.get('meta_score', 0) < 0.6])
            }
            
            for category, count in score_distribution.items():
                percentage = (count / len(results)) * 100 if results else 0
                f.write(f"{category}: {count} ({percentage:.1f}%)\n")
        
        logger.info(f"Detailed summary saved to {summary_path}")
    else:
        logger.info("No results to summarize.")