import os
import csv
import json
import random
import base64
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import logging
from openai import OpenAI
from pydantic import BaseModel

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY",
                "<your OpenAI API key if not set as env var>"))


class Answer(BaseModel):
    option: str
    reason: str


def setup_logging(experiment_name):
    """Setup logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_filename = log_dir / f'{experiment_name}_{timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return timestamp


def encode_image(image_path):
    """Encode an image to base64 string"""
    if not os.path.exists(image_path):
        logging.error(f"Image not found: {image_path}")
        return None

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def create_prompt(context_data, reaction_a_data, reaction_b_data):
    """Create a prompt for the model with context and two possible reactions"""
    prompt = "Here is a scene from a TV show and two possible reactions.\n\n"

    # Add context scene
    prompt += "Context Scene:\n"
    prompt += f"- Image: [context frame]\n"
    if context_data.get('transcript'):
        prompt += f"- Transcript: \"{context_data.get('transcript')}\"\n\n"

    # Add reaction A
    prompt += "Reaction A:\n"
    prompt += f"- Image: [reaction A frame]\n"
    if reaction_a_data.get('emotions'):
        prompt += f"- Detected Emotion: \"{reaction_a_data.get('emotions')}\"\n\n"

    # Add reaction B
    prompt += "Reaction B:\n"
    prompt += f"- Image: [reaction B frame]\n"
    if reaction_b_data.get('emotions'):
        prompt += f"- Detected Emotion: \"{reaction_b_data.get('emotions')}\"\n\n"

    prompt += "Question: Which reaction better fits the context and why?\nAnswer:"

    return prompt


def query_model(context_data, reaction_a_data, reaction_b_data, model_name="gpt-4o"):
    """Query the OpenAI model with the context and reactions"""
    context_image = encode_image(context_data.get('middle_frame'))
    reaction_a_image = encode_image(reaction_a_data.get('middle_frame'))
    reaction_b_image = encode_image(reaction_b_data.get('middle_frame'))

    if not all([context_image, reaction_a_image, reaction_b_image]):
        return {"error": "Failed to encode one or more images"}

    # Build the message content directly with optional emotion fields
    try:
        # Start with the basic content
        content = [
            {"type": "text",
                "text": "Here is a scene from a TV show and two possible reactions.\n\nContext Scene:"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{context_image}", "detail": "low"}},
            {"type": "text",
                "text": f"- Transcript: \"{context_data.get('transcript', '')}\"\n\nReaction A:"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{reaction_a_image}", "detail": "low"}}
        ]

        # Add emotion for Reaction A only if it exists
        reaction_a_text = ""
        if reaction_a_data.get('emotions'):
            reaction_a_text = f"- Detected Emotion: \"{reaction_a_data.get('emotions')}\"\n\n"
        else:
            reaction_a_text = "\n\n"
        content.append(
            {"type": "text", "text": reaction_a_text + "Reaction B:"})

        # Add Reaction B image
        content.append({"type": "image_url", "image_url": {
                       "url": f"data:image/jpeg;base64,{reaction_b_image}", "detail": "low"}})

        # Add emotion for Reaction B only if it exists
        reaction_b_text = ""
        if reaction_b_data.get('emotions'):
            reaction_b_text = f"- Detected Emotion: \"{reaction_b_data.get('emotions')}\"\n\n"
        else:
            reaction_b_text = "\n\n"
        content.append({"type": "text", "text": reaction_b_text +
                       "Question: Which reaction better fits the context and why?\nAnswer:"})

        response = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that analyzes scenes from TV shows and determines which reaction best fits a given context."
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=1000,
            response_format=Answer,
        )

        return {
            "response": response.choices[0].message.parsed.option,
            "reason": response.choices[0].message.parsed.reason,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "error": None
        }
    except Exception as e:
        return {"error": str(e), "response": None, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def load_scene_data(scene_file_path):
    """Load scene data from a JSON file"""
    try:
        with open(scene_file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading scene file {scene_file_path}: {e}")
        return {}


def run_experiment(csv_file, results_dir, model_name="gpt-4o", sample_size=None):
    """Run the experiment on context-reaction pairs"""
    # Setup logging and create results directory
    timestamp = setup_logging("context_reaction_experiment")
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    # Load CSV data
    df = pd.read_csv(csv_file)
    logging.info(f"Loaded {len(df)} entries from {csv_file}")

    # Filter rows where criteria_overall_result is True
    if 'criteria_overall_result' in df.columns:
        df = df[df['criteria_overall_result'] == True]
        logging.info(
            f"Filtered to {len(df)} entries where criteria_overall_result is True")
    else:
        logging.warning(
            "Column 'criteria_overall_result' not found in CSV. Using all rows.")

    # Sample if needed
    if sample_size and sample_size < len(df):
        sampled_df = df.sample(sample_size, random_state=42)
        logging.info(f"Sampled {sample_size} entries for experiment")

    # Prepare results storage
    results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    # Process each entry
    for idx, row in tqdm(sampled_df.iterrows(), total=len(sampled_df)):
        video_id = row['video_id']

        # Load context scene data
        context_data = {
            'transcript': row['context_transcript'],
            'emotions': row['context_emotions'],
            'middle_frame': row['context_middle_frame']
        }

        # Load true reaction scene data
        true_reaction_data = {
            'transcript': row['reaction_transcript'],
            'emotions': row['reaction_emotions'],
            'middle_frame': row['reaction_middle_frame']
        }

        # Find a random scene from the same video to use as a distractor
        other_scenes = df[(df['video_id'] == video_id) & (
            df['reaction_scene_number'] != row['reaction_scene_number'])]

        if len(other_scenes) == 0:
            logging.warning(
                f"No other scenes found for video {video_id}, skipping entry")
            continue

        random_scene = other_scenes.sample(1).iloc[0]
        random_reaction_data = {
            'transcript': random_scene['reaction_transcript'],
            'emotions': random_scene['reaction_emotions'],
            'middle_frame': random_scene['reaction_middle_frame']
        }

        # Randomly assign true reaction to A or B
        true_is_a = random.choice([True, False])
        reaction_a_data = true_reaction_data if true_is_a else random_reaction_data
        reaction_b_data = random_reaction_data if true_is_a else true_reaction_data
        correct_answer = "A" if true_is_a else "B"

        # Query the model
        model_result = query_model(
            context_data, reaction_a_data, reaction_b_data, model_name)

        print(f"Model result: {model_result}")

        if model_result.get("error"):
            logging.error(
                f"Error querying model for entry {idx}: {model_result['error']}")
            continue

        total_prompt_tokens += model_result["prompt_tokens"]
        total_completion_tokens += model_result["completion_tokens"]
        total_tokens += model_result["total_tokens"]

        # Record result - ensure all values are JSON serializable
        result = {
            "entry_id": int(idx),  # Convert numpy.int64 to Python int
            "video_id": str(video_id),  # Ensure string
            "context_scene_number": int(row['context_scene_number']) if pd.notna(row['context_scene_number']) else None,
            "true_reaction_scene_number": int(row['reaction_scene_number']) if pd.notna(row['reaction_scene_number']) else None,
            "random_reaction_scene_number": int(random_scene['reaction_scene_number']) if pd.notna(random_scene['reaction_scene_number']) else None,
            "correct_answer": str(correct_answer),
            "model_response": str(model_result["response"]) if model_result["response"] else None,
            "model_reason": str(model_result["reason"]) if model_result["reason"] else None,
            "prompt_tokens": int(model_result["prompt_tokens"]),
            "completion_tokens": int(model_result["completion_tokens"]),
            "total_tokens": int(model_result["total_tokens"])
        }

        results.append(result)

        # Log progress
        logging.info(
            f"Processed entry {idx}, prompt tokens: {model_result['prompt_tokens']}, completion tokens: {model_result['completion_tokens']}, total tokens: {model_result['total_tokens']}")

    # Save results
    results_file = results_dir / f"context_reaction_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(
        f"Experiment completed. Prompt tokens: {total_prompt_tokens}, Completion tokens: {total_completion_tokens}, Total tokens: {total_tokens}")
    logging.info(f"Results saved to {results_file}")

    return results


def analyze_results(results):
    """Analyze the experiment results"""
    total = len(results)
    if total == 0:
        logging.warning("No results to analyze")
        return

    # Simple analysis - count how many responses contain the correct answer letter
    correct_count = 0
    for result in results:
        response = result["model_response"]
        correct_answer = result["correct_answer"]

        # Very basic check - see if the correct letter appears in the response
        if f"Reaction {correct_answer}" in response or f"reaction {correct_answer}" in response:
            correct_count += 1

    accuracy = correct_count / total
    logging.info(
        f"Simple accuracy analysis: {correct_count}/{total} = {accuracy:.2f}")

    # More detailed analysis would require parsing the model's reasoning

    return {
        "total_samples": total,
        "correct_count": correct_count,
        "accuracy": accuracy
    }


if __name__ == "__main__":
    # Configuration
    csv_file = "output/context_reaction_pairs.csv"
    results_dir = "output/vlm_experiment"
    model_name = "gpt-4o"
    sample_size = 2  # Set to None to use all entries

    # Run the experiment
    results = run_experiment(csv_file, results_dir, model_name, sample_size)

    # Analyze results
    analysis = analyze_results(results)
