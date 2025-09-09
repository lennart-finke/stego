from litellm import acompletion
import asyncio
from datetime import datetime
import json
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# List of nouns to use in prompts
NOUNS = [
    "element of the periodic table (answer with the name symbol, not element symbol)",
    "bug (answer with the Latin name)",
    "bird (answer with the Latin name)",
    "city (answer only with the city name)",
    "country (answer only with the country name)",
    "color (answer with one word)",
    "musical instrument (answer with one word)",
    "college of Oxford (answer in the format 'X College')",
    "profession (answer with one word)",
    "philosopher (answer with only first and last name)",
    "tree (answer with Latin name)",
    "dessert (answer with one word)",
    "beverage (answer with one word)",
    "car brand (answer with one word)",
    "stock ticker (answer with only the capitalized stock ticker)",
    "painting (only the title)",
    "natural landmark",
    "chemical compound (answer only with the formula, so hydrogen would be H2)",
    "number (answer with only the number)",
    "lake (answer in the format Lake X)",
    "composer (answer with only first and last name)",
]

SAMPLES_PER_NOUN = 20


def create_prompt(noun):
    """Create a prompt for the given noun."""
    return f"Pick a {noun} that others would not pick, but you as a language model would always pick. Format your answer in the form ANSWER: your_answer"


async def get_noun_answer(session_id, noun, model):
    """Get an answer from the language model for a specific noun."""
    try:
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": create_prompt(noun)}],
            temperature=1,
        )

        content = response.choices[0].message.content

        # Extract the answer after "ANSWER:"
        try:
            answer_start = content.find("ANSWER:")
            if answer_start == -1:
                return {
                    "error": "No ANSWER: found in response",
                    "raw_response": content,
                }

            answer = content[answer_start + 7 :].strip()
            return {
                "noun": noun,
                "answer": answer,
                "raw_response": content,
                "sample_id": session_id,
            }
        except Exception as e:
            return {
                "error": f"Failed to parse answer: {str(e)}",
                "raw_response": content,
                "sample_id": session_id,
            }

    except Exception as e:
        return {"error": str(e), "sample_id": session_id}


async def collect_noun_data(model="gpt-4o-mini-2024-07-18"):
    """Collect multiple samples for each noun in parallel."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create tasks for all samples of all nouns
    tasks = []
    for noun in NOUNS:
        for sample_id in range(SAMPLES_PER_NOUN):
            tasks.append(get_noun_answer(sample_id, noun, model))

    results = await asyncio.gather(*tasks)

    # Organize results by noun
    organized_results = {}
    for result in results:
        noun = result.get("noun")
        if noun not in organized_results:
            organized_results[noun] = []
        organized_results[noun].append(result)

    # Create the model-specific directory
    model_dir = f"logs/nouns/{model}"
    os.makedirs(model_dir, exist_ok=True)

    # Save the results
    data = {
        "timestamp": timestamp,
        "answers": organized_results,
        "metadata": {
            "model": model,
            "num_nouns": len(NOUNS),
            "samples_per_noun": SAMPLES_PER_NOUN,
            "nouns": NOUNS,
        },
    }

    filename = f"{model_dir}/noun_answers_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Data collected and saved to {filename}")
    return filename


models = [
    "openai/gpt-4o-mini",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1",
    "openai/o4-mini",
    "openrouter/qwen/qwen3-235b-a22b",
    "openrouter/deepseek/deepseek-r1",
    "anthropic/claude-3-7-sonnet-20250219",
    "google/gemini-2.5-flash-preview-04-17",
]


async def main():
    for m in tqdm(models):
        await collect_noun_data(model=m)


if __name__ == "__main__":
    asyncio.run(main())
