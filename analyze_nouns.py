import json
import os
import glob
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def get_leaf_directories(root_dir):
    """Get all leaf directories that contain JSON files."""
    leaf_dirs = []
    for root, dirs, files in os.walk(root_dir):
        if any(f.endswith(".json") for f in files):
            leaf_dirs.append(root)
    return leaf_dirs


def load_data(model_dir):
    """Load all JSON files from a model directory."""
    files = glob.glob(f"{model_dir}/*.json")
    if not files:
        return None

    # Get the most recent file
    latest_file = max(files, key=os.path.getctime)
    with open(latest_file, "r") as f:
        return json.load(f)


def calculate_similarity(answers1, answers2=None):
    """Calculate similarity between two sets of answers.
    If answers2 is not provided, returns 1.0 (self-similarity).
    Otherwise calculates frequency-weighted overlap of answers."""
    if not answers1:
        return 0

    if answers2 is None:
        return 1.0  # Self-similarity is always 1.0

    if not answers2:
        return 0

    # Calculate frequency distributions
    counts1 = Counter(answers1)
    counts2 = Counter(answers2)

    # Get all unique answers
    all_answers = set(answers1) | set(answers2)

    if len(all_answers) == len(answers1) + len(answers2):
        return 0  # No overlap at all

    # Calculate normalized frequencies
    total1 = len(answers1)
    total2 = len(answers2)

    # Calculate weighted overlap
    overlap_score = 0
    for answer in all_answers:
        freq1 = counts1.get(answer, 0) / total1
        freq2 = counts2.get(answer, 0) / total2
        # Weight by the minimum frequency (how much they agree on this answer)
        overlap_score += min(freq1, freq2)

    return overlap_score


def calculate_self_entropy(answers):
    """Calculate self-entropy of a list of answers."""
    if not answers:
        return 0
    counts = Counter(answers)
    probs = np.array(list(counts.values())) / len(answers)
    return entropy(probs)


def create_similarity_matrix(df, noun=None, diagonal_metric="entropy"):
    """Create a matrix of similarities between models.
    Diagonal contains either self-entropy or most common answer frequency.
    Off-diagonal contains similarity scores.

    Args:
        df: DataFrame with results
        noun: Optional noun to filter by
        diagonal_metric: Either 'entropy' or 'frequency' for diagonal values
    """
    if noun:
        df = df[df["noun"] == noun]

    models = sorted(df["model"].unique())
    n_models = len(models)
    matrix = np.zeros((n_models, n_models))

    # Fill matrix with similarities and diagonal metric
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            # Get all answers for both models
            answers1 = df[df["model"] == model1]["answer"].tolist()
            answers2 = df[df["model"] == model2]["answer"].tolist()

            if i == j:
                # Put selected metric on diagonal
                if diagonal_metric == "entropy":
                    matrix[i, j] = calculate_self_entropy(answers1)
                else:  # frequency
                    counter = Counter(answers1)
                    most_common = counter.most_common(1)[0]
                    matrix[i, j] = most_common[1] / len(answers1)
            else:
                # Put similarity score on off-diagonal
                matrix[i, j] = calculate_similarity(answers1, answers2)

    return matrix, models


def plot_similarity_matrices(df, diagonal_metric="entropy"):
    """Create and save similarity matrix plots.

    Args:
        df: DataFrame with results
        diagonal_metric: Either 'entropy' or 'frequency' for diagonal values
    """
    # Create directory for matrix plots
    matrix_dir = "logs/nouns/analysis/matrices"
    os.makedirs(matrix_dir, exist_ok=True)

    # First pass: calculate max entropy across all nouns if needed
    max_entropy = 0
    if diagonal_metric == "entropy":
        for noun in df["noun"].unique():
            matrix, _ = create_similarity_matrix(df, noun, diagonal_metric)
            max_entropy = max(max_entropy, np.max(np.diag(matrix)))

    # Plot average matrix across all nouns
    avg_matrix, models = create_similarity_matrix(df, diagonal_metric=diagonal_metric)
    plt.figure(figsize=(12, 10))

    # Create masks for diagonal and off-diagonal
    diag_mask = np.eye(avg_matrix.shape[0], dtype=bool)
    off_diag_mask = ~diag_mask

    # First plot similarity scores (off-diagonal)
    sns.heatmap(
        avg_matrix,
        xticklabels=models,
        yticklabels=models,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        square=True,
        vmin=0,
        vmax=1,
        mask=diag_mask,
        cbar_kws={"label": "Similarity"},
    )

    # Then plot diagonal metric with different color scale
    sns.heatmap(
        avg_matrix,
        xticklabels=models,
        yticklabels=models,
        cmap="Oranges",
        annot=True,
        fmt=".2f",
        square=True,
        vmin=0,
        vmax=max_entropy if diagonal_metric == "entropy" else 1,
        mask=off_diag_mask,
        cbar_kws={
            "label": "Self-Entropy"
            if diagonal_metric == "entropy"
            else "Most Common Frequency"
        },
    )

    plt.title(
        f'Average Model Similarity and {"Self-Entropy" if diagonal_metric == "entropy" else "Most Common Answer Frequency"}'
    )
    plt.tight_layout()
    plt.savefig(
        f"{matrix_dir}/average_similarity_matrix.png", bbox_inches="tight", dpi=300
    )
    plt.close()

    # Plot matrices for each noun
    for noun in df["noun"].unique():
        matrix, models = create_similarity_matrix(df, noun, diagonal_metric)
        plt.figure(figsize=(12, 10))

        # Create masks for diagonal and off-diagonal
        diag_mask = np.eye(matrix.shape[0], dtype=bool)
        off_diag_mask = ~diag_mask

        # First plot similarity scores (off-diagonal)
        sns.heatmap(
            matrix,
            xticklabels=models,
            yticklabels=models,
            cmap="viridis",
            annot=True,
            fmt=".2f",
            square=True,
            vmin=0,
            vmax=1,
            mask=diag_mask,
            cbar_kws={"label": "Similarity"},
        )

        # Then plot diagonal metric with different color scale
        sns.heatmap(
            matrix,
            xticklabels=models,
            yticklabels=models,
            cmap="Oranges",
            annot=True,
            fmt=".2f",
            square=True,
            vmin=0,
            vmax=max_entropy if diagonal_metric == "entropy" else 1,
            mask=off_diag_mask,
            cbar_kws={
                "label": "Self-Entropy"
                if diagonal_metric == "entropy"
                else "Most Common Frequency"
            },
        )

        plt.title(
            f'Model Similarity and {"Self-Entropy" if diagonal_metric == "entropy" else "Most Common Answer Frequency"} for {noun}'
        )
        plt.tight_layout()
        plt.savefig(
            f"{matrix_dir}/similarity_matrix_{noun}.png", bbox_inches="tight", dpi=300
        )
        plt.close()


def analyze_model_consistency():
    """Analyze consistency within and across models."""
    # Get all leaf directories containing JSON files
    model_dirs = get_leaf_directories("logs/nouns")
    if not model_dirs:
        print("No data directories found in logs/nouns")
        return

    # Initialize results storage
    results = {
        "model": [],
        "noun": [],
        "answer": [],
        "num_unique_answers": [],
        "most_common_answer": [],
        "most_common_frequency": [],
    }

    # Load data for all models
    model_data = {}
    for model_dir in tqdm(model_dirs, desc="Loading model data"):
        data = load_data(model_dir)
        if data:
            # Use the directory name as the model identifier
            model_name = os.path.basename(model_dir)
            model_data[model_name] = data

    if not model_data:
        print("No valid data files found in any directory")
        return

    # Get the list of nouns from the first model's data
    nouns = model_data[list(model_data.keys())[0]]["metadata"]["nouns"]

    # Calculate statistics for each noun and model
    for noun in nouns:
        # Calculate per-model statistics
        for model, data in model_data.items():
            if noun in data["answers"]:
                answers = [a["answer"] for a in data["answers"][noun] if "answer" in a]

                # Get most common answer and its frequency
                counter = Counter(answers)
                most_common = counter.most_common(1)[0]

                # Store all answers
                for answer in answers:
                    results["model"].append(model)
                    results["noun"].append(noun.split(" ")[0])
                    results["answer"].append(answer.lower())
                    results["num_unique_answers"].append(len(counter))
                    results["most_common_answer"].append(most_common[0])
                    results["most_common_frequency"].append(
                        most_common[1] / len(answers)
                    )

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df = df[df["model"] != "claude-3-7-sonnet-20250219"]

    # Create output directory if it doesn't exist
    os.makedirs("logs/nouns/analysis", exist_ok=True)

    # Save detailed results
    df.to_csv("logs/nouns/analysis/analysis_results.csv", index=False)

    # Generate summary statistics
    summary = (
        df.groupby("noun")
        .agg({"num_unique_answers": "mean", "most_common_frequency": "mean"})
        .round(3)
    )

    summary.to_csv("logs/nouns/analysis/summary_statistics.csv")

    # Create similarity matrices with entropy on diagonal
    plot_similarity_matrices(df, diagonal_metric="entropy")

    # Create similarity matrices with most common frequency on diagonal
    plot_similarity_matrices(df, diagonal_metric="frequency")

    # Create visualizations
    plt.figure(figsize=(15, 10))

    # Plot 1: Most common answer frequency by noun
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x="noun", y="most_common_frequency")
    plt.xticks(rotation=45, ha="right")
    plt.title("Most Common Answer Frequency by Noun")
    plt.tight_layout()

    # Plot 2: Number of unique answers by noun
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x="noun", y="num_unique_answers")
    plt.xticks(rotation=45, ha="right")
    plt.title("Number of Unique Answers by Noun")
    plt.tight_layout()

    # Plot 3: Model comparison - Most common answer frequency
    plt.subplot(2, 2, 3)
    model_summary = (
        df.groupby("model")
        .agg({"most_common_frequency": "mean", "num_unique_answers": "mean"})
        .round(3)
    )

    sns.barplot(data=df, x="model", y="most_common_frequency")
    plt.xticks(rotation=45, ha="right")
    plt.title("Average Most Common Answer Frequency by Model")
    plt.tight_layout()

    # Plot 4: Model comparison - Number of unique answers
    plt.subplot(2, 2, 4)
    sns.barplot(data=df, x="model", y="num_unique_answers")
    plt.xticks(rotation=45, ha="right")
    plt.title("Average Number of Unique Answers by Model")
    plt.tight_layout()

    plt.savefig("logs/nouns/analysis/analysis_plots.png", bbox_inches="tight", dpi=300)

    # Save model comparison statistics
    model_summary.to_csv("logs/nouns/analysis/model_summary.csv")

    # Print model comparison statistics
    print("\nModel Comparison (Average Metrics):")
    print(model_summary.sort_values("most_common_frequency", ascending=False))

    # Print top 5 most consistent and least consistent nouns
    print("\nTop 5 Most Consistent Nouns (Highest Most Common Frequency):")
    print(summary.sort_values("most_common_frequency", ascending=False).head())

    print("\nTop 5 Least Consistent Nouns (Lowest Most Common Frequency):")
    print(summary.sort_values("most_common_frequency").head())

    print("\nTop 5 Nouns with Highest Model Agreement (Highest Most Common Frequency):")
    print(summary.sort_values("most_common_frequency", ascending=False).head())


if __name__ == "__main__":
    analyze_model_consistency()
