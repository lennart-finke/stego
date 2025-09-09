#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import pandas as pd
from plotly.graph_objects import Figure, Sankey


def load_analysis_file(filepath):
    """Load and parse the analysis JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def categorize_encryption_method(method_str):
    """Categorize encryption methods into main types."""
    method_str = method_str.lower()

    if "xor" in method_str:
        return "XOR Cipher"
    # Using '‐' (hyphen) and '-' (minus sign) for one-time pad
    if "one-time pad" in method_str or "one‐time pad" in method_str:
        return "One-Time Pad"
    if (
        "caesar" in method_str
        or "vigenère" in method_str
        or "shift" in method_str
        or "rotational" in method_str
        or "modular addition" in method_str
        or "additive" in method_str
        or "modulo" in method_str
        or "mod 26" in method_str
        or "mod26" in method_str
    ):
        return "Shift/Rotational Cipher"
    if "substitution" in method_str and "cyclic" in method_str:
        return "Cyclic Substitution"
    if "substitution" in method_str:
        return "Substitution Cipher"

    return "Other/Unspecified"


def categorize_steganography_method(method_str):
    """Categorize steganography methods into main types."""
    method_str = method_str.lower()

    if "zero-width" in method_str or "unicode" in method_str:
        return "Zero-width Unicode"
    if (
        "acrostic" in method_str
        or "initial" in method_str
        or "first letter" in method_str
    ):
        return "Acrostic/Initial Letters"
    if (
        "positional" in method_str
        or "book cipher" in method_str
        or "book‐cipher" in method_str
        or "index" in method_str
        or "position" in method_str
        or "extraction" in method_str
    ):
        return "Positional/Book Cipher"
    if (
        "word choice" in method_str
        or "punctuation" in method_str
        or "sentence structure" in method_str
        or "word lengths" in method_str
    ):
        return "Linguistic Manipulation"
    if (
        "end of" in method_str
        or "ending letter" in method_str
        or "final letter" in method_str
        or "appended" in method_str
    ):
        return "Text Modification"
    if "word" in method_str and ("select" in method_str or "index" in method_str):
        return "Word-based Selection"

    return "Other/Unspecified"


def create_sankey_diagram(data):
    """Create a Sankey diagram showing the flow of success through different stages."""
    # Extract success metrics for each stage
    stages = [
        "encoding_success",
        "innocuous",
        "entropy_used",
        "decoder_aware",
        "method_match",
        "decryption_success",
    ]

    # Create descriptive labels for failure nodes
    failure_labels = {
        "encoding_success": "Encoding<br>Failed",
        "innocuous": "No<br>Steganography",
        "entropy_used": "Key<br>Unused",
        "decoder_aware": "Decoder<br>Unaware",
        "method_match": "Method<br>Mismatch",
        "decryption_success": "Decryption<br>Failed",
    }

    # Create more descriptive stage labels
    stage_labels = {
        "encoding_success": "Encoding<br>Success",
        "innocuous": "Message<br>Innocuous",
        "entropy_used": "Key<br>Used",
        "decoder_aware": "Decoder<br>Detected<br>Message",
        "method_match": "Correct<br>Decoding<br>Method",
        "decryption_success": "Decryption<br>Success",
    }

    print(len(data["samples"]))

    # Count successes and failures at each stage
    stage_counts = []
    total_samples = len(data["samples"])

    # Start with all samples
    stage_counts.append(total_samples)

    # For each stage, count samples that succeeded in this stage AND all previous stages
    # Also track failures at each stage
    success_counts = []
    failure_counts = []

    for i, stage in enumerate(stages):
        # Count successes (passed this stage and all previous)
        success_count = sum(
            1
            for sample in data["samples"]
            if all(sample["analysis"][prev_stage] for prev_stage in stages[: i + 1])
        )
        success_counts.append(success_count)

        # Count failures (passed all previous but failed this stage)
        if i == 0:
            failure_count = total_samples - success_count
        else:
            failure_count = sum(
                1
                for sample in data["samples"]
                if all(sample["analysis"][prev_stage] for prev_stage in stages[:i])
                and not sample["analysis"][stage]
            )
        failure_counts.append(failure_count)

    # Create nodes for each stage plus start and end nodes
    # Interleave the failure nodes with the stages
    node_labels = ["All<br>Samples"]
    node_colors = ["#1f77b4"]

    for i, stage in enumerate(stages):
        # Add the stage node
        node_labels.append(stage_labels[stage])
        node_colors.append("#2ca02c")

        # Add the corresponding failure node
        node_labels.append(failure_labels[stage])
        node_colors.append("#d62728")

    # Add the final success node
    node_labels.append("")
    node_colors.append("#ff7f0e")

    nodes = {
        "label": node_labels,
        "pad": 15,
        "thickness": 20,
        "line": dict(color="black", width=0.5),
        "color": node_colors,
    }

    # Create links between stages
    links = {"source": [], "target": [], "value": []}

    # Add links from Start to first stage
    links["source"].append(0)
    links["target"].append(1)
    links["value"].append(stage_counts[0])

    # Add links between stages and failure paths
    for i in range(len(stages)):
        current_stage_idx = i * 2 + 1  # Index of current stage node
        failure_node_idx = current_stage_idx + 1  # Index of corresponding failure node

        # Success path to next stage
        if i < len(stages) - 1:
            next_stage_idx = (i + 1) * 2 + 1  # Index of next stage node
            links["source"].append(current_stage_idx)
            links["target"].append(next_stage_idx)
            links["value"].append(success_counts[i])
        else:
            # Last stage success goes to Success node
            links["source"].append(current_stage_idx)
            links["target"].append(len(node_labels) - 1)  # Success node
            links["value"].append(success_counts[i])

        # Failure path to specific failure node
        links["source"].append(current_stage_idx)
        links["target"].append(failure_node_idx)
        links["value"].append(failure_counts[i])

    # Create Sankey diagram
    fig = Figure(data=[Sankey(node=nodes, link=links)])

    fig.update_layout(
        font_size=30,
        height=600,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig


def create_stacked_bar_chart(data):
    """Create a stacked bar chart showing encryption and steganography methods."""
    # Extract and categorize methods
    encryption_methods = []
    steganography_methods = []

    for sample in data["samples"]:
        method = sample["analysis"].get("encryption_method", "")
        encryption_methods.append(categorize_encryption_method(method))
        steganography_methods.append(categorize_steganography_method(method))

    # Count occurrences
    encryption_counts = pd.Series(encryption_methods).value_counts()
    steganography_counts = pd.Series(steganography_methods).value_counts()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot encryption methods
    encryption_counts.plot(kind="bar", ax=ax1, color="skyblue")
    ax1.set_title("Encryption Methods Used")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=45)

    # Plot steganography methods
    steganography_counts.plot(kind="bar", ax=ax2, color="lightgreen")
    ax2.set_title("Steganography Methods Used")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig


def main():
    """Main function to generate visualizations."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize steganography experiment results"
    )
    parser.add_argument("analysis_file", help="Path to the analysis JSON file")
    args = parser.parse_args()

    # Load data
    data = load_analysis_file(args.analysis_file)

    # Create Sankey diagram
    sankey_fig = create_sankey_diagram(data)
    sankey_fig.write_html("figures/steganography_sankey.html")

    # Create stacked bar chart
    bar_fig = create_stacked_bar_chart(data)
    bar_fig.savefig("figures/steganography_methods.png")

    print("Visualizations generated successfully!")
    print("- Sankey diagram saved as 'figures/steganography_sankey.html'")
    print("- Method distribution saved as 'figures/steganography_methods.png'")


if __name__ == "__main__":
    main()
