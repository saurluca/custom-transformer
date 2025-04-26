from rouge_score import rouge_scorer
import numpy as np
import matplotlib.pyplot as plt
import os


def calculate_rouge_scores(generated_summaries, reference_summaries):
    """
    Calculate ROUGE scores for generated summaries against reference summaries.

    Args:
        generated_summaries (list): List of generated summary texts
        reference_summaries (list): List of reference summary texts

    Returns:
        dict: Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for gen_sum, ref_sum in zip(generated_summaries, reference_summaries):
        scores = scorer.score(ref_sum, gen_sum)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": np.mean(rouge1_scores),
        "rouge2": np.mean(rouge2_scores),
        "rougeL": np.mean(rougeL_scores),
    }


def plot_rouge_scores(rouge_scores, save_path="plots/rouge_scores.png"):
    """
    Plot ROUGE scores as a bar chart.

    Args:
        rouge_scores (dict): Dictionary containing ROUGE scores
        save_path (str): Path to save the plot
    """
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create the plot
    plt.figure(figsize=(10, 6))
    metrics = list(rouge_scores.keys())
    scores = list(rouge_scores.values())

    plt.bar(metrics, scores)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("ROUGE Scores for Summarization")

    # Add score values on top of bars
    for i, score in enumerate(scores):
        plt.text(i, score + 0.02, f"{score:.3f}", ha="center")

    plt.grid(True, axis="y")
    plt.savefig(save_path)
    print(f"ROUGE scores plot saved to {save_path}")
    plt.close()
