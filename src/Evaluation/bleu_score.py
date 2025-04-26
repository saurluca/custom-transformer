from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def calculate_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Calculate the BLEU score for a candidate translation against one or more references.

    Args:
        references (list of list of str): A list of reference translations (each tokenized).
        candidate (list of str): The candidate translation (tokenized).
        weights (tuple): Weights for BLEU score calculation (default is 4-gram BLEU).

    Returns:
        float: The BLEU score.
    """
    # Use smoothing to handle cases with no n-gram matches
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu(
        references, candidate, weights, smoothing_function=smoothing_function
    )


def evaluate_bleu_batch(
    tokenized_translations, tokenized_references, weights=(0.25, 0.25, 0.25, 0.25)
):
    """
    Evaluate BLEU scores for a batch of tokenized translations.

    Args:
        tokenized_translations (list of list of str): The tokenized translated sentences.
        tokenized_references (list of list of list of str): The tokenized reference translations (each sentence can have multiple references).
        weights (tuple): Weights for BLEU score calculation (default is 4-gram BLEU).

    Returns:
        list of float: BLEU scores for each translation.
        float: The average BLEU score for the batch.
    """
    bleu_scores = []

    for translated, references in zip(tokenized_translations, tokenized_references):
        # Calculate BLEU score for the current translation
        bleu_score = calculate_bleu(references, translated, weights)
        bleu_scores.append(bleu_score)

    # Calculate the average BLEU score
    average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    return bleu_scores, average_bleu
