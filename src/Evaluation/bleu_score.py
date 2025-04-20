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
    return sentence_bleu(references, candidate, weights, smoothing_function=smoothing_function)