import warnings
from transformers import AutoTokenizer


def assert_tokenizer_consistency(model_id_1, model_id_2):
    identical_tokenizers = (
            AutoTokenizer.from_pretrained(model_id_1).vocab
            == AutoTokenizer.from_pretrained(model_id_2).vocab
    )
    if not identical_tokenizers:
        warnings.warn(
            f"Warning: Tokenizers for models '{model_id_1}' and '{model_id_2}' have different vocabularies. "
            f"This may lead to inconsistent results when comparing these models. "
            f"Consider using models with compatible tokenizers.",
            UserWarning
        )
