# Import pytest for the test framework.
import pytest

# Import the tokenizer constructor regression-test target.
from tokenization.tokenizer import BaseTokenizer


# Ensure a new base tokenizer starts with valid merge state and byte vocabulary.
def test_base_tokenizer_initializes_merge_state():
    # Construct the tokenizer using its default state.
    tokenizer = BaseTokenizer()

    # Verify no merges exist before training.
    assert tokenizer.idx_merges == {}
    # Verify the vocabulary contains one entry for every byte value.
    assert len(tokenizer.vocab) == 256


# Ensure learned merges are applied during encoding and can be decoded.
def test_base_tokenizer_trains_encodes_and_decodes():
    # Train one repeated byte pair into a new token.
    tokenizer = BaseTokenizer()
    tokenizer.train("ababab", vocab_size=257)

    # Encode the training text using the learned token.
    token_ids = tokenizer.encode("ababab")

    # Confirm the merge is used and reconstructs the original text.
    assert token_ids == [256, 256, 256]
    assert tokenizer.decode(token_ids) == "ababab"
