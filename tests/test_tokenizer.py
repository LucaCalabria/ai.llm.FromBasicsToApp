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
