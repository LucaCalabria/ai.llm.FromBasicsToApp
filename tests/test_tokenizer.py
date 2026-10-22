# Import pytest for the test framework.
import pytest
import torch

# Import the tokenizer constructor regression-test target.
from tokenization.tokenizer import BPETokenizer, CharTokenizer


# Ensure a new BPE tokenizer starts with valid merge state and byte vocabulary.
def test_bpe_tokenizer_initializes_merge_state():
    # Construct the tokenizer using its default state.
    tokenizer = BPETokenizer()

    # Verify no merges exist before training.
    assert tokenizer.idx_merges == {}
    # Verify the vocabulary contains one entry for every byte value.
    assert len(tokenizer.vocab) == 256


# Ensure learned merges are applied during encoding and can be decoded.
def test_bpe_tokenizer_trains_encodes_and_decodes():
    # Train one repeated byte pair into a new token.
    tokenizer = BPETokenizer()
    tokenizer.train("ababab", vocab_size=257)

    # Encode the training text using the learned token.
    token_ids = tokenizer.encode("ababab")

    # Confirm the merge is used and reconstructs the original text.
    assert torch.equal(token_ids, torch.tensor([256, 256, 256]))
    assert tokenizer.decode(token_ids) == "ababab"


# Ensure a new character tokenizer starts with empty vocabulary mappings.
def test_char_tokenizer_initializes_empty_vocabulary():
    # Construct the tokenizer before it has been trained.
    tokenizer = CharTokenizer()

    # Verify no characters have been added yet.
    assert tokenizer.vocab == {}
    assert tokenizer.char_to_token == {}


# Ensure training creates a deterministic vocabulary with sorted characters.
def test_char_tokenizer_trains_sorted_character_vocabulary():
    # Train using repeated characters in non-sorted order.
    tokenizer = CharTokenizer()
    tokenizer.train("caba")

    # Verify each unique character has a token in sorted order.
    assert tokenizer.vocab == {0: b"a", 1: b"b", 2: b"c"}
    assert tokenizer.char_to_token == {b"a": 0, b"b": 1, b"c": 2}


# Ensure Unicode characters round-trip through encoding and decoding.
def test_char_tokenizer_encodes_and_decodes_unicode_text():
    # Train on ASCII, accented, and emoji characters.
    tokenizer = CharTokenizer()
    tokenizer.train("baé🙂")

    # Encode a sequence containing the known characters in a new order.
    token_ids = tokenizer.encode("🙂éba🙂")

    # Verify the token IDs and reconstructed Unicode text.
    assert torch.equal(token_ids, torch.tensor([3, 2, 1, 0, 3]))
    assert tokenizer.decode(token_ids) == "🙂éba🙂"


# Ensure empty training data and text produce empty token sequences.
def test_char_tokenizer_handles_empty_text():
    # Train without any characters.
    tokenizer = CharTokenizer()
    tokenizer.train("")

    # Verify encoding and decoding empty text both remain empty.
    assert torch.equal(tokenizer.encode(""), torch.tensor([]))
    assert tokenizer.decode(torch.tensor([])) == ""


# Ensure encoding fails when text contains a character absent from training.
def test_char_tokenizer_rejects_characters_outside_training_vocabulary():
    # Train a vocabulary that does not contain "d".
    tokenizer = CharTokenizer()
    tokenizer.train("abc")

    # Verify an unknown character raises the expected lookup error.
    with pytest.raises(KeyError):
        tokenizer.encode("d")
