# Prefer package-relative imports when this module is loaded through tokenization.
try:
    # Import tokenizer helpers and the base class from the containing package.
    from .common import text_to_idx, get_most_frequent_pair, merge_idx, Tokenizer
# Support the notebook's top-level import when its directory is on sys.path.
except ImportError:
    # Import the same helpers from the notebook's local module path.
    from common import text_to_idx, get_most_frequent_pair, merge_idx, Tokenizer


# Implement byte-pair training, encoding, and decoding.
class BaseTokenizer(Tokenizer):
    # Initialize the shared tokenizer state.
    def __init__(self):
        # Build the base byte vocabulary through the parent class.
        super().__init__()

    # Learn pair merges from the input until the requested vocabulary size.
    def train(self, input_text: str, vocab_size: int):
        # Reserve the first 256 entries for individual byte values.
        num_merges = vocab_size - 256
        # Convert the training text to its UTF-8 byte values.
        input_idxs = text_to_idx(input_text)
        # Collect each learned pair and its assigned token index.
        merges = {}
        # Initialize the vocabulary with all single-byte tokens.
        vocab = {idx: bytes([idx]) for idx in range(256)}
        # Learn no more than the requested number of merges.
        for i in range(num_merges):
            # Report which merge is currently being trained.
            print(f"Training merge {i+1}/{num_merges}...")
            # Find the most frequent adjacent pair in the current sequence.
            most_frequent_pair = get_most_frequent_pair(input_idxs)
            # Stop if no pair is available to merge.
            if not most_frequent_pair:
                # End training when the input has no remaining pairs.
                break
            # Assign the next index after the base byte-token range.
            new_idx = 256 + i
            # Store the two byte values represented by the new token.
            vocab[new_idx] = most_frequent_pair[0]+most_frequent_pair[1]
            # Record the learned pair using its serialized key.
            merges[f"{most_frequent_pair[0]},{most_frequent_pair[1]}"] = new_idx
            # Report the pair and the token index assigned to it.
            print(f"Merging pair {most_frequent_pair} into new token index {new_idx}.")
            # Apply the merge before selecting the next frequent pair.
            input_idxs = merge_idx(input_idxs, most_frequent_pair, new_idx)
        # Save the learned pair mappings on the tokenizer.
        self.idx_merges = merges
        # Replace the base vocabulary with the newly trained vocabulary.
        self.vocab = vocab

    # Convert text bytes into token indices by applying learned merges.
    def encode(self, input_text: str) -> list:
        # Begin with one token per UTF-8 byte.
        indexed_tokens = text_to_idx(input_text)
        # Continue while at least one adjacent pair remains.
        while len(indexed_tokens) >= 2:
            # Select the most frequent pair in the current token sequence.
            most_frequent_pair = get_most_frequent_pair(indexed_tokens)
            # Stop if that pair has no learned merge entry.
            if not most_frequent_pair or most_frequent_pair not in self.idx_merges:
                # No applicable merge remains.
                break
            # Look up the token index assigned to the selected pair.
            new_idx = self.idx_merges.get(tuple(most_frequent_pair))
            # Stop if the merge mapping contains no matching token index.
            if new_idx is None:
                # No corresponding new index is available.
                break
            # Replace all selected pairs with their learned token index.
            indexed_tokens = merge_idx(indexed_tokens, most_frequent_pair, new_idx)
        # Return the sequence after all applicable merges.
        return indexed_tokens

    # Convert token indices back into a UTF-8 string.
    def decode(self, token_indices: list) -> str:
        # Concatenate the byte representation for each token index.
        text_bytes = b"".join(self.vocab[idx] for idx in token_indices)
        # Decode the bytes and replace malformed UTF-8 sequences if needed.
        return text_bytes.decode("utf-8", errors="replace")