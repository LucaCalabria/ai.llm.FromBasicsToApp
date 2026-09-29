try:
    from .common import text_to_idx, get_most_frequent_pair, merge_idx, Tokenizer
except ImportError:
    from common import text_to_idx, get_most_frequent_pair, merge_idx, Tokenizer


class BaseTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, input_text: str, vocab_size: int):
        num_merges = vocab_size - 256  # Assuming 256 single-byte tokens
        input_idxs = text_to_idx(input_text) # list of integers 0-255 representing the input text
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}  # Initialize with single-byte tokens
        for i in range(num_merges):
            print(f"Training merge {i+1}/{num_merges}...")
            most_frequent_pair = get_most_frequent_pair(input_idxs)
            if not most_frequent_pair:
                break  # No more pairs to merge
            new_idx = 256 + i  # New token index for the merged pair
            vocab[new_idx] = most_frequent_pair[0]+most_frequent_pair[1]  # Add new token to vocabulary
            merges[f"{most_frequent_pair[0]},{most_frequent_pair[1]}"] = new_idx
            print(f"Merging pair {most_frequent_pair} into new token index {new_idx}.")
            input_idxs = merge_idx(input_idxs, most_frequent_pair, new_idx)  # Merge the pair in the input indices        
        self.idx_merges = merges
        self.vocab = vocab  # Update the vocabulary with merged tokens

    def encode(self, input_text: str) -> list:
        indexed_tokens = text_to_idx(input_text) # list of integers 0-255 representing the input text
        while len(indexed_tokens) >= 2:
            most_frequent_pair = get_most_frequent_pair(indexed_tokens)
            if not most_frequent_pair or most_frequent_pair not in self.idx_merges:
                break  # No more pairs to merge
            new_idx = self.idx_merges.get(tuple(most_frequent_pair))
            if new_idx is None:
                break  # No corresponding new index for the most frequent pair
            indexed_tokens = merge_idx(indexed_tokens, most_frequent_pair, new_idx)  # Merge the pair in the input indices
        return indexed_tokens  # Return the final list of token indices after all merges

    def decode(self, token_indices: list) -> str:
        text_bytes = b"".join(self.vocab[idx] for idx in token_indices)
        return text_bytes.decode("utf-8", errors="replace")  # Decode the bytes back to a UTF-8 string