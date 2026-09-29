# Convert input text into a list of UTF-8 byte values.
def text_to_idx(input_text: str) -> list:
    idx_list = []
    if input_text:
        # Store each encoded byte as an integer from 0 through 255.
        idx_list = list(input_text.encode('utf-8'))
    return idx_list

# Find the adjacent byte pair that appears most often, if one exists.
def get_most_frequent_pair(list_idxs: list) -> tuple[int, int] | None:
    # Store the number of occurrences for each adjacent pair.
    counts = {}
    # Visit every overlapping pair of neighboring values.
    for pair in zip(list_idxs, list_idxs[1:]):
        # Increment the count for this pair.
        counts[pair] = counts.get(pair, 0) + 1
    # Return None when fewer than two values were supplied.
    if not counts:
        return None
    # Return the pair with the highest count.
    return max(counts, key=counts.get)

# Replace every occurrence of a pair with its new token index.
def merge_idx(list_idxs: list, most_frequent_pair: tuple[int, int] | list[int], new_idx: int) -> list:
    # Collect the sequence after applying the merge.
    updated_list_idxs = []
    i = 0
    while i < len(list_idxs):
        # Preserve the last value when only one remains to inspect.
        if i == len(list_idxs) - 1:
            # Append the last value to the updated sequence.
            updated_list_idxs += [list_idxs[i]]
            # Advance one position to exit the loop.
            i += 1
        else:
            # Read the current adjacent values as a pair.
            idxs_pair = [list_idxs[i], list_idxs[i+1]]
            # Replace a matching pair with the merged token.
            if idxs_pair == list(most_frequent_pair):
                # Append the index that represents the merged pair.
                updated_list_idxs += [new_idx]
                # Skip both original values because they were merged.
                i += 2
            # Preserve the current value when this pair does not match.
            else:
                # Append only the current value so overlapping pairs can be checked.
                updated_list_idxs += [list_idxs[i]]
                # Advance one position to inspect the next pair.
                i += 1
    return updated_list_idxs


# Define the shared interface and vocabulary setup for tokenizers.
class Tokenizer:
    # Initialize merge state before constructing the base vocabulary.
    def __init__(self):
        # Store learned pair-to-token mappings.
        self.idx_merges = {}
        # Build the initial vocabulary from the empty merge mapping.
        self.vocab = self._build_vocab_from_merges()

    # Declare the training operation implemented by concrete tokenizers.
    def train(self, input_text: str, vocab_size: int):
        # Require subclasses to implement vocabulary training.
        raise NotImplementedError("The train method should be implemented in subclasses.")
         
    # Declare the text-to-token operation implemented by concrete tokenizers.
    def encode(self, input_text: str) -> list:
        # Require subclasses to implement text encoding.
        raise NotImplementedError("The encode method should be implemented in subclasses.")

    # Declare the token-to-text operation implemented by concrete tokenizers.
    def decode(self, token_indices: list) -> str:
        # Require subclasses to implement token decoding.
        raise NotImplementedError("The decode method should be implemented in subclasses.") 

    # Build byte-token entries and expand each learned merge into bytes.
    def _build_vocab_from_merges(self) -> dict:
        # Map every possible byte value to its one-byte representation.
        vocabulary = {idx: bytes([idx]) for idx in range(256)}
        # Add each merged token by concatenating its component bytes.
        for (pair0, pair1), idx in self.idx_merges.items():
            # Store the bytes represented by this merged token index.
            vocabulary[idx] = vocabulary[pair0] + vocabulary[pair1]
        # Return the completed byte vocabulary.
        return vocabulary
    