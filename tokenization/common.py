def text_to_idx(input_text: str) -> list:
    idx_list = []
    if input_text:
        # split text into ascii characters
        # this produces the same result as: idx_list = [ord(char) for char in input_text]
        idx_list = list(input_text.encode('utf-8')) # list of integers 0-255
    return idx_list

def get_most_frequent_pair(list_idxs: list) -> tuple[int,int]:
    frequent_pair = []
    counts = {}
    for pair in zip(list_idxs, list_idxs[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    frequent_pair = max(counts, key=counts.get)
    return frequent_pair

def merge_idx(list_idxs: list, most_frequent_pair: list, new_idx: int) -> list:
    updated_list_idxs = []
    i = 0
    while i < len(list_idxs) - 1:
        idxs_pair = [list_idxs[i], list_idxs[i+1]]
        if idxs_pair == list(most_frequent_pair):
            updated_list_idxs += [new_idx]
            i += 2
        else:
            updated_list_idxs += [list_idxs[i]]
            i += 1
    assert most_frequent_pair not in updated_list_idxs, "The most frequent pair should not be present in the updated list."    
    return updated_list_idxs


# Base Tokenizer class
class Tokenizer:
    def __init__(self):
        self.idx_merges = {}
        self.vocab = self._build_vocab_from_merges()

    def train(self, input_text: str, vocab_size: int):
        # Train the tokenizer vocabulary from input text
        raise NotImplementedError("The train method should be implemented in subclasses.")
         
    def encode(self, input_text: str) -> list:
        # Encode the input text into a list of token indices
        raise NotImplementedError("The encode method should be implemented in subclasses.")

    def decode(self, token_indices: list) -> str:
        # Decode the list of token indices back into a string
        raise NotImplementedError("The decode method should be implemented in subclasses.") 

    def _build_vocab_from_merges(self) -> dict:
        # Build the vocabulary from the learned merges
        # Initialize with single-byte tokens
        vocabulary = {idx: bytes([idx]) for idx in range(256)}
        # Add merged tokens to the vocabulary
        for (pair0, pair1), idx in self.idx_merges.items():
            vocabulary[idx] = vocabulary[pair0] + vocabulary[pair1]
        return vocabulary
    