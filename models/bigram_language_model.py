import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from language_model import LanguageModel
except ImportError:
    # Import the same helpers from the notebook's local module path.
    from .language_model import LanguageModel

class BigramLanguageModel(LanguageModel):
    
    def __init__(self, vocab_size):
        super().__init__()
        # Initialize the embedding layer to learn token embeddings.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # Get the logits for the next token predictions.
        logits = self.token_embedding_table(idx)

        if targets is None:
            # If no targets are provided, return only the logits.
            return logits, None

        # Compute the cross-entropy loss between logits and targets.
        B, T, C = logits.shape
        # Reshape logits and targets to match the expected input for cross-entropy loss.
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx