import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModel(nn.Module):

    def __init__(self, vocab_size=None, embedding_dim=None, hidden_dim=None):
        super(LanguageModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

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