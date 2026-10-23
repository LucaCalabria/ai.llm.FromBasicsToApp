import torch
import torch.nn.functional as F

from models.bigram_language_model import BigramLanguageModel


# Ensure the bigram model creates a square token embedding table.
def test_bigram_language_model_initializes_embedding_table():
    model = BigramLanguageModel(vocab_size=5)

    assert model.token_embedding_table.num_embeddings == 5
    assert model.token_embedding_table.embedding_dim == 5


# Ensure forward returns unflattened logits and no loss without targets.
def test_forward_without_targets_returns_logits_and_no_loss():
    model = BigramLanguageModel(vocab_size=5)
    token_ids = torch.tensor([[0, 1, 2], [3, 4, 0]])

    logits, loss = model(token_ids)

    assert logits.shape == (2, 3, 5)
    assert loss is None
    assert torch.equal(logits, model.token_embedding_table(token_ids))


# Ensure forward computes cross-entropy loss when targets are provided.
def test_forward_with_targets_returns_flattened_logits_and_loss():
    model = BigramLanguageModel(vocab_size=5)
    token_ids = torch.tensor([[0, 1, 2], [3, 4, 0]])
    targets = torch.tensor([[1, 2, 3], [4, 0, 1]])

    logits, loss = model(token_ids, targets)
    expected_logits = model.token_embedding_table(token_ids).view(-1, 5)
    expected_loss = F.cross_entropy(expected_logits, targets.view(-1))

    assert logits.shape == (6, 5)
    assert torch.equal(logits, expected_logits)
    assert torch.allclose(loss, expected_loss)


# Ensure generation appends the requested number of valid token IDs.
def test_generate_appends_requested_number_of_tokens():
    model = BigramLanguageModel(vocab_size=5)
    context = torch.tensor([[0, 1], [2, 3]])

    generated = model.generate(context, max_new_tokens=4)

    assert generated.shape == (2, 6)
    assert torch.equal(generated[:, :2], context)
    assert torch.all((generated >= 0) & (generated < 5))


# Ensure generation with no requested tokens preserves the input tensor.
def test_generate_with_zero_new_tokens_returns_input_unchanged():
    model = BigramLanguageModel(vocab_size=5)
    context = torch.tensor([[0, 1], [2, 3]])

    generated = model.generate(context, max_new_tokens=0)

    assert torch.equal(generated, context)
