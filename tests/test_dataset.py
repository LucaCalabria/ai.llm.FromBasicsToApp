from unittest.mock import MagicMock, patch

import pytest

from datasets_collector.dataset import DatasetManager


# Ensure a GitHub file page is fetched and processed during manager initialization.
def test_dataset_manager_downloads_and_initializes_dataset_state():
    # Return deterministic dataset bytes without making a network request.
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b"First line\nSecond line\n"
    url = "https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt"

    # Construct the manager while replacing the HTTP opener with the mock.
    with patch("datasets_collector.dataset.urlopen", return_value=response) as open_url:
        manager = DatasetManager(url)

    # Check URL conversion and preservation of the downloaded text.
    open_url.assert_called_once_with(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=30,
    )
    assert manager.url == url
    assert manager.dataset_text == "First line\nSecond line\n"
    assert manager.statistics["lines"] == 2
    assert manager.train_set == "First line"
    assert manager.val_set == "Second line"


# Ensure unsupported hosts are rejected before a request is made.
def test_dataset_manager_rejects_non_github_urls():
    manager = DatasetManager.__new__(DatasetManager)

    with pytest.raises(ValueError, match="GitHub"):
        manager.get_dataset("https://example.com/dataset.txt")


# Ensure previews contain no more than the requested number of lines.
def test_dataset_manager_preview_returns_requested_lines():
    manager = DatasetManager.__new__(DatasetManager)
    manager.dataset_text = "one\ntwo\nthree\n"

    assert manager.preview(num_lines=2) == "one\ntwo"


# Ensure text and frequent-word statistics match the sample dataset.
def test_dataset_manager_statistics_report_text_and_word_counts():
    manager = DatasetManager.__new__(DatasetManager)

    assert manager.get_dataset_statistics("To be, to be.\n", top_n=2) == {
        "characters": 14,
        "utf8_bytes": 14,
        "lines": 1,
        "words": 4,
        "unique_characters": 9,
        "unique_words": 2,
        "most_common_words": [("to", 2), ("be", 2)],
    }


# Ensure exploration helpers reject invalid negative limits.
@pytest.mark.parametrize(
    ("method", "text", "keyword"),
    [
        ("preview_dataset", "line", "num_lines"),
        ("get_dataset_statistics", "line", "top_n"),
    ],
)
def test_exploration_helpers_reject_negative_limits(method, text, keyword):
    manager = DatasetManager.__new__(DatasetManager)

    with pytest.raises(ValueError, match=keyword):
        getattr(manager, method)(text, -1)


# Ensure text is divided into training and validation sets by line.
def test_dataset_manager_splits_dataset_into_train_and_validation_sets():
    manager = DatasetManager.__new__(DatasetManager)

    assert manager.split_dataset("one\ntwo\nthree\nfour", train_ratio=0.5) == (
        "one\ntwo",
        "three\nfour",
    )


# Ensure the split ratio must leave data on both sides of the split.
@pytest.mark.parametrize("train_ratio", [0, 1, -0.1, 1.1])
def test_dataset_manager_rejects_invalid_train_ratios(train_ratio):
    manager = DatasetManager.__new__(DatasetManager)

    with pytest.raises(ValueError, match="train_ratio"):
        manager.split_dataset("one\ntwo", train_ratio)
