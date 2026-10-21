from unittest.mock import MagicMock, patch

import pytest

from datasets_collector.dataset import (
    get_dataset,
    get_dataset_statistics,
    preview_dataset,
)


# Ensure a GitHub file page is fetched through its raw-content URL.
def test_get_dataset_downloads_raw_github_content():
    # Return deterministic dataset bytes without making a network request.
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b"First line\nSecond line\n"
    url = "https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt"

    # Fetch the dataset while replacing the HTTP opener with the mock.
    with patch("datasets_collector.dataset.urlopen", return_value=response) as open_url:
        dataset_text = get_dataset(url)

    # Check URL conversion and preservation of the downloaded text.
    open_url.assert_called_once_with(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=30,
    )
    assert dataset_text == "First line\nSecond line\n"


# Ensure unsupported hosts are rejected before a request is made.
def test_get_dataset_rejects_non_github_urls():
    with pytest.raises(ValueError, match="GitHub"):
        get_dataset("https://example.com/dataset.txt")


# Ensure previews contain no more than the requested number of lines.
def test_preview_dataset_returns_requested_lines():
    assert preview_dataset("one\ntwo\nthree\n", num_lines=2) == "one\ntwo"


# Ensure text and frequent-word statistics match the sample dataset.
def test_get_dataset_statistics_reports_text_and_word_counts():
    statistics = get_dataset_statistics("To be, to be.\n", top_n=2)

    assert statistics == {
        "characters": 14,
        "utf8_bytes": 14,
        "lines": 1,
        "words": 4,
        "unique_characters": 9,
        "unique_words": 2,
        "most_common_words": [("to", 2), ("be", 2)],
    }


# Ensure both exploration helpers reject invalid negative limits.
@pytest.mark.parametrize(
    ("function", "text", "keyword"),
    [
        (preview_dataset, "line", "num_lines"),
        (get_dataset_statistics, "line", "top_n"),
    ],
)
def test_exploration_helpers_reject_negative_limits(function, text, keyword):
    # The exception should identify which limit argument was invalid.
    with pytest.raises(ValueError, match=keyword):
        function(text, -1)