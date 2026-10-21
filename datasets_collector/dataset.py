"""Fetch and inspect text datasets hosted on GitHub."""

import re
from collections import Counter
from urllib.error import URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen


def _raw_github_url(url: str) -> str:
	"""Convert a GitHub file page URL to its raw-content URL."""
	parsed_url = urlsplit(url)
	hostname = parsed_url.hostname

	if parsed_url.scheme not in {"http", "https"} or hostname not in {
		"github.com",
		"raw.githubusercontent.com",
	}:
		raise ValueError("url must be a GitHub file URL")

	if hostname == "github.com":
		path_parts = parsed_url.path.strip("/").split("/")
		if len(path_parts) < 5 or path_parts[2] != "blob":
			raise ValueError("url must point to a file on GitHub")
		raw_path = "/".join(path_parts[:2] + path_parts[3:])
		return urlunsplit(("https", "raw.githubusercontent.com", f"/{raw_path}", "", ""))

	return urlunsplit(("https", parsed_url.netloc, parsed_url.path, "", ""))


def get_dataset(url: str) -> str:
	"""Download a UTF-8 text dataset from a GitHub file URL."""
	raw_url = _raw_github_url(url)
	try:
		with urlopen(raw_url, timeout=30) as response:
			return response.read().decode("utf-8")
	except URLError as error:
		raise OSError(f"Could not download dataset from {raw_url}") from error


def preview_dataset(dataset_text: str, num_lines: int = 10) -> str:
	"""Return the first ``num_lines`` lines of a dataset."""
	if num_lines < 0:
		raise ValueError("num_lines must be non-negative")
	return "\n".join(dataset_text.splitlines()[:num_lines])


def get_dataset_statistics(dataset_text: str, top_n: int = 10) -> dict:
	"""Return basic size, vocabulary, and frequent-word statistics."""
	if top_n < 0:
		raise ValueError("top_n must be non-negative")

	words = re.findall(r"\b[\w']+\b", dataset_text.casefold())
	return {
		"characters": len(dataset_text),
		"utf8_bytes": len(dataset_text.encode("utf-8")),
		"lines": len(dataset_text.splitlines()),
		"words": len(words),
		"unique_characters": len(set(dataset_text)),
		"unique_words": len(set(words)),
		"most_common_words": Counter(words).most_common(top_n),
	}
