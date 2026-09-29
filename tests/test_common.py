# Import pytest for the test framework.
import pytest

# Import byte-pair helper functions exercised by these tests.
from tokenization.common import get_most_frequent_pair, merge_idx


# Ensure repeated byte pairs are counted accurately.
def test_get_most_frequent_pair():
    # Include a duplicated pair and one distinct pair.
    pairs = [97, 98, 97, 98, 98, 99]
    # Calculate occurrence counts for the supplied pairs.
    counts = get_most_frequent_pair(pairs)
    # Check the expected count for each pair.
    assert counts == (97, 98)


# Ensure the selected pair is replaced with its new token index.
def test_merge_idx():
    # Supply pairs with one repeated pair and two other pairs.
    pairs_even = [97, 98, 97, 98, 98, 99, 99, 100]
    pairs_odd = [97, 98, 97, 98, 98, 99, 99]
    # Select the pair that should be merged.
    most_common_pair = (97, 98)

    # Merge matching occurrences using token index 256.
    merged_even = merge_idx(pairs_even, most_common_pair, new_idx=256)

    # Check both matching occurrences were merged and other pairs remain.
    assert merged_even == [256, 256, 98, 99, 99, 100]

    # Merge matching occurrences using token index 256.
    merged_odd = merge_idx(pairs_odd, most_common_pair, new_idx=256)

    # Check both matching occurrences were merged and other pairs remain.
    assert merged_odd == [256, 256, 98, 99, 99]


# Ensure pair lookup safely reports that an empty input has no pair.
def test_get_most_frequent_pair_returns_none_without_pair():
    # A single value cannot form an adjacent pair.
    assert get_most_frequent_pair([97]) is None