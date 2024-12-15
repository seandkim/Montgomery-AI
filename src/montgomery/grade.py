from typing import List

from montgomery.helper import print_error, print_verbose
from .guitar import GuitarTab, Pitch, tabs2string
import json


def read_tabs_from_file(filename):
    tabs = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if line == "":
                continue
            elif line == "None":
                tabs.append(None)
            else:
                string_note, fret_index = line.strip().split(": ")
                string_note = Pitch(string_note)
                if string_note not in GuitarTab.BASE_STRINGS:
                    raise ValueError(
                        f"Found invalid line. Invalid string_note: {string_note}"
                    )
                string_index = GuitarTab.BASE_STRINGS.index(string_note)
                fret_index = int(fret_index)
                if 0 > fret_index or fret_index > GuitarTab.MAX_FRET_INDEX:
                    raise ValueError(
                        f"Found valid line. Invalid fret_index: {fret_index}"
                    )
                line = GuitarTab(string_index, fret_index)
                tabs.append(line)
    return tabs


def longest_common_subsequence(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def calculate_score(expected, actual):
    lcs_length = longest_common_subsequence(expected, actual)
    return lcs_length / len(expected) if expected else 0.0


def test_calculate_score():
    assert abs(calculate_score([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9
    assert abs(calculate_score([1, 2, 3], [2, 3]) - (2 / 3)) < 1e-9
    assert abs(calculate_score([10, 20, 30, 40, 50], [20, 30, 50]) - (3 / 5)) < 1e-9
    assert abs(calculate_score([1, 3, 5, 7], [1, 2, 3, 4, 5, 6, 7]) - 1.0) < 1e-9
    assert abs(calculate_score([2, 4, 6, 8], [2, 6, 4, 8]) - (3 / 4)) < 1e-9
    assert abs(calculate_score([1, 2, 3, 4], [4, 3, 2, 1]) - 0.25) < 1e-9
    assert abs(calculate_score([100, 200, 300], []) - 0.0) < 1e-9
    assert (
        abs(calculate_score([5, 10, 15, 20, 25], [5, 5, 10, 10, 15, 15, 25, 25]) - 0.8)
        < 1e-9
    )


if __name__ == "__main__":
    # test_calculate_score()

    OUT_DIR = "files/satisfaction"
    expected_file = f"{OUT_DIR}/answer.txt"
    expected = read_tabs_from_file(expected_file)

    scores = dict()
    actual_files = [f"{OUT_DIR}/predicted_tabs_tiny.txt", f"{OUT_DIR}/chatgpt.txt"]
    for actual_file in actual_files:
        actual = read_tabs_from_file(actual_file)
        score = calculate_score(expected, actual)
        print_verbose(
            f"{actual_file}: {score} (Actual length={len(actual)}, Expected length={len(expected)})"
        )
        scores[actual_file] = score

    output_data = {"scores": scores}
    with open(f"{OUT_DIR}/grade.json", "w") as json_file:
        json.dump(output_data, json_file, indent=4)
