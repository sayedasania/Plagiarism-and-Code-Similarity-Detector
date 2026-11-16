"""Module 5: Similarity engine (Jaccard, LCS, Edit Distance)."""


def jaccard(a, b):
    """|A ∩ B| / |A ∪ B| on the fingerprint sets (hash-set operations)."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def lcs_length(x, y):
    """Longest Common Subsequence, dynamic programming, O(n*m) time, O(m) space."""
    if len(x) < len(y):
        x, y = y, x
    prev = [0] * (len(y) + 1)
    for xi in x:
        cur = [0]
        for j, yj in enumerate(y, 1):
            cur.append(prev[j - 1] + 1 if xi == yj else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def lcs_ratio(x, y):
    if not x or not y:
        return 0.0
    return lcs_length(x, y) / max(len(x), len(y))


def edit_distance(x, y):
    """Levenshtein distance by dynamic programming, O(n*m)."""
    if len(x) < len(y):
        x, y = y, x
    prev = list(range(len(y) + 1))
    for i, xi in enumerate(x, 1):
        cur = [i]
        for j, yj in enumerate(y, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (xi != yj)))
        prev = cur
    return prev[-1]


def edit_similarity(x, y):
    m = max(len(x), len(y))
    return 1.0 if m == 0 else 1 - edit_distance(x, y) / m


def combined_score(jac, lcs):
    """Final % = average of the two methods."""
    return 100 * (0.8 * jac + 0.2 * lcs)
