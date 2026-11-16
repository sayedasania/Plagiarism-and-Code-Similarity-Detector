"""Module 4: Fingerprinting with k-grams + Rabin-Karp rolling hash.

Sliding window of k tokens; each window's hash is computed from the previous one
in O(1) (rolling hash), so all k-grams are hashed in O(n) total.
"""
import zlib

BASE = 257
MOD = (1 << 61) - 1


def token_ids(tokens):
    return [zlib.crc32(t.encode()) for t in tokens]


def kgram_hashes(tokens, k=5):
    """Rabin-Karp rolling hash over every window of k tokens."""
    ids = token_ids(tokens)
    n = len(ids)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    high = pow(BASE, k - 1, MOD)          # weight of the leftmost token
    h = 0
    for i in range(k):
        h = (h * BASE + ids[i]) % MOD
    hashes = [h]
    for i in range(k, n):                 # slide the window by one token
        h = ((h - ids[i - k] * high) * BASE + ids[i]) % MOD
        hashes.append(h)
    return hashes


def fingerprint(tokens, k=5):
    """Return (list of hashes by position, set of unique hashes)."""
    hashes = kgram_hashes(tokens, k)
    return hashes, set(hashes)
