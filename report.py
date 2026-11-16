"""Modules 6 + 7: Comparison manager and report (ranking, verdict, clusters)."""
import heapq
from collections import defaultdict
from itertools import combinations

from preprocess import tokenize
from fingerprint import fingerprint
from similarity import jaccard, lcs_ratio, combined_score


class Doc:
    def __init__(self, name, code, k):
        self.name, self.code = name, code
        self.tokens, self.lines = tokenize(code)
        self.k = min(k, len(self.tokens)) or k
        self.hashes, self.fp = fingerprint(self.tokens, k)

    def matched_lines(self, other_fp):
        """Source lines covered by k-grams that also occur in the other file."""
        out = set()
        for i, h in enumerate(self.hashes):
            if h in other_fp:
                out.update(self.lines[i:i + self.k])
        return out


def verdict(score):
    return "High" if score >= 60 else "Medium" if score >= 30 else "Low"


def compare_all(files, k=5, lcs_top=30, lcs_cutoff=0.125):
    """files: {name: code}.

    Fast version (the idea behind real tools like MOSS):
      1. Inverted index  hash -> files containing it.
      2. One pass over the index counts shared hashes of every pair,
         so Jaccard for all pairs comes almost for free.
      3. A heap picks the `lcs_top` most suspicious pairs (Jaccard >= cutoff).
      4. The slow LCS, O(n*m), runs only on those pairs.
    """
    docs = {n: Doc(n, c, k) for n, c in files.items()}

    index = defaultdict(set)
    for name, d in docs.items():
        for h in d.fp:
            index[h].add(name)

    shared = defaultdict(int)
    for owners in index.values():
        if len(owners) > 1:
            for pair in combinations(sorted(owners), 2):
                shared[pair] += 1

    jac_of = {}
    for a, b in combinations(sorted(docs), 2):
        inter = shared.get((a, b), 0)
        union = len(docs[a].fp) + len(docs[b].fp) - inter
        jac_of[(a, b)] = inter / union if union else 0.0

    candidates = [p for p, j in jac_of.items() if j >= lcs_cutoff]
    refine = set(heapq.nlargest(lcs_top, candidates, key=jac_of.get))

    results = []
    for (a, b), jac in jac_of.items():
        lcs = lcs_ratio(docs[a].tokens, docs[b].tokens) if (a, b) in refine else None
        score = combined_score(jac, lcs or 0.0)
        results.append({"File A": a, "File B": b,
                        "Jaccard %": round(100 * jac, 1),
                        "LCS %": None if lcs is None else round(100 * lcs, 1),
                        "Similarity %": round(score, 1),
                        "Verdict": verdict(score)})
    return docs, results


def compare_one(files, target, k=5):
    """Compare ONE student's file against all the others (n-1 comparisons).
    Returns (docs, results) sorted from most to least similar."""
    docs = {n: Doc(n, c, k) for n, c in files.items()}
    t = docs[target]
    results = []
    for name, d in docs.items():
        if name == target:
            continue
        jac = jaccard(t.fp, d.fp)
        lcs = lcs_ratio(t.tokens, d.tokens)
        score = combined_score(jac, lcs)
        results.append({"Student": name,
                        "Jaccard %": round(100 * jac, 1),
                        "LCS %": round(100 * lcs, 1),
                        "Similarity %": round(score, 1),
                        "Verdict": verdict(score)})
    results.sort(key=lambda r: r["Similarity %"], reverse=True)
    return docs, results


def top_pairs(results, n=5):
    """Heap-based ranking of the most similar pairs."""
    return heapq.nlargest(n, results, key=lambda r: r["Similarity %"])


class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path compression
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def clusters(names, results, threshold=60):
    """Group files that copied from each other (groups of size > 1)."""
    uf = UnionFind(names)
    for r in results:
        if r["Similarity %"] >= threshold:
            uf.union(r["File A"], r["File B"])
    groups = {}
    for n in names:
        groups.setdefault(uf.find(n), []).append(n)
    return [g for g in groups.values() if len(g) > 1]
