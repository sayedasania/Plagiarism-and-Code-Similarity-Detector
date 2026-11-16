"""Module 1 + 8: Input and interface (Streamlit)."""
import html
import pandas as pd
import streamlit as st

from report import compare_all, compare_one, top_pairs, clusters
from similarity import edit_similarity


def highlight(doc, matched):
    rows = []
    for i, line in enumerate(doc.code.splitlines(), 1):
        style = "background:rgba(255,80,80,0.35);" if i in matched else ""
        rows.append(f'<div style="{style}white-space:pre;font-family:monospace;'
                    f'font-size:13px">{i:>3} | {html.escape(line)}</div>')
    return '<div style="overflow-x:auto">' + "".join(rows) + "</div>"


def run():
    st.set_page_config(page_title="Code Similarity Detector", layout="wide")
    st.title("Plagiarism & Code Similarity Detector")
    st.caption("Tokenizing, Rabin-Karp k-gram hashing, Jaccard, LCS and Union-Find")

    st.sidebar.header("Settings")
    k = st.sidebar.slider("k-gram size (tokens)", 3, 10, 5)
    threshold = st.sidebar.slider("Copy-group threshold %", 30, 100, 60)

    uploads = st.file_uploader("Upload 2 or more source files", accept_multiple_files=True,
                               type=["py", "c", "cpp", "java", "js", "txt"])
    if len(uploads) < 2:
        st.info("Upload at least two files to compare.")
        return

    files = {u.name: u.read().decode("utf-8", errors="ignore") for u in uploads}
    mode = st.radio("What do you want to check?",
                    ["One student vs all others", "All pairs (everyone vs everyone)"],
                    horizontal=True)

    if mode.startswith("One"):
        target = st.selectbox("Select the student to check", list(files))
        docs, res = compare_one(files, target, k)
        df1 = pd.DataFrame(res)
        st.subheader(f"{target} compared with {len(res)} other students")
        st.dataframe(df1, use_container_width=True, hide_index=True)
        best = res[0]
        st.error(f"Most similar to: {best['Student']} ({best['Similarity %']}%, {best['Verdict']})")
        other = st.selectbox("See matching code with", [r["Student"] for r in res])
        dt, do = docs[target], docs[other]
        c1, c2 = st.columns(2)
        c1.markdown(f"**{target}**")
        c1.markdown(highlight(dt, dt.matched_lines(do.fp)), unsafe_allow_html=True)
        c2.markdown(f"**{other}**")
        c2.markdown(highlight(do, do.matched_lines(dt.fp)), unsafe_allow_html=True)
        return

    docs, results = compare_all(files, k)
    df = pd.DataFrame(results).sort_values("Similarity %", ascending=False)

    st.subheader("All pairs")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Top matching pairs")
    for r in top_pairs(results, 3):
        st.write(f"**{r['File A']}** vs **{r['File B']}**: {r['Similarity %']}% ({r['Verdict']})")

    groups = clusters(list(files), results, threshold)
    st.subheader("Suspected copy groups")
    if groups:
        for g in groups:
            st.warning(" , ".join(g))
    else:
        st.success("No groups above the threshold.")

    st.subheader("Side-by-side view")
    pair = st.selectbox("Choose a pair", [f"{r['File A']}  |  {r['File B']}" for r in
                                          df.to_dict("records")])
    a, b = [p.strip() for p in pair.split("|")]
    da, db = docs[a], docs[b]
    st.caption(f"Edit-distance similarity on tokens: {edit_similarity(da.tokens, db.tokens):.1%}")
    c1, c2 = st.columns(2)
    c1.markdown(f"**{a}**")
    c1.markdown(highlight(da, da.matched_lines(db.fp)), unsafe_allow_html=True)
    c2.markdown(f"**{b}**")
    c2.markdown(highlight(db, db.matched_lines(da.fp)), unsafe_allow_html=True)
