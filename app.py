import glob, os, re
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from sentence_transformers import CrossEncoder, SentenceTransformer
import requests

st.set_page_config(page_title="ResearchPilot AI", page_icon="🔬", layout="wide")
LLM_ID = "openai/gpt-oss-20b"  # open-weight model (Apache 2.0) served by Groq free API

# ---------- demo login ----------
if not st.session_state.get("ok"):
    st.title("🔬 ResearchPilot AI")
    st.caption("Evidence-first research assistant")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u == "demo" and p == "demo1234":
            st.session_state.ok = True
            st.rerun()
        st.error("Invalid credentials")
    st.info("Demo credentials: **demo / demo1234**")
    st.stop()

# ---------- models ----------
@st.cache_resource(show_spinner="Loading open-source models (first run takes a few minutes)...")
def load_models():
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    nli = CrossEncoder("cross-encoder/nli-deberta-v3-small")  # labels: contradiction, entailment, neutral
    return emb, nli

emb, nli = load_models()

# ---------- ingestion ----------
def chunk_pdf(name, data):
    doc = fitz.open(stream=data, filetype="pdf")
    out = []
    for pn, page in enumerate(doc, 1):
        words = page.get_text().split()
        for i in range(0, len(words), 120):
            text = " ".join(words[i:i + 150])
            if len(text) > 80:
                out.append({"doc": name, "page": pn, "text": text})
    return out

@st.cache_data(show_spinner="Indexing documents...")
def build_index(files):
    chunks = []
    for name, data in files:
        chunks += chunk_pdf(name, data)
    vecs = emb.encode([c["text"] for c in chunks], normalize_embeddings=True, batch_size=32)
    return chunks, np.array(vecs)

def retrieve(q, chunks, vecs, k=5, doc=None):
    qv = emb.encode([q], normalize_embeddings=True)[0]
    scores = vecs @ qv
    order = np.argsort(-scores)
    res = [(chunks[i], float(scores[i])) for i in order if doc is None or chunks[i]["doc"] == doc]
    return res[:k]

# ---------- generation + verification ----------
def answer(q, passages):
    ctx = "\n\n".join(f"[{i+1}] ({c['doc']}, p.{c['page']}) {c['text']}" for i, (c, _) in enumerate(passages))
    msgs = [
        {"role": "system", "content": "Answer ONLY using the numbered sources. Cite sources like [1]. If the sources do not contain the answer, say 'Insufficient evidence in the provided documents.'"},
        {"role": "user", "content": f"Sources:\n{ctx}\n\nQuestion: {q}"},
    ]
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}"},
        json={"model": LLM_ID, "messages": msgs, "temperature": 0, "max_tokens": 1200, "reasoning_effort": "low"},
        timeout=60,
    )
    if not r.ok:
        st.error(f"LLM API error {r.status_code}: {r.text[:300]}")
        st.stop()
    return r.json()["choices"][0]["message"]["content"].strip()

def verify(text, passages):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]
    results = []
    for s in sents:
        pairs = [(c["text"], s) for c, _ in passages]
        preds = np.array(nli.predict(pairs))
        ent = int(np.argmax(preds, axis=1).tolist().count(1) > 0)
        best = int(np.argmax([p[1] for p in preds])) if ent else None
        results.append((s, bool(ent), best))
    return results

# ---------- UI ----------
st.title("🔬 ResearchPilot AI")
st.caption("Ask questions across papers. Every claim is cited and verified against the source text.")

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    files = [(os.path.basename(f), open(f, "rb").read()) for f in sorted(glob.glob("samples/*.pdf"))]
    files += [(u.name, u.read()) for u in uploads]
    if not files:
        st.warning("No documents yet. Upload a PDF.")
        st.stop()
    chunks, vecs = build_index(tuple(files))
    st.success(f"{len(files)} papers, {len(chunks)} passages indexed")
    for n, _ in files:
        st.write("•", n)

tab1, tab2 = st.tabs(["Ask with citations", "Compare papers"])

with tab1:
    q = st.text_input("Your question", placeholder="What retrieval method does the paper use?")
    if st.button("Ask") and q:
        passages = retrieve(q, chunks, vecs)
        with st.spinner("Generating grounded answer..."):
            ans = answer(q, passages)
            checks = verify(ans, passages)
        st.subheader("Answer")
        st.write(ans)
        if checks:
            pct = 100 * sum(ok for _, ok, _ in checks) / len(checks)
            st.metric("Claims supported by sources", f"{pct:.0f}%")
            st.subheader("Claim verification")
            for s, ok, best in checks:
                src = f" → source [{best+1}]" if ok else ""
                st.write(("✅ Supported" if ok else "⚠️ Not supported") + src + f": {s}")
        st.subheader("Evidence")
        for i, (c, sc) in enumerate(passages, 1):
            with st.expander(f"[{i}] {c['doc']} — page {c['page']} (similarity {sc:.2f})"):
                st.write(c["text"])

with tab2:
    names = [n for n, _ in files]
    if len(names) < 2:
        st.info("Add at least two papers to compare.")
    else:
        a = st.selectbox("Paper A", names, index=0)
        b = st.selectbox("Paper B", names, index=1)
        topic = st.text_input("Topic or claim to compare", placeholder="effect of retrieval on accuracy")
        if st.button("Compare") and topic:
            pa = retrieve(topic, chunks, vecs, k=1, doc=a)[0][0]
            pb = retrieve(topic, chunks, vecs, k=1, doc=b)[0][0]
            pred = np.array(nli.predict([(pa["text"], pb["text"])]))[0]
            label = ["⚠️ Conflicting evidence", "✅ Supporting evidence", "ℹ️ Different or unrelated findings"][int(np.argmax(pred))]
            st.subheader(label)
            c1, c2 = st.columns(2)
            c1.markdown(f"**{a}, p.{pa['page']}**"); c1.write(pa["text"])
            c2.markdown(f"**{b}, p.{pb['page']}**"); c2.write(pb["text"])
