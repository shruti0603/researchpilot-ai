import glob, os, re
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
NLI_MODELS = [
    ("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", None),  # stronger NLI; label ids read from model config
    ("cross-encoder/nli-deberta-v3-small", (1, 0)),          # fallback: entailment=1, contradiction=0
]

@st.cache_resource(show_spinner="Loading open-source models (first run takes a few minutes)...")
def load_models():
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    for mid, fixed in NLI_MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            mdl = AutoModelForSequenceClassification.from_pretrained(mid).eval()
            lab = {str(v).lower(): int(k) for k, v in mdl.config.id2label.items()}
            if "entailment" in lab and "contradiction" in lab:
                return emb, (tok, mdl, lab["entailment"], lab["contradiction"], mid)
            if fixed:
                return emb, (tok, mdl, fixed[0], fixed[1], mid)
        except Exception:
            continue
    raise RuntimeError("Could not load an NLI model")

emb, nli = load_models()

def nli_probs(pairs):
    """pairs = [(premise, hypothesis)] -> (entailment_probs, contradiction_probs)"""
    tok, mdl, ei, ci, _ = nli
    enc = tok([p for p, _ in pairs], [h for _, h in pairs], truncation="only_first",
              max_length=384, padding=True, return_tensors="pt")
    with torch.no_grad():
        pr = torch.softmax(mdl(**enc).logits, dim=-1).numpy()
    return pr[:, ei], pr[:, ci]

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

def windows(text, size=3):
    sents = [x for x in re.split(r"(?<=[.!?])\s+", text) if len(x) > 20]
    return [" ".join(sents[i:i + size]) for i in range(max(1, len(sents) - size + 1))] or [text]

def verify(text, passages):
    """Label each claim: supported / partial / unsupported, with the best source passage."""
    clean = re.sub(r"\[\d+\]|\*\*|[#*_`]", "", text)
    claims = [re.sub(r"^\s*\d+\.\s*", "", c).strip() for c in re.split(r"(?<=[.!?])\s+|\n+", clean)]
    claims = [c for c in claims if len(c) > 25]
    cand = [(w, i) for i, (c, _) in enumerate(passages) for w in windows(c["text"])]
    cv = emb.encode([w for w, _ in cand], normalize_embeddings=True)
    results = []
    for claim in claims:
        qv = emb.encode([claim], normalize_embeddings=True)[0]
        sims = cv @ qv
        top = np.argsort(-sims)[:5]
        ents, cons = nli_probs([(cand[j][0], claim) for j in top])
        k = int(np.argmax(ents))
        ent, con, sim = float(ents[k]), float(cons[k]), float(sims[top[k]])
        label = "supported" if ent >= 0.5 else "partial" if (sim >= 0.55 and con < 0.5) else "unsupported"
        results.append((claim, label, cand[top[k]][1], float(ent)))
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
        passages = retrieve(q, chunks, vecs, k=8)
        with st.spinner("Generating grounded answer..."):
            ans = answer(q, passages)
            checks = verify(ans, passages)
        st.subheader("Answer")
        st.write(ans)
        if checks:
            n = len(checks)
            sup = sum(l == "supported" for _, l, _, _ in checks)
            par = sum(l == "partial" for _, l, _, _ in checks)
            c1, c2, c3 = st.columns(3)
            c1.metric("Fully supported", f"{100*sup/n:.0f}%")
            c2.metric("Partially supported", f"{100*par/n:.0f}%")
            c3.metric("Not supported", f"{100*(n-sup-par)/n:.0f}%")
            st.subheader("Claim verification")
            icons = {"supported": "✅ Supported", "partial": "🟡 Partially supported", "unsupported": "⚠️ Not supported"}
            for claim, label, src, ent in checks:
                st.write(f"{icons[label]} (best source [{src+1}], entailment {ent:.2f}): {claim}")
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
            e1, c1_ = nli_probs([(pa["text"], pb["text"])])
            e2, c2_ = nli_probs([(pb["text"], pa["text"])])
            ent, con = max(e1[0], e2[0]), max(c1_[0], c2_[0])
            label = "⚠️ Conflicting evidence" if con > 0.5 else "✅ Supporting evidence" if ent > 0.5 else "ℹ️ Different or complementary findings"
            st.caption(f"entailment {ent:.2f} · contradiction {con:.2f} · NLI model: {nli[4]}")
            st.subheader(label)
            c1, c2 = st.columns(2)
            c1.markdown(f"**{a}, p.{pa['page']}**"); c1.write(pa["text"])
            c2.markdown(f"**{b}, p.{pb['page']}**"); c2.write(pb["text"])
