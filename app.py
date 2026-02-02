import tempfile
import streamlit as st

from rag.document_loader import load_and_split_pdf
from rag.vector_store import build_vectorstore
from rag.retriever import create_retriever
from agent.agent_core import build_agent
from evaluation.eval_runner import run_pdf_evaluation


# =================================================
# Cat GIFs
# =================================================
CAT_GIFS = {
    "idle": "https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif",       # dancing
    "upload": "https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif",# sus
    "happy": "https://media.giphy.com/media/111ebonMs90YLu/giphy.gif",    # celebration
    "sad": "https://media.giphy.com/media/ROF8OQvDmxytW/giphy.gif",        # sad
}


# =================================================
# Page Config
# =================================================
st.set_page_config(
    page_title="Agentic RAG Evaluator",
    page_icon="🌌",
    layout="wide",
)


# =================================================
# Midnight Purple Theme + Layout CSS
# =================================================
st.markdown(
    """
    <style>
    body {
        background: radial-gradient(
            circle at top left,
            #2b0f3f 0%,
            #14001f 40%,
            #0b0014 100%
        );
        color: #e5e7eb;
    }

    .block-container {
        padding-top: 2rem;
    }

    .card {
        background: linear-gradient(
            145deg,
            rgba(88, 28, 135, 0.35),
            rgba(24, 5, 44, 0.9)
        );
        border-radius: 16px;
        padding: 1.4rem;
        border: 1px solid rgba(168, 85, 247, 0.25);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6);
        margin-bottom: 1.5rem;
    }

    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        color: #e9d5ff;
    }

    .answer-box {
        line-height: 1.7;
        font-size: 0.95rem;
        color: #f5f3ff;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-top: 1rem;
    }

    .metric-card {
        background: rgba(17, 5, 34, 0.85);
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(168, 85, 247, 0.35);
    }

    .metric-label {
        font-size: 0.7rem;
        color: #c4b5fd;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e9d5ff;
    }

    .overall-score {
        margin-top: 1.2rem;
        font-size: 1.6rem;
        font-weight: 800;
        text-align: center;
        color: #a855f7;
        text-shadow: 0 0 12px rgba(168, 85, 247, 0.6);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #1a0428 0%,
            #0b0014 100%
        );
        border-right: 1px solid rgba(168, 85, 247, 0.25);
    }

    textarea, input {
        background-color: #0f0020 !important;
        color: #f5f3ff !important;
        border: 1px solid rgba(168, 85, 247, 0.35) !important;
        border-radius: 10px !important;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(17, 5, 34, 0.7);
        border-radius: 12px;
        padding: 0.8rem;
        border: 1px dashed rgba(168, 85, 247, 0.4);
    }

    .upload-row {
        display: flex;
        gap: 1.2rem;
        align-items: stretch;
    }

    .upload-card {
        flex: 1.2;
    }

    .cat-card {
        flex: 0.8;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(
            145deg,
            rgba(88, 28, 135, 0.35),
            rgba(24, 5, 44, 0.9)
        );
        border-radius: 16px;
        border: 1px solid rgba(168, 85, 247, 0.25);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6);
    }

    .cat-card img {
        width: 100%;
        max-width: 180px;
        opacity: 0.95;
        filter: drop-shadow(0 0 12px rgba(168,85,247,0.6));
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =================================================
# Header
# =================================================
st.markdown(
    """
    <h1 style="color:#f5f3ff;">🌌 Agentic RAG Evaluation</h1>
    <p style="color:#c4b5fd;">
        Midnight-themed agentic RAG system with document-grounded evaluation.
    </p>
    <hr style="border:1px solid rgba(168,85,247,0.3);">
    """,
    unsafe_allow_html=True,
)


# =================================================
# Sidebar
# =================================================
st.sidebar.title("⚙️ Controls")
show_context = st.sidebar.checkbox("Show retrieved context")
st.sidebar.markdown("---")
st.sidebar.caption("Agent • RAG • FAISS • LLM-as-Judge")


# =================================================
# State
# =================================================
cat_state = "idle"


# =================================================
# Layout
# =================================================
left, right = st.columns([1.1, 1.9])


# =================================================
# LEFT: Upload + Cat
# =================================================
with left:
    st.markdown("<div class='upload-row'>", unsafe_allow_html=True)

    # Upload card
    st.markdown("<div class='card upload-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>📄 Document</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    agent = None
    retriever_fn = None
    question = None

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        cat_state = "upload"
        with st.spinner("Indexing document..."):
            docs = load_and_split_pdf(pdf_path)
            vectorstore = build_vectorstore(docs)
            retriever_fn = create_retriever(vectorstore)
            agent = build_agent(retriever_fn)

        st.success("Document indexed")

        st.markdown("<div class='card-title'>❓ Question</div>", unsafe_allow_html=True)
        question = st.text_area(
            "Ask a question",
            height=110,
            placeholder="What is agentic AI?",
        )
    else:
        st.info("Upload a PDF to begin")

    st.markdown("</div>", unsafe_allow_html=True)

    # Cat card (dynamic)
    st.markdown(
        f"""
        <div class="cat-card">
            <img src="{CAT_GIFS[cat_state]}" />
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# =================================================
# RIGHT: Answer + Evaluation
# =================================================
with right:
    if uploaded_file and question and agent and retriever_fn:
        with st.spinner("Running agent and evaluation..."):
            report = run_pdf_evaluation(
                question=question,
                agent=agent,
                retriever_fn=retriever_fn,
            )

        # Score-based cat reaction
        if report["overall_score"] >= 0.75:
            cat_state = "happy"
        elif report["overall_score"] <= 0.4:
            cat_state = "sad"
        else:
            cat_state = "idle"

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🤖 Agent Answer</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='answer-box'>{report['answer']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📊 Evaluation</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Relevance</div>
                    <div class="metric-value">{report['relevance']:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Faithfulness</div>
                    <div class="metric-value">{report['faithfulness']:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Groundedness</div>
                    <div class="metric-value">{report['groundedness']:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Latency</div>
                    <div class="metric-value">{report['latency_sec']}</div>
                </div>
            </div>

            <div class="overall-score">
                Overall Score: {report['overall_score']}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if show_context:
            with st.expander("📄 Retrieved Context"):
                st.text_area("Context", retriever_fn(question), height=300)
