import tempfile
import streamlit as st

# =================================================
# Page Config (must be first Streamlit call)
# =================================================
st.set_page_config(
    page_title="Agentic RAG Evaluator",
    page_icon="🌌",
    layout="wide",
)

# =================================================
# Early render safeguard (prevents infinite spinner)
# =================================================
st.markdown("<!-- app booted -->", unsafe_allow_html=True)

# =================================================
# Lazy imports (CRITICAL for Streamlit Cloud)
# =================================================
def lazy_imports():
    from rag.document_loader import load_and_split_pdf
    from rag.vector_store import build_vectorstore
    from rag.retriever import create_retriever
    from agent.agent_core import build_agent
    from evaluation.eval_runner import run_pdf_evaluation
    return (
        load_and_split_pdf,
        build_vectorstore,
        create_retriever,
        build_agent,
        run_pdf_evaluation,
    )

# =================================================
# Cat GIFs
# =================================================
CAT_GIFS = {
    "idle": "https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif",
    "upload": "https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif",
    "happy": "https://media.giphy.com/media/111ebonMs90YLu/giphy.gif",
    "sad": "https://media.giphy.com/media/ROF8OQvDmxytW/giphy.gif",
}

# =================================================
# Session State (prevents loops)
# =================================================
if "cat_state" not in st.session_state:
    st.session_state.cat_state = "idle"

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "agent" not in st.session_state:
    st.session_state.agent = None

# =================================================
# Theme CSS
# =================================================
st.markdown(
    """
    <style>
    body {
        background: radial-gradient(circle at top left, #2b0f3f, #14001f 40%, #0b0014);
        color: #e5e7eb;
    }
    .card {
        background: linear-gradient(145deg, rgba(88,28,135,.35), rgba(24,5,44,.9));
        border-radius: 16px;
        padding: 1.4rem;
        border: 1px solid rgba(168,85,247,.25);
        box-shadow: 0 10px 40px rgba(0,0,0,.6);
        margin-bottom: 1.5rem;
    }
    .upload-row {
        display: flex;
        gap: 1.2rem;
    }
    .cat-card {
        display: flex;
        justify-content: center;
        align-items: center;
        background: linear-gradient(145deg, rgba(88,28,135,.35), rgba(24,5,44,.9));
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid rgba(168,85,247,.25);
    }
    .cat-card img {
        width: 160px;
        filter: drop-shadow(0 0 12px rgba(168,85,247,.6));
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
    <p style="color:#c4b5fd;">Agentic RAG + evaluation dashboard</p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# =================================================
# Sidebar
# =================================================
st.sidebar.title("⚙️ Controls")
show_context = st.sidebar.checkbox("Show retrieved context")

# =================================================
# Layout
# =================================================
left, right = st.columns([1.2, 1.8])

# =================================================
# LEFT PANEL (Upload + Cat)
# =================================================
with left:
    st.markdown("<div class='upload-row'>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="cat-card">
            <img src="{CAT_GIFS[st.session_state.cat_state]}">
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

# =================================================
# PROCESS UPLOAD (SAFE)
# =================================================
if uploaded_file and st.session_state.vectorstore is None:
    (
        load_and_split_pdf,
        build_vectorstore,
        create_retriever,
        build_agent,
        run_pdf_evaluation,
    ) = lazy_imports()

    st.session_state.cat_state = "upload"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    with st.spinner("Indexing document..."):
        docs = load_and_split_pdf(pdf_path)
        st.session_state.vectorstore = build_vectorstore(docs)
        retriever_fn = create_retriever(st.session_state.vectorstore)
        st.session_state.agent = build_agent(retriever_fn)

    st.session_state.cat_state = "idle"

# =================================================
# RIGHT PANEL (Q&A + Evaluation)
# =================================================
with right:
    if st.session_state.agent:
        question = st.text_area(
            "❓ Ask a question",
            height=120,
            placeholder="What is agentic AI?",
        )

        if question:
            (
                _,
                _,
                create_retriever,
                _,
                run_pdf_evaluation,
            ) = lazy_imports()

            retriever_fn = create_retriever(st.session_state.vectorstore)

            with st.spinner("Running agent + evaluation..."):
                report = run_pdf_evaluation(
                    question=question,
                    agent=st.session_state.agent,
                    retriever_fn=retriever_fn,
                )

            score = report["overall_score"]

            if score >= 0.75:
                st.session_state.cat_state = "happy"
            elif score <= 0.4:
                st.session_state.cat_state = "sad"
            else:
                st.session_state.cat_state = "idle"

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🤖 Agent Answer")
            st.write(report["answer"])
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 📊 Evaluation")
            st.json({
                "relevance": report["relevance"],
                "faithfulness": report["faithfulness"],
                "groundedness": report["groundedness"],
                "latency_sec": report["latency_sec"],
                "overall_score": score,
            })
            st.markdown("</div>", unsafe_allow_html=True)

            if show_context:
                with st.expander("📄 Retrieved Context"):
                    st.text_area(
                        "Context",
                        retriever_fn(question),
                        height=300,
                    )
