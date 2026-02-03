Copyright 2026 Hrithik Ghosh (hrithik373)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


# 🌌 Agentic RAG Evaluator

A **production-ready Agentic Retrieval-Augmented Generation (RAG) system** with
document-grounded evaluation, built using **LangChain, FAISS, OpenAI LLMs, and Streamlit**.

🚀 **Live Demo**  
👉 https://araassistant-ky7xiosunekrfmu427tt2f.streamlit.app/

---

## 🐱 Because every serious AI system needs a cat

<p align="center">
  <img src="https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif" width="200"/>
</p>

*(The cat reacts to evaluation scores — happy for good answers, sad for weak ones.)(Currently this feature is disabled :( )*

---

## ✨ Features

- 📄 **PDF Upload & Indexing**
  - Upload any PDF document
  - Automatic chunking and vector indexing

- 🧠 **Agentic RAG Pipeline**
  - LLM + Tool-using agent
  - Retrieval-augmented reasoning over documents
  - Conversation-aware execution

- 📊 **LLM-as-Judge Evaluation**
  - Relevance
  - Faithfulness
  - Groundedness
  - Latency tracking
  - Overall quality score

- 🎨 **Modern Streamlit UI**
  - Cyberpunk / midnight purple theme
  - Responsive layout
  - Visual metrics dashboard
  - Reactive GIFs for fun UX feedback

- ☁️ **Cloud-Ready**
  - Deployed on Streamlit Cloud
  - CPU-only FAISS
  - Environment-safe dependency handling

---

## 🧠 System Architecture

```text
User Question
     ↓
Document Retriever (FAISS)
     ↓
Relevant Context Chunks
     ↓
Agentic LLM (Reason + Tool Use)
     ↓
Generated Answer
     ↓
LLM-as-Judge Evaluation
     ↓
Metrics + Visualization
