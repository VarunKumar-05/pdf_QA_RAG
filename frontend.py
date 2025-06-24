
import base64
import tempfile
import pandas as pd
import plotly.express as px
from io import BytesIO

import streamlit as st
import requests
import json
import os
from datetime import datetime

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

if "pdf_collections" not in st.session_state:
    st.session_state.pdf_collections = []
if "selected_collection" not in st.session_state:
    st.session_state.selected_collection = None
if "combine_mode" not in st.session_state:
    st.session_state.combine_mode = True
def clear_qa():
    st.session_state.qa_question = ""

def main():
    st.set_page_config(page_title="PDF AI Assistant (Voice & Archive)", page_icon="🤖", layout="wide")
    
    st.title("🤖 PDF AI Assistant with Groq & Qdrant")
    st.markdown("Upload multiple PDFs, ask questions (by text or voice), generate context-aware questions, or search previous Q&A. Powered by Groq and Qdrant!")
    if "selected_pdf_hash" not in st.session_state:
        st.session_state.selected_pdf_hash = ""
    if "selected_pdf_filename" not in st.session_state:
        st.session_state.selected_pdf_filename = ""
  
    try:
        response = requests.get(f"{BACKEND_URL}/")
        if response.status_code == 200:
            st.success("🟢 Backend API connected successfully!")
        else:
            st.error("🔴 Backend API connection failed!")
            return
    except Exception as e:
        st.error(f"🔴 Cannot connect to backend API: {e}")
        return

    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("Qdrant Storage Mode")
        st.session_state.combine_mode = st.radio(
            "How to store PDFs?",
            options=["Combine all PDFs", "Keep PDFs separate"],
            index=0 if st.session_state.combine_mode else 1
        ) == "Combine all PDFs"

        # Collection name input for combine mode
        if st.session_state.combine_mode:
            collection_name_combine = st.text_input(
                "Enter a name for the combined collection:",
                value=st.session_state.get("collection_name_combine", ""),
                key="collection_name_combine"
            )
            if collection_name_combine:
                try:
                    requests.post(
                        f"{BACKEND_URL}/process-name",
                        json={"name": collection_name_combine}
                    )
                except Exception as e:
                    st.warning(f"Could not set collection name in backend: {e}")
        st.subheader("Retrieval & Answering Settings")
        top_k_results_for_retrieval = st.slider("Top-K Chunks for Display", 1, 10, 3, 1)
        top_k_results_for_llm = st.slider("Top-K Chunks for LLM Context", 1, 5, 3, 1)
        st.markdown("Clears the entire Qdrant collection(vector)")
        clear_collections = st.button("Clear Collections", on_click=lambda: requests.post(f"{BACKEND_URL}/clear-collections"))
        st.markdown("Clears the entire MongoDB collection(Q&A collection)")
        clear_collection = st.button("clear collections", on_click=lambda: requests.post(f"{BACKEND_URL}/clear-collection"))

    tab1, tab2, tab3 = st.tabs(["📄 PDF Q&A", "❓ Context-Aware QG", "📚 Q&A Archive collection"])
    # if (len(pdf_files) == 0):
    pdf_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True, key="pdf_uploads")
    combine_mode = st.session_state.combine_mode
    mode = "combine" if combine_mode else "separate"



    if pdf_files:
        new_files = [f for f in pdf_files if f.name not in [d["filename"] for d in st.session_state.pdf_collections]]
        if new_files:
            if combine_mode and not st.session_state.get("collection_name_combine"):
                st.warning("Please enter a collection name for the combined PDFs before uploading.")
            else:
                with st.spinner("🔄 Processing PDFs into sentences..."):
                    files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in new_files]
                    params = f"?mode={mode}"
                    if combine_mode:
                        params += f"&collection_name={st.session_state.collection_name_combine}"
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/upload-pdfs{params}",
                            files=files
                        )
                        if response.status_code == 200:
                            batch_results = response.json()["results"]
                            for i, result in enumerate(batch_results):
                                if result["success"]:
                                    st.success(f"✅ PDF {new_files[i].name} processed! Uploaded {result['chunks_count']} sentence-level chunks.")
                                    st.session_state.pdf_collections.append({
                                        "hash": result["pdf_hash"],
                                        "filename": os.path.basename(new_files[i].name),
                                        "chunks": result["chunks_count"]
                                    })
                                    st.info(f"{new_files[i].name} avg. sentence length: {result['avg_chunk_length']:.0f} chars.")
                                else:
                                    st.error(f"❌ {new_files[i].name}: {result['message']}")
                        else:
                            st.error(f"❌ Error processing PDFs: {response.json().get('detail', 'Unknown error')}")
                            return
                    except Exception as e:
                        st.error(f"❌ Error uploading PDFs: {e}")
                        return

    # PDF list with remove (X) buttons
    if st.session_state.pdf_collections:
        st.markdown("#### Uploaded PDFs")
        for idx, pdf in enumerate(st.session_state.pdf_collections):
            col1, col2, col3 = st.columns([6, 2, 1])
            with col1:
                st.write(f"{pdf['filename']} ({pdf['hash'][:8]})")
            with col2:
                st.caption(f"{pdf['chunks']} chunks")
            with col3:
                remove_btn = st.button("❌", key=f"remove_{pdf['hash']}_{idx}")
                if remove_btn:
                    filename_to_remove = os.path.basename(pdf["filename"])
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/remove-pdf",
                            json={
                                "pdf_hash": pdf["hash"],
                                "filename": filename_to_remove,
                                "mode": mode,
                                "collection_name": st.session_state.collection_name_combine if combine_mode else None
                            }
                        )
                        if response.status_code == 200:
                            st.success(f"Removed {pdf['filename']} from Qdrant!")
                            st.session_state.pdf_collections.pop(idx)
                            st.rerun()
                        else:
                            st.error(f"Failed to remove PDF embeddings from Qdrant: {response.text}")
                    except Exception as e:
                        st.error(f"Error removing PDF: {e}")

    if st.session_state.pdf_collections:
        if combine_mode:
            st.session_state.selected_collection = st.session_state.get("collection_name_combine", "")
            st.session_state.selected_pdf_filename = None
        else:
            selected = st.selectbox(
                "Select a PDF for Q&A and Question Generation:",
                [f"{d['filename']} ({d['hash'][:8]})" for d in st.session_state.pdf_collections],
                key="collection_selector"
            )
            idx = [f"{d['filename']} ({d['hash'][:8]})" for d in st.session_state.pdf_collections].index(selected)
            selected_pdf = st.session_state.pdf_collections[idx]
            st.session_state.selected_collection = f"{selected_pdf['hash']}_{selected_pdf['filename']}"
            st.session_state.selected_pdf_filename = selected_pdf['filename']

    with tab1:
        st.header("💬 Ask Questions About Your PDF (Text or Voice)")
        st.markdown("**Or record your question:**")

        audio_file = st.file_uploader(
            "🎤 Record or upload your question (WAV/MP3/OGG)",
            type=["wav", "mp3", "ogg"],
            key="audio_uploader"
        )
        mic_question = ""
        if audio_file:
            with st.spinner("Transcribing your voice..."):
                try:
                    files = {"file": (audio_file.name, audio_file.getvalue(), audio_file.type)}
                    response = requests.post(f"{BACKEND_URL}/speech-to-text", files=files)
                    if response.status_code == 200:
                        result = response.json()
                        mic_question = result["transcribed_text"]
                        if mic_question:
                            st.success(f"Transcribed: {mic_question}")
                            user_question = mic_question
                    else:
                        st.warning("Could not transcribe audio")
                except Exception as e:
                    st.warning(f"Audio transcription error: {e}")
        user_question = st.text_input("Enter your question:", key="qa_question")
        col1, spacer, col2 = st.columns([1, 6, 1])
        with col1:
            search_clicked = st.button("search", key="search_button")
        with col2:
            st.button("Clear", on_click=clear_qa)

        if search_clicked:
            if st.session_state.selected_collection and user_question:
                with st.spinner("🔍 Thinking..."):
                    question_data = {
                        "question": user_question,
                        "top_k_results": top_k_results_for_retrieval,
                        "collection_name": st.session_state.selected_collection
                    }
                    response = requests.post(f"{BACKEND_URL}/ask-question", json=question_data)
                    if response.status_code == 200:
                        result = response.json()
                        st.markdown(f"**Question:** {result.get('question','')}")
                        if result.get("corrected_question"):
                            st.info(f"Corrected Question: {result['corrected_question']}")
                        st.markdown(f"**Answer:**")
                        st.info(result.get("answer",""))
                        if result.get("similar_found", False):
                            st.success("✅ Found a similar question in the database!")
                        if result.get("sources"):
                            with st.expander(f"Relevant Sentences (Top {len(result['sources'])})"):
                                for i, r in enumerate(result["sources"], 1):
                                    st.markdown(f"**Sentence {i} (Score: {r['score']:.3f})**")
                                    st.caption(f"From: {r.get('source_pdf','')} | Page {r.get('page_number','')}")
                                    snippet = r["text"][:500] + "..." if len(r["text"]) > 500 else r["text"]
                                    st.caption(snippet)
                                    st.markdown("---")
                    else:
                        try:
                            st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
                        except:
                            st.error("❌ Unknown error during question answering.")

    with tab2:
        st.header("❓ Context-Aware Question Generation from Top Similar Sentences")
        if st.session_state.selected_collection:
            context_topic = st.text_input("Enter a context/topic for which to generate questions:", key="qg_context")
            num_questions = st.slider("How many questions (top sentences) to generate?", 1, 10, 5)
            spacers,spacer,colq = st.columns([1, 6, 1])
            with colq:
                generate_clicked= st.button("generate", key="generate_button")
            if generate_clicked :
                if context_topic:
                    with st.spinner("Searching for relevant sentences..."):
                        data = {
                            "context_topic": context_topic,
                            "num_questions": num_questions,
                            "pdf_hash": st.session_state.selected_pdf_hash,
                            "collection_name": st.session_state.selected_collection
                        }
                        try:
                            response = requests.post(f"{BACKEND_URL}/generate-questions", json=data)
                            if response.status_code == 200:
                                gqa = response.json()["generated_qa"]
                                st.markdown("### Generated Questions and Answers")
                                for i, qa in enumerate(gqa, 1):
                                    st.markdown(f"**Sentence {i}:** {qa['sentence']}")
                                    st.caption(f"From: {qa.get('source_pdf','')} | Page {qa.get('page_number','')}")
                                    st.info(f"**Generated Question:** {qa['generated_question']}")
                                    st.success(f"**Answer:** {qa['answer']}")
                                    st.divider()
                            else:
                                st.warning(f"Error: {response.json().get('detail','Unknown error')}")
                        except Exception as e:
                            st.warning(f"Error generating questions: {e}")
                else:
                    st.info("📄 Please upload and process a PDF to generate questions and answers.")

    with tab3:
        st.header("📚 Q&A Archive collection")
        try:
            response = requests.get(f"{BACKEND_URL}/collections")
            if response.status_code == 200:
                collections = response.json()["collections"]
                if not collections:
                    st.info("No Q&A collections found. Process a PDF first.")
                else:
                    selected_collection = st.selectbox("Select a PDF (collection):", collections)
                    if selected_collection:
                        qa_response = requests.get(f"{BACKEND_URL}/qa-pairs/{selected_collection}")
                        if qa_response.status_code == 200:
                            qa_pairs = qa_response.json().get("qa_pairs", [])
                            if not qa_pairs:
                                st.info("No Q&A pairs found in this collection.")
                            else:
                                for i, qa in enumerate(qa_pairs, 1):
                                    st.markdown(f"---\n**Q{i}:** {qa.get('question', '')}")
                                    st.markdown(f"**A{i}:** {qa.get('answer', '')}")
                                    if qa.get("tags"):
                                        st.caption(" | ".join(qa["tags"]))
                                    if qa.get("timestamp"):
                                        st.caption(f"🕒 {qa['timestamp']}")
                        else:
                            st.warning(f"Could not fetch Q&A pairs for {selected_collection}")
            else:
                st.warning("Could not fetch collections from backend.")
        except Exception as e:
            st.warning(f"Error loading archive: {e}")

    st.markdown("---")
    st.caption("💡 Tips: Use the microphone to ask, and listen to answers! For best results, use specific questions.")
    if st.session_state.get('selected_chunks_count', 0):
        st.caption(f"ℹ️ Current PDF: {st.session_state.selected_chunks_count} sentences loaded.")

if __name__ == "__main__":
    main()