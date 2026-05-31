import os
import tempfile
import shutil
import streamlit as st
from config import APP_TITLE, APP_ICON, PAGE_LAYOUT, SUPPORTED_EXTENSIONS
from ingest import (
    process_file,
    list_indexed_files,
    delete_collection,
    get_chunk_count,
    get_collection_stats,
    delete_file,
)
from query import ask_document_stream, _conversation_memory

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=PAGE_LAYOUT)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

def refresh_indexed_files():
    st.session_state.processed_files = set(list_indexed_files())

def handle_file_upload(uploaded_files):
    if not uploaded_files:
        return

    files_to_process = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    if not files_to_process:
        return

    total_chunks = 0
    first_file = True
    progress_bar = st.progress(0, text="Processing files...")

    for i, uploaded_file in enumerate(files_to_process):
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            replace = first_file and (get_chunk_count() == 0)
            chunks = process_file(temp_path, replace=replace)
            total_chunks += chunks
            
            if chunks == 0:
                st.warning(f"No text could be extracted from {uploaded_file.name}. It might be scanned or image-based.")
                
            first_file = False
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        progress_bar.progress((i + 1) / len(files_to_process),
                              text=f"Processed {i+1}/{len(files_to_process)}")

    progress_bar.empty()

    if total_chunks > 0:
        st.success(f"Successfully indexed {len(files_to_process)} file(s) — {total_chunks} chunks!")
        refresh_indexed_files()
        st.rerun()

with st.sidebar:
    st.header("Document Management")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=list(SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
        help=f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
    )

    if uploaded_files:
        with st.spinner("Parsing, chunking, embedding & indexing..."):
            handle_file_upload(uploaded_files)

    st.subheader("Index Status")
    stats = get_collection_stats()

    col1, col2, col3 = st.columns(3)
    col1.metric("Chunks", stats["total_chunks"])
    col2.metric("Documents", stats["total_documents"])
    col3.metric("Files", len(stats["per_file"]))

    if stats["per_file"]:
        st.subheader("Indexed Documents")
        
        if "selected_docs" not in st.session_state:
            st.session_state.selected_docs = set()
        
        for filename, chunk_count_val in sorted(stats["per_file"].items()):
            checked = st.checkbox(
                f"{filename} ({chunk_count_val} chunks)",
                key=f"doc_{filename}",
                value=filename in st.session_state.selected_docs,
            )
            if checked:
                st.session_state.selected_docs.add(filename)
            else:
                st.session_state.selected_docs.discard(filename)
        
        if st.session_state.selected_docs:
            if st.button(f"Delete Selected ({len(st.session_state.selected_docs)})"):
                for fname in list(st.session_state.selected_docs):
                    delete_file(fname)
                st.session_state.selected_docs.clear()
                refresh_indexed_files()
                st.success("Selected files deleted!")
                st.rerun()

    if stats["total_chunks"] > 0:
        if st.button("Clear All Indexed Data", type="secondary"):
            if delete_collection():
                st.session_state.messages = []
                _conversation_memory.clear()
                if "selected_docs" in st.session_state:
                    st.session_state.selected_docs.clear()
                refresh_indexed_files()
                st.success("Index cleared!")
                st.rerun()

    st.subheader("Search Settings")
    use_hybrid = st.toggle(
        "Enable Hybrid Search", 
        value=False, 
        help="Combines dense vector search with BM25 keyword matching"
    )
    use_reranker = st.toggle(
        "Enable Re-ranking",
        value=False,
        help="Uses a Cross-Encoder to re-rank the top candidates for better accuracy"
    )
    use_transform = st.toggle(
        "Enable Query Transformation",
        value=False,
        help="Automatically rewrites, decomposes, or expands vague/complex queries"
    )
    use_self_rag = st.toggle(
        "Enable Self-RAG Verification",
        value=False,
        help="Checks answer for hallucinations and automatically corrects them"
    )

    if st.button("Clear Chat History", type="secondary"):
        _conversation_memory.clear()
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption(
        "**what we are using:**\n"
        "- **Groq LLaMA 3.3 70B**\n"
        "- **Sentence-Transformers** (locally)\n"
        "- **ChromaDB** (locally stored)\n"
        "- **Streamlit**"
    )

st.title(APP_TITLE)

tab_chat, tab_eval = st.tabs(["Chat", "Evaluation"])

with tab_chat:
    st.markdown("Upload documents in the sidebar, then ask questions about them below.")
    
    if get_chunk_count() == 0:
        st.info(
            "No documents indexed yet. Upload PDF, DOCX, TXT, or MD files "
            "using the sidebar to get started."
        )
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("View Sources", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**Source {i}:** `{src['file']}` — Relevance: {src['score']:.1%}")
                        st.caption(src["content"])
    
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
    
        with st.chat_message("assistant"):
            if get_chunk_count() == 0:
                response = "No documents indexed yet. Please upload files in the sidebar first."
                st.warning(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                })
            else:
                placeholder = st.empty()
                accumulated = ""
                sources_data = []
                
                try:
                    if use_self_rag:
                        from self_rag import self_rag_query
                        with st.spinner("Thinking... (Self-RAG enabled)"):
                            result = self_rag_query(prompt)
                            
                        if result.get("corrections_made"):
                            st.info(f"Self-correction performed ({result['iterations']} iterations)")
                        
                        if result.get("verified"):
                            st.success("Answer verified against source documents")
                        else:
                            st.warning(result.get('warning', 'Answer may need verification'))
                            
                        placeholder.markdown(result["answer"])
                        
                        sources_data = [
                            {
                                "content": chunk[0][:300] + ("..." if len(chunk[0]) > 300 else ""),
                                "file": chunk[1],
                                "score": chunk[2],
                            }
                            for chunk in result["sources"]
                        ]
                        
                        if sources_data:
                            with st.expander("View Sources", expanded=True):
                                for i, src in enumerate(sources_data, 1):
                                    st.markdown(
                                        f"**Source {i}:** `{src['file']}` — "
                                        f"Relevance: {src['score']:.1%}"
                                    )
                                    st.caption(src["content"])
                                    
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": sources_data,
                        })
    
                    else:
                        if use_reranker:
                            with st.spinner("Retrieving and re-ranking candidates..."):
                                stream_gen = ask_document_stream(prompt, use_hybrid=use_hybrid, use_reranker=use_reranker, use_transform=use_transform)
                                first_event = next(stream_gen, None)
                        else:
                            stream_gen = ask_document_stream(prompt, use_hybrid=use_hybrid, use_reranker=use_reranker, use_transform=use_transform)
                            first_event = next(stream_gen, None)
                        
                        def process_event(event_type, data):
                            global sources_data, accumulated
                            if event_type == "strategy" and data != "direct":
                                st.caption(f"Query Transformation Used: {data.capitalize()}")
                            elif event_type == "sources":
                                sources_data = data
                            elif event_type == "token":
                                accumulated += data
                                placeholder.markdown(accumulated + "▌")
                            elif event_type == "done":
                                placeholder.markdown(accumulated)
        
                        if first_event:
                            process_event(*first_event)
                            for event_type, data in stream_gen:
                                process_event(event_type, data)
                                
                        if sources_data:
                            with st.expander("View Sources", expanded=True):
                                for i, src in enumerate(sources_data, 1):
                                    st.markdown(
                                        f"**Source {i}:** `{src['file']}` — "
                                        f"Relevance: {src['score']:.1%}"
                                    )
                                    st.caption(src["content"])
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": accumulated,
                            "sources": sources_data,
                        })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.info(
                        "Troubleshooting:\n"
                        "1. Ensure GROQ_API_KEY is set in your .env file\n"
                        "2. Check you have an active internet connection\n"
                        "3. The Groq free tier has rate limits — wait a moment and retry"
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })
    
    st.divider()
    st.caption(
        "How to use: Ask specific questions for best results. "
        "Example: *\"What are the key findings in the research paper?\"*"
    )

with tab_eval:
    from evaluation import load_test_set, evaluate_system
    
    st.header("RAG System Evaluation")
    
    test_set_path = "test_set.json"
    if not os.path.exists(test_set_path):
        st.warning(f"Test set file {test_set_path} not found.")
    elif not list_indexed_files():
        st.warning("Please index documents before running evaluation.")
    else:
        test_set = load_test_set(test_set_path)
        test_set_size = len(test_set)
        st.info(f"Evaluation uses a test set of {test_set_size} questions.")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("Run Evaluation", type="primary"):
                with st.spinner("Running evaluation on test set..."):
                    df = evaluate_system(test_set)
                    st.session_state.eval_results = df
        
        with col2:
            st.metric("Test Questions", test_set_size)
        
        if "eval_results" in st.session_state:
            df = st.session_state.eval_results
            
            st.subheader("Summary")
            cols = st.columns(4)
            cols[0].metric("Avg Faithfulness", f"{df['faithfulness'].mean():.1%}")
            cols[1].metric("Avg Relevancy", f"{df['relevancy'].mean():.1%}")
            cols[2].metric("Context Recall", f"{df['context_recall'].mean():.1%}")
            cols[3].metric("Avg Latency", f"{df['latency_seconds'].mean():.1f}s")
            
            st.subheader("Per-Question Results")
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False)
            st.download_button(
                "Download Results (CSV)", 
                csv, 
                "eval_results.csv", 
                "text/csv"
            )