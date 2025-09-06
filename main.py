import streamlit as st
from functions import load_and_chunk, summarize_chunks, update_faiss, build_answer_chain, translate_article, embedding, llm

st.set_page_config(page_title="Student Article Helper")
st.title("Student Article Helper")
st.write("Welcome! Use this app to help with student articles.")

# Keep chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



# --------------------- Main action ---------------------
action = st.radio(
    "What do you want to do?",
    (
        "Summarize an article",
        "Get an answer from your knowledge base",
        "Translate article to another language"
    ),
    key="main_action_radio"
)
# --------------------- Summarize an article ---------------------
if action == "Summarize an article":
    url = st.text_input("Enter the URL of the article to summarize:", key="summary_url")
    if url:
        url_chunks = load_and_chunk(url)
        summary = summarize_chunks(url_chunks[1], llm, url_chunks[0])
        update_faiss(summary, embedding)
        st.markdown("---")
        st.subheader("Summary")
        st.write(summary.page_content)
    else:
        st.info("Please enter a URL to proceed.")

# --------------------- Get an answer from KB ---------------------
elif action == "Get an answer from your knowledge base":
    question = st.text_input("Enter your question:", key="kb_question")
    if question:
        st.markdown(f"**You asked:** {question}")

        # Build RAG chain and retrieve documents
        rag_chain, docs = build_answer_chain(question, chat_history=st.session_state.chat_history)

        # Get answer from chain
        result = rag_chain.invoke({"input": question, "chat_history": st.session_state.chat_history})
        st.markdown("**Assistant:**")
        st.write(result["answer"])

        # Update chat history
        st.session_state.chat_history.append(("user", question))
        st.session_state.chat_history.append(("assistant", result["answer"]))

        
        if "" not in result["answer"]:
            sources = {doc.metadata.get("source", "Unknown") for doc in docs}
            st.markdown("**This answer came from:**")
            st.write(", ".join(sources))


elif action == "Translate article to another language":
    language = st.radio(
        "Choose language",
        ("Spanish", "French", "Chinese", "Hindi", "Japanese"),
        key="language_radio"
    )
    url = st.text_input("Enter the URL of the article to translate:", key="translate_url")
    if url:
        web_url, chunks = load_and_chunk(url)
        translation = translate_article(language, chunks, url)
        st.write(translation.page_content)
    else:
        st.write("Please enter url to proceed.")
