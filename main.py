import streamlit as st
from functions import load_and_chunk, summarize_chunks, update_faiss, build_answer_chain, translate_article, rewrite_article, embedding, llm
import asyncio

st.set_page_config(page_title="Student Article Helper")

# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #ADD8E6;  /* your desired color */
#         color: #000000;
#     }
#     /* Style all radio button labels */
#     div[role="main_action_radio"] label {
#         color: red;  
#         font-weight: bold;
#         font-size: 18px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
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
        "Translate article to another language",
        "Rewrite article"
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
        asyncio.run(translate_article(language, chunks, url))
    else:
        st.write("Please enter url to proceed.")

elif action == "Rewrite article":

    url = st.text_input("Enter the url of the article you would like to rewrite: ", key = "rewrite_url")
    if url:
        url, chunks = load_and_chunk(url)
        style = st.text_input("Enter how you would like to rewrite the article (ex. 'simpler, suitable for a 5th grader' or 'more professional and formal' or 'translate to spanish'))")
        rewrite_article(style, chunks, url)
    else:
        st.info("Please enter a URL to proceed.")

    

