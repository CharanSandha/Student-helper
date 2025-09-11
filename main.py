import streamlit as st
from functions import load_and_chunk, summarize_chunks, update_faiss, build_answer_chain, translate_article, rewrite_article, tone_detection, most_similar_docs, embedding, llm
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

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "article_text" not in st.session_state:
    st.session_state.article_text = None

# --------------------- Main action ---------------------
url = st.text_input("Enter article URL")

if url and st.session_state.article_text is None:
    url, article_text = load_and_chunk(url)
    st.session_state.article_text = article_text

feature = st.radio("Choose feature:", ["Summary", "Rewrite", "Translation", "Q&A", "Tone Detection", "Check Similarity"])

# Only proceed if we have article text
if st.session_state.article_text:
    text = st.session_state.article_text
    update_faiss(text, embedding)


    # --------------------- Summarize an article ---------------------
    if feature == "Summary":
        summary = summarize_chunks(text, llm, url)
        update_faiss(summary, embedding)
        st.markdown("---")
        st.subheader("Summary")
        st.write(summary.page_content)

    # --------------------- Get an answer from KB ---------------------
    elif feature == "Q&A":
        question = st.text_input("Enter your question:", key="kb_question")
        if question:
            st.markdown(f"**You asked:** {question}")
            rag_chain, docs = build_answer_chain(question, chat_history=st.session_state.chat_history)
            result = rag_chain.invoke({"input": question, "chat_history": st.session_state.chat_history})
            st.markdown("**Assistant:**")
            st.write(result["answer"])
            st.session_state.chat_history.append(("user", question))
            st.session_state.chat_history.append(("assistant", result["answer"]))
            if "" not in result["answer"]:
                sources = {doc.metadata.get("source", "Unknown") for doc in docs}
                st.markdown("**This answer came from:**")
                st.write(", ".join(sources))

    elif feature == "Translation":
        language = st.radio(
            "Choose language",
            ("Spanish", "French", "Chinese", "Hindi", "Japanese"),
            key="language_radio"
        )
        asyncio.run(translate_article(language, text, url))

    elif feature == "Rewrite":
        style = st.text_input(
            "Enter how you would like to rewrite the article (ex. 'simpler, suitable for a 5th grader' or 'more professional and formal' or 'translate to spanish')"
        )
        rewrite_article(style, text, url)

    elif feature == "Tone Detection":
        tone_detection(text)
    
    elif feature == "Check Similarity":
        most_similar = most_similar_docs(text, 6)
        sliced = dict(list(most_similar.items())[1:])
        for url, score in sliced.items():
            st.write(f"{url} — similarity: {score:.2f}")


else:
    st.info("Please enter a URL to load an article first.")
