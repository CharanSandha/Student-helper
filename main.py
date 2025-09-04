import streamlit as st
from functions import load_and_chunk, summarize_chunks, update_faiss,build_answer_chain, embedding, llm

st.title("Student Article Helper")

st.write("Welcome! Use this app to help with student articles.")

url = st.text_input("Enter the url of the article: ")
chat_history = []
while True:
    action = st.radio(
        "What do you want to do?",
            ("Summarize an article", "Get an answer from your knowledge base")
        )
    if action == "Summarize":
        st.write("You chose to summarize the article")
        url_chunks = load_and_chunk()
        summary = summarize_chunks(url_chunks[1], llm, url_chunks[0])
        update_faiss(summary, embedding)
        st.write("\n--- Summary ---\n", summary.page_content, "\n")
    elif action == "Answer Questions":
        question = st.text_input("Enter your question: ")
        if question:
            st.write("You asked {question}")
            rag_chain = build_answer_chain()
            result = rag_chain.invoke({"input": question, "chat_history": chat_history})
            st.write("Assistant: ", result["answer"])
            chat_history.append(("user", question))
            chat_history.append(("assistant", result["answer"]))
