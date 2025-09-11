from dotenv import load_dotenv
import os
import streamlit as st
from pydantic import BaseModel
import asyncio
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# LangChain imports
from langchain.output_parsers import PydanticOutputParser
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever, LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.document_loaders import WebBaseLoader

# -----------------------------
# Setup
# -----------------------------
load_dotenv()  # Load environment variables (including OpenAI API key)
api_key = os.environ.get("OPENAI_API_KEY")

#check if api_key is defined and valid
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. "
        "Please create a .env file with your key:\n"
        "OPENAI_API_KEY=sk-..."
    )

embedding = OpenAIEmbeddings(openai_api_key=api_key) # Embedding model (used later for FAISS vector store)
# -----------------------------
# Setup LLM (OpenAI GPT-4o)
# -----------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)

# -----------------------------
# Load webpage and split into chunks
# -----------------------------
def load_and_chunk(web_url):
    # web_url = input("Enter url to summarize: ")
    loader = WebBaseLoader(web_url)
    # Load documents safely
    try:
        docs = loader.load()
    except Exception as e:
        st.write("Error loading document: ", e)
        docs = []
    if not docs:
        st.write("No content to process")
        exit()

    # Break documents into smaller chunks (1000 tokens each, no overlap)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=3000,
        chunk_overlap=200
)
    chunks = text_splitter.split_documents(docs)
    for chunk in chunks:
        if not hasattr(chunk, "metadata") or chunk.metadata is None:
            chunk.metadata = {}
        chunk.metadata = {"source": web_url}
    return web_url, chunks


def summarize_chunks(chunks, llm, web_url):
# -----------------------------
# Map step → summarize each chunk
# -----------------------------
    map_template = "Summarize the following text in 3–4 sentences, highlighting key facts and important details. Ignore filler or repetition: {docs}."
    map_prompt = ChatPromptTemplate([("human", map_template)])
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
# -----------------------------
# Reduce step → combine chunk summaries
# -----------------------------
    reduce_template = """ 
    The following is a set of summaries:
    {docs}
    Take these and distill it into a final, consolidated summary
    of the main themes. Make it professional, clear, and natural.
    """
    reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Combine documents into a single string for final summary
    combine_documents_chain = StuffDocumentsChain(
     llm_chain=reduce_chain, document_variable_name="docs"
    )

    # Reduce chain handles combining summaries (collapsing if too long)
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=1000,  # max tokens allowed for combining
    )

    # MapReduce chain = summarize chunks → combine → final summary
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
    )

    # Run summarization
    result = map_reduce_chain.invoke(chunks)

    # Store the summary as a Document (with source metadata)
    summary_doc = Document(page_content=result["output_text"], metadata={"source": web_url})
    return summary_doc
    
    
# -----------------------------
# Save or update FAISS vector store
# -----------------------------
def update_faiss(docs, embedding):
    """
    Add new documents to FAISS index only if they are not already present.
    """
    if not isinstance(docs, list):
        docs = [docs]

    if not os.path.exists("faiss_index"):
        # First time → create new index
        my_vectorstore = FAISS.from_documents(docs, embedding)
    else:
        # Load existing index once
        my_vectorstore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

        # Extract existing contents from docstore
        existing_texts = [d.page_content for d in my_vectorstore.docstore._dict.values()]

        # Only keep docs that aren’t already in the index
        new_docs = [doc for doc in docs if doc.page_content not in existing_texts]

        if new_docs:
            my_vectorstore.add_documents(new_docs)

    # Save updated index
    my_vectorstore.save_local("faiss_index")



def build_answer_chain(query, chat_history=[]):
    my_vectorstore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    retriever = my_vectorstore.as_retriever()
    # Prompt for reformulating follow-up questions into standalone ones
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that reformulates follow-up questions into standalone questions."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Prompt for answering based on retrieved context
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a careful assistant that answers based only on the provided articles. "
                "If you don't know, you must say that. Be short and concise. Articles: {context}"),
        ("human", "{input}")
    ])

    # Create history-aware retriever (handles conversation memory)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    docs = history_aware_retriever.invoke({"input": query, "chat_history": chat_history})
    
    # Create QA chain (retriever → LLM)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Full retrieval-augmented generation pipeline
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return rag_chain, docs

async def translate_article(language, chunks, web_url):
    map_prompt = PromptTemplate(input_variables=["context", "language"], template = "Translate the following text to {language}:\n\n{context}")
    map_chain = LLMChain(llm = llm, prompt = map_prompt)

    reduce_prompt = PromptTemplate(input_variables = ["context", "language"], template = "Combine the following translated chunks into a single coherent translation in {language}:\n\n {context}")
    reduce_chain = LLMChain(llm = llm, prompt = reduce_prompt)

    reduce_docs_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_variable_name="context"
    )

    reduce_docs_chain = ReduceDocumentsChain(
        combine_documents_chain = reduce_docs_chain,
        collapse_documents_chain = reduce_docs_chain, 
        token_max = 4000
    )

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain = map_chain, 
        reduce_documents_chain = reduce_docs_chain,
        document_variable_name = "context"
                                            )
    
    translated_chunks = []

    for chunk in chunks:
        # Each chunk is a Document, so chunk.page_content has the text
        prompt_input = {
            "context": chunk.page_content,
            "language": language
        }
        translated_text = map_chain.run(prompt_input)
        translated_chunks.append(Document(page_content=translated_text, metadata=chunk.metadata))
        st.write(translated_text)

def rewrite_article(style, chunks, url):
    SUPPORTED_LANGUAGES = [
    "english", "spanish", "french", "german", "italian",
    "portuguese", "hindi", "chinese", "japanese", "korean"
   ]
    if "translate" in style:
        for lang in SUPPORTED_LANGUAGES:
            if lang in style:
                asyncio.run(translate_article(lang, chunks, url))
    else:
        map_prompt = PromptTemplate(input_variables = ["style", "context"], template = "Rewrite the following article chunk to be {style}: {context}. Remember to preserve the actual meaning of the text and all important info, keep it concise")
        reduce_prompt = PromptTemplate(input_variables = ["style", "context"], template = "Take the following rewritten chunks and chain them together to be {style}: {context} to make one coherent rewritten article. Make sure the article flows well.")

        map_chain = LLMChain(llm = llm, prompt = map_prompt)
        reduce_chain = LLMChain(llm = llm, prompt = reduce_prompt)

        reduce_docs_chain = StuffDocumentsChain(llm_chain = reduce_chain, document_variable_name = "context")

        my_reduce_docs_chain = ReduceDocumentsChain(
            combine_documents_chain = reduce_docs_chain,
            collapse_documents_chain = reduce_docs_chain, 
            token_max = 4000

        )

        map_reduce_chain = MapReduceDocumentsChain(llm_chain = map_chain, 
                                                   reduce_documents_chain = my_reduce_docs_chain,
                                                   document_variable_name = "context")
        
        result = map_reduce_chain.invoke({"input_documents": chunks, "style": style})
        st.write(result["output_text"])

def tone_detection(chunks):
    class ToneAnalysis(BaseModel):
        tone: str
        justification: str
    parser = PydanticOutputParser(pydantic_object=ToneAnalysis)
    tone_prompt = ChatPromptTemplate.from_template("""
    Analyze the following text and return the tone analysis.

    {text}

    {format_instructions}
    """).partial(format_instructions=parser.get_format_instructions())


    tone_chain = tone_prompt | llm | parser
    result = tone_chain.invoke({"text": chunks})
    st.write(result.tone)
    st.write(result.justification)
    
def most_similar_docs(chunks, top_k = 5):
    #compare the chunks of new article to the existing chunks
    #if similar 
    embedding_model = OpenAIEmbeddings()
    new_embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])
    my_vectorstore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization = True)
    #store similarity scores per document
    doc_sims = defaultdict(list)

    for chunk, emb in zip(chunks, new_embeddings):
        results = my_vectorstore.similarity_search_by_vector(emb, k=top_k)
        for doc in results:
            url = doc.metadata["source"]
            score = doc.metadata.get("score", 1.0)
            doc_sims[url].append(score)
    
    avg_doc_sims = {url: sum(scores)/len(scores) for url, scores in doc_sims.items()}

    most_similar = dict(sorted(avg_doc_sims.items(), key = lambda x: x[1], reverse = True))
    
    return most_similar
    #compare the new_embeddings to the existing_embeddings and find the new embeddings that are really similar to existing ones 



    

#     {
#   new_chunk_1: [(existing_chunk_1, 0.8), (existing_chunk_2, 0.9), ...],
#   new_chunk_2: [(existing_chunk_1, 0.7), (existing_chunk_2, 0.5), ...],
#   ...
# }
    
    #find the similarity of each chunk to each existing chunk
    #then get the chunks from each url and compute the average similarities
    #then average the similarities for each doc 
        
    
    
    
#wanna find the 



    


def start_chat():
    chat_history = []
    while True:
        user_option = input("Pick an option:\n 1) Summarize a text\n 2) Ask about a saved text\n Type 'exit' to quit: ")

        if user_option == "exit":
            st.write("Goodbye 👋")
            break

        elif user_option == "1":
            url_chunks = load_and_chunk()
            summary = summarize_chunks(url_chunks[1], llm, url_chunks[0])
            update_faiss(summary, embedding)
            st.write("\n--- Summary ---\n", summary.page_content, "\n")

        elif user_option == "2":
            query = input("User: ")
            rag_chain = build_answer_chain(query)
            result = rag_chain[0].invoke({"input": query, "chat_history": chat_history})
            st.write("Assistant: ", result["answer"])
            chat_history.append(("user", query))
            chat_history.append(("assistant", result["answer"]))

            

        else:
            print("Invalid input, try again.")






