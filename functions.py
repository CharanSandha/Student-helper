from dotenv import load_dotenv
import os

# LangChain imports
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
def load_and_chunk():
    web_url = input("Enter url to summarize: ")
    loader = WebBaseLoader(web_url)
    # Load documents safely
    try:
        docs = loader.load()
    except Exception as e:
        print("Error loading document: ", e)
        docs = []
    if not docs:
        print("No content to process")
        exit()

    # Break documents into smaller chunks (1000 tokens each, no overlap)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=0
)
    chunks = text_splitter.split_documents(docs)
    return [web_url, chunks]


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
def update_faiss(summary, embedding):
    if not os.path.exists("faiss_index"):
        # First time → create new index
        my_vectorstore = FAISS.from_documents([summary], embedding)
        my_vectorstore.save_local("faiss_index")
    else:
        # Load existing index and add new summary
        my_vectorstore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
        my_vectorstore.add_documents([summary])
        my_vectorstore.save_local("faiss_index")


def build_answer_chain():
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
        ("system", "You are a careful assistant that answers based only on the provided context. "
                "If you don't know, say so. Be short and concise. Context: {context}"),
        ("human", "{input}")
    ])

    # Create history-aware retriever (handles conversation memory)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Create QA chain (retriever → LLM)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Full retrieval-augmented generation pipeline
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return rag_chain

def start_chat():
    chat_history = []
    while True:
        user_option = input("Pick an option:\n 1) Summarize a text\n 2) Ask about a saved text\n Type 'exit' to quit: ")

        if user_option == "exit":
            print("Goodbye 👋")
            break

        elif user_option == "1":
            url_chunks = load_and_chunk()
            summary = summarize_chunks(url_chunks[1], llm, url_chunks[0])
            update_faiss(summary, embedding)
            print("\n--- Summary ---\n", summary.page_content, "\n")

        elif user_option == "2":
            rag_chain = build_answer_chain()
            query = input("User: ")
            result = rag_chain.invoke({"input": query, "chat_history": chat_history})
            print("Assistant: ", result["answer"])
            chat_history.append(("user", query))
            chat_history.append(("assistant", result["answer"]))

        else:
            print("Invalid input, try again.")

start_chat()





