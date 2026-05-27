import os
from typing import List, Dict, TypedDict
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.vectorstores import FAISS
from langchain_classic.schema import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END, START

# -----------------------------
# 📦 MOCK DATA INGESTION
# -----------------------------
from langchain_classic.document_loaders import PyPDFLoader, CSVLoader
import pandas as pd
from langchain_classic.schema import Document
import os

def load_documents():
    docs = []

    data_path = "./data"  # folder with your files

    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)

        # ---------------- PDF ----------------
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            pdf_docs = loader.load()

            # Add metadata for RBAC
            for doc in pdf_docs:
                doc.metadata["department"] = "HR"  # customize logic
            docs.extend(pdf_docs)

        # ---------------- CSV ----------------
        elif file.endswith(".csv"):
            df = pd.read_csv(file_path)

            for _, row in df.iterrows():
                content = " | ".join([f"{col}: {row[col]}" for col in df.columns])
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"department": "Finance"}  # customize
                    )
                )

        # ---------------- JSON ----------------
        elif file.endswith(".json"):
            df = pd.read_json(file_path)

            for _, row in df.iterrows():
                content = str(row.to_dict())
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"department": "Engineering"}
                    )
                )

    return docs

# -----------------------------
# ✂️ CHUNKING + EMBEDDING
# -----------------------------
def create_vector_store(documents: List[Document]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


# -----------------------------
# 🔐 RBAC FILTERING
# -----------------------------
def filter_docs_by_role(docs: List[Document], user_role: str) -> List[Document]:
    return [doc for doc in docs if doc.metadata.get("department") == user_role]


# -----------------------------
# 🧠 STATE DEFINITION (LangGraph)
# -----------------------------
class RAGState(TypedDict):
    query: str
    role: str
    retrieved_docs: List[Document]
    answer: str


# -----------------------------
# 🔎 RETRIEVAL NODE
# -----------------------------
def retrieve_node(state: RAGState):
    query = state["query"]
    role = state["role"]

    docs = vectorstore.similarity_search(query, k=5)

    # Apply RBAC filtering
    filtered_docs = filter_docs_by_role(docs, role)

    return {"retrieved_docs": filtered_docs}


# -----------------------------
# 🤖 GENERATION NODE
# -----------------------------
def generate_node(state: RAGState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    docs = state["retrieved_docs"]
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = ChatPromptTemplate.from_template("""
You are an enterprise AI assistant.

Answer the question ONLY using the context below.
If the answer is not found, say "Not enough information."

Context:
{context}

Question:
{query}
""")

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "query": state["query"]
    })

    return {"answer": response.content}


# -----------------------------
# 🔁 BUILD LANGGRAPH
# -----------------------------
def build_graph():
    builder = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    builder.set_entry_point("retrieve")

    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


# -----------------------------
# 🚀 MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    # Load + index documents
    docs = load_documents()
    vectorstore = create_vector_store(docs)

    # Build graph
    graph = build_graph()

    # Example query
    input_state = {
        "query": "What are the latest HR policies?",
        "role": "HR",  # Change role to test RBAC
        "retrieved_docs": [],
        "answer": ""
    }

    result = graph.invoke(input_state)

    print("\n=== FINAL ANSWER ===\n")
    print(result["answer"])