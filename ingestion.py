import os
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(
        path="langchain-docs/api.python.langchain.com/en/latest/", encoding="utf-8"
    )
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Split into {len(documents)} documents")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} documents to Pinecone")
    embeddings = OpenAIEmbeddings()
    vector_store = PineconeVectorStore.from_documents(
        documents, embeddings, index_name=os.environ["PINECONE_INDEX_NAME"]
    )
    print("*** Added to Pinecone vectorstore vectors")


if __name__ == "__main__":
    ingest_docs()
