import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_pinecone import PineconeVectorStore
from typing import Any, List, Tuple

load_dotenv()


def run_llm(query: str, chat_history: List[Tuple[str, Any]] = []) -> Any:

    embeddings = OpenAIEmbeddings()
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)

    # qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)
    # "stuff" takes context and plugs it into our query
    # as_retriever() is a method that converts the vectorstore into a retriever class
    # see https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/retrieval_qa/prompt.py to understand prompt

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    # same as above function but augments prompt with chat history

    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":

    result = run_llm("Write some langchain code to retrieve documents from Pinecone.")
    print(result)
