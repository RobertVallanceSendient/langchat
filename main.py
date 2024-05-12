from backend.core import run_llm
import streamlit as st
from typing import Set

from streamlit_chat import message

def create_sources_string(source_urls: Set[str]) -> str:
    """Create a string with the sources of the documents in the response."""
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    backslash_char = "\\"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source.replace(backslash_char, '/').replace('///', '//')}\n"
    return sources_string


st.header("Langchain Documentation Helper Bot")

prompt = st.text_input("Prompt:", placeholder="Enter your prompt here...")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
        sources = set([doc.metadata["source"] for doc in generated_response["source_documents"]])
        
        formatted_response = f"{generated_response['answer']} \n\n {create_sources_string(sources)}"

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append((prompt, generated_response["answer"]))

if st.session_state["chat_answers_history"]:
    for user_query, generated_response in zip(st.session_state["user_prompt_history"], st.session_state["chat_answers_history"]):
        message(user_query, is_user=True)
        message(generated_response)
        
