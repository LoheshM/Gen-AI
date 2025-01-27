import requests
import streamlit as st

def query_agent(input_text):
    """Query the Langchain agent hosted on the FastAPI server."""
    response = requests.post(
        "http://localhost:8000/agent/invoke",
        json={"input": {"input": input_text}}
    )
    return response.json().get('output', 'No response')

# Streamlit interface
st.title("Langchain Tools Demo")
st.write("Explore Wikipedia, Arxiv, and LangSmith documentation through a unified agent.")

# User input
input_text = st.text_input("Ask a question:")

if input_text:
    with st.spinner("Querying the agent..."):
        response = query_agent(input_text)
    st.write("### Response")
    st.write(response)
