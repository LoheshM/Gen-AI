import streamlit as st
from pytube import YouTube
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from model_add import LlamaCPPInvocationLayer
import time
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader

st.set_page_config(layout="wide")

def download_video(url):
    yt = YouTube(url)
    video = yt.streams.filter(abr='160kbps').last()
    return video.download()

def initialize_model(full_path):
    return PromptModel(
        model_name_or_path=full_path,
        invocation_layer_class=LlamaCPPInvocationLayer,
        use_gpu=False,
        max_length=512
    )

def initialize_prompt_node(model):
    summary_prompt = "deepset/summarization"
    return PromptNode(model_name_or_path=model, default_prompt_template=summary_prompt, use_gpu=False)

def transcribe_audio(file_path, prompt_node):
    whisper = WhisperTranscriber()
    pipeline = Pipeline()
    pipeline.add_node(component=whisper, name="whisper", inputs=["File"])
    pipeline.add_node(component=prompt_node, name="prompt", inputs=["whisper"])
    output = pipeline.run(file_paths=[file_path])
    return output

def setup_rag_chain(transcription_text):
    docs = [transcription_text]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)
    vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
    retriever = vectordb.as_retriever()
    llm = Ollama(model="llama2")
    document_chain = create_stuff_documents_chain(llm, prompt="deepset/question-answering")
    return document_chain, retriever

def answer_question(question, document_chain, retriever):
    retrieved_docs = retriever.get_relevant_documents(question)
    response = document_chain.run(input_documents=retrieved_docs, question=question)
    return response

def main():
    st.title("YouTube Video Summarizer and Q&A Bot 🎥")
    st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
    st.subheader('Built with Llama 2 , Haystack, Langchain, Streamlit')
    st.markdown('<style>h3{color: pink;  text-align: center;}</style>', unsafe_allow_html=True)

    with st.expander("About the App"):
        st.write("This app allows you to summarize YouTube videos and ask questions about their content.")
        st.write("Enter a YouTube URL in the input box below and click 'Submit' to start.")

    youtube_url = st.text_input("Enter YouTube URL")

    if st.button("Submit") and youtube_url:
        start_time = time.time()
        file_path = download_video(youtube_url)
        full_path = "llama-2-7b-32k-instruct.Q4_K_S.gguf"
        model = initialize_model(full_path)
        prompt_node = initialize_prompt_node(model)
        transcription_output = transcribe_audio(file_path, prompt_node)
        transcription_text = transcription_output["results"][0]
        document_chain, retriever = setup_rag_chain(transcription_text)
        end_time = time.time()
        elapsed_time = end_time - start_time

        col1, col2 = st.columns([1,1])

        with col1:
            st.video(youtube_url)

        with col2:
            st.header("Summarization of YouTube Video")
            st.write(transcription_output)
            st.success(transcription_text.split("\n\n[INST]")[0])
            st.write(f"Time taken for summarization: {elapsed_time:.2f} seconds")

        question = st.text_input("Ask a question about the video:")

        if question:
            answer = answer_question(question, document_chain, retriever)
            st.write("Answer:", answer)

if _name_ == "_main_":
    main()