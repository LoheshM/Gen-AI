from fastapi import FastAPI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langserve import add_routes
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI
app = FastAPI(
    title="Langchain Tools Server",
    version="1.0",
    description="A server for Langchain-based tools"
)

# Wikipedia and Arxiv tools
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

# Document loader and retriever
loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vectordb.as_retriever()
retriever_tool = create_retriever_tool(retriever, "langsmith_search", "Search for LangSmith-related information.")

# Combine tools
tools = [wiki_tool, arxiv_tool, retriever_tool]

# LLM and agent
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Add routes for tools
add_routes(
    app,
    agent_executor,
    path="/agent"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
