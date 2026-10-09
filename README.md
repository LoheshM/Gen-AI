# Gen-AI: a collection of generative AI mini-projects

**Three small, self-contained experiments with LLMs:** QLoRA fine-tuning of Llama 2 7B on a free Colab GPU, a LangChain tool-using agent served over FastAPI with a Streamlit client, and a YouTube video summariser with Q&A built on Whisper and a local Llama 2.

> Personal learning projects, January 2025 · Python · Hugging Face Transformers / PEFT / TRL, LangChain, LangServe, Haystack, Streamlit

## Projects at a glance

| Project | What it does | Stack | Entry point | State |
|---|---|---|---|---|
| [Llama 2 fine-tuning](#1-llama-2-7b-fine-tuning-with-qlora) | Fine-tunes `Llama-2-7b-chat-hf` on 1,000 instruction samples in 4-bit (QLoRA) and tests generation | transformers 4.31, peft 0.4, trl 0.4.7, bitsandbytes, Colab T4 | [Fine_tune_Llama_2.ipynb](Fine_tune_Llama_2.ipynb) | Run end to end, outputs saved |
| [MultiAgentRAG](#2-multiagentrag-a-tool-using-langchain-agent) | An OpenAI-functions agent that picks between Wikipedia, arXiv and a FAISS retriever over the LangSmith docs | LangChain, LangServe, FastAPI, FAISS, OpenAI, Streamlit | [MultiAgentRAG/app.py](MultiAgentRAG/app.py), [MultiAgentRAG/client.py](MultiAgentRAG/client.py) | Runnable with an OpenAI key |
| [YouTube summariser and Q&A](#3-youtube-video-summariser-and-qa) | Downloads a video's audio, transcribes it with Whisper, summarises with a local Llama 2 GGUF, and answers questions with RAG | pytube, Haystack 1.x, llama.cpp, LangChain, FAISS, Ollama, Streamlit | [Youtube_Q&A_Summarization.py](Youtube_Q&A_Summarization.py) | Prototype, does not run as committed |

---

## 1. Llama 2 7B fine-tuning with QLoRA

### The problem

A free Colab GPU has about 15 GB of VRAM. That is barely enough to hold Llama 2 7B's weights, before optimiser states, gradients and activations. Full fine-tuning won't fit. Parameter-efficient fine-tuning with LoRA adapters on a 4-bit quantised base model (QLoRA) does.

### Approach

| Stage | Detail |
|---|---|
| Base model | `NousResearch/Llama-2-7b-chat-hf` (chat model, so inputs use the `<s>[INST] … [/INST]` template) |
| Dataset | [`mlabonne/guanaco-llama2-1k`](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k): 1,000 samples of [`timdettmers/openassistant-guanaco`](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) reformatted to the Llama 2 prompt template |
| Quantisation | bitsandbytes 4-bit, `nf4`, compute dtype `float16`, no double quantisation |
| LoRA | `r=64`, `alpha=16`, `dropout=0.1`, `task_type=CAUSAL_LM` |
| Training | TRL `SFTTrainer`, 1 epoch, batch size 4, lr 2e-4 cosine, warmup 3%, `paged_adamw_32bit`, gradient checkpointing, `group_by_length`, max grad norm 0.3, logging to TensorBoard |
| Output | LoRA adapter saved as `Llama-2-7b-chat-finetune`, then a `text-generation` pipeline test |

```
guanaco-llama2-1k ──► SFTTrainer ◄── Llama-2-7b-chat-hf (4-bit NF4) + LoRA (r=64, α=16)
                          │
                          ├─► results/runs   (TensorBoard logs)
                          └─► Llama-2-7b-chat-finetune  (adapter) ──► pipeline("text-generation")
```

### Results

From the saved outputs in [Fine_tune_Llama_2.ipynb](Fine_tune_Llama_2.ipynb), run on a Colab T4:

| Metric | Value |
|---|---|
| Steps | 250 (1 epoch, 1,000 samples) |
| Mean training loss | 1.362 |
| Logged loss at steps 25 → 250 | 1.409, 1.662, 1.215, 1.444, 1.177, 1.367, 1.174, 1.468, 1.158, 1.543 |
| Runtime | 1,634 s (about 27 minutes), 0.61 samples/s |

There is no held-out evaluation. The only qualitative check is one prompt, "What is a large language model?", whose generated answer is saved in the last cell.

### Run it

Open the notebook in Google Colab with a GPU runtime and run the cells in order. The first cell installs the pinned versions: `accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7`.

---

## 2. MultiAgentRAG: a tool-using LangChain agent

### What it does

| | |
|---|---|
| **Wikipedia tool** | `WikipediaQueryRun`, top 1 result, 200 characters |
| **arXiv tool** | `ArxivQueryRun`, top 1 result, 200 characters |
| **LangSmith docs retriever** | Loads `https://docs.smith.langchain.com/` with `WebBaseLoader`, splits it into 1,000-character chunks with 200 overlap, embeds them with `OpenAIEmbeddings` into FAISS, and exposes them as the `langsmith_search` tool |
| **Agent** | `create_openai_tools_agent` with `gpt-3.5-turbo-0125` (temperature 0) and the `hwchase17/openai-functions-agent` prompt from LangChain Hub. The agent decides which tool to call. |
| **Serving** | LangServe `add_routes` exposes the `AgentExecutor` at `/agent` on a FastAPI app (`localhost:8000`) |
| **Client** | Streamlit page that posts the question to `/agent/invoke` and shows the response |

```
Streamlit client.py ──POST /agent/invoke──► FastAPI + LangServe (app.py, :8000)
                                              └─ AgentExecutor (gpt-3.5-turbo-0125)
                                                   ├─ wikipedia
                                                   ├─ arxiv
                                                   └─ langsmith_search  (FAISS over docs.smith.langchain.com)
```

### Run it

```bash
pip install fastapi uvicorn langchain langchain-community langchain-openai langchain-text-splitters langserve[all] \
            faiss-cpu wikipedia arxiv beautifulsoup4 python-dotenv streamlit requests
echo "OPENAI_API_KEY=sk-..." > .env
python MultiAgentRAG/app.py              # builds the FAISS index at startup, then serves on :8000
streamlit run MultiAgentRAG/client.py    # in a second terminal
```

There is no `requirements.txt`. The package list above is taken from the imports.

---

## 3. YouTube video summariser and Q&A

### What it is meant to do

| Stage | Component |
|---|---|
| Download audio | `pytube`: the last 160 kbps audio stream of the given URL |
| Transcribe and summarise | Haystack 1.x pipeline: `WhisperTranscriber` → `PromptNode` with the `deepset/summarization` template on a local `llama-2-7b-32k-instruct.Q4_K_S.gguf` via a custom `LlamaCPPInvocationLayer` (CPU, max length 512) |
| Index for Q&A | LangChain `RecursiveCharacterTextSplitter` (1,000 / 200) → FAISS with `OpenAIEmbeddings` |
| Answer questions | Retrieve relevant chunks, then a stuff-documents chain on `Ollama(model="llama2")` |
| UI | Streamlit: video on the left, summary and elapsed time on the right, question box below |

```
YouTube URL ─► pytube (audio) ─► Whisper ─► PromptNode (Llama 2 GGUF, summarise) ─► summary
                                                │
                                                └─► text splitter ─► FAISS ─► retriever ─► Ollama llama2 ─► answer
```

### Run it

The script is a prototype and needs fixes before it runs (see Limitations). It expects:
- `llama-2-7b-32k-instruct.Q4_K_S.gguf` in the working directory.
- A `model_add.py` providing `LlamaCPPInvocationLayer`.
- An Ollama server with the `llama2` model.
- `OPENAI_API_KEY` for embeddings.

```bash
streamlit run "Youtube_Q&A_Summarization.py"
```

---

## Repository layout

```
Fine_tune_Llama_2.ipynb        QLoRA fine-tuning notebook (Colab T4), with outputs (~9 MB)
MultiAgentRAG/
├── app.py                     FastAPI + LangServe server exposing the tool-using agent at /agent
└── client.py                  Streamlit client calling /agent/invoke
Youtube_Q&A_Summarization.py   Streamlit YouTube summariser + Q&A prototype
```

## Limitations

- **No dependency files.** There is no `requirements.txt` for either app. The notebook pins its own 2023-era versions, which are old and have since changed APIs (`SFTTrainer` arguments, `prepare_model_for_int8_training`).
- **The fine-tune has no evaluation.** It reports training loss and one sample generation only. The adapter is saved locally in Colab, not merged or published.
- **The YouTube app does not run as committed:**
  - The entry guard is written `if _name_ == "_main_":`, so `main()` is never called.
  - `model_add` (the `LlamaCPPInvocationLayer`) is not in the repo.
  - `split_documents` is given a plain string instead of `Document` objects.
  - `create_stuff_documents_chain` is given a prompt name string instead of a prompt template.
  - The question box sits inside the `Submit` button branch, so Streamlit's rerun loses the indexed transcript.
  - It mixes Haystack 1.x and LangChain, and `pytube` often breaks when YouTube changes.
- **The MultiAgentRAG client shows the raw LangServe `output` object** rather than just the final answer string. Despite the folder name, it is one agent with three tools, not several cooperating agents.
- **Network at startup.** `app.py` downloads and embeds the LangSmith docs every time it starts, which costs OpenAI embedding calls.

## Credits

- **Llama 2 notebook:**
  - The approach, explanations and hyperparameters follow a public QLoRA tutorial for Colab.
  - Base weights are from `NousResearch/Llama-2-7b-chat-hf`.
  - The dataset is `mlabonne/guanaco-llama2-1k`, derived from `timdettmers/openassistant-guanaco`.
  - The notebook links the [dataset reformatting notebook](https://colab.research.google.com/drive/1Ad7a9zMmkxuXTOh1Z7-rNSICA4dybpM2?usp=sharing).
- **Agent prompt:** `hwchase17/openai-functions-agent` from LangChain Hub.

## Licence

MIT. See [LICENSE](LICENSE). The licence covers the code in this repository only. Llama 2 weights are under Meta's Llama 2 Community License. The datasets, the OpenAI API, Whisper and the other libraries keep their own licences and terms.
