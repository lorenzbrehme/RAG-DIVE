# System Architecture Overview

The industrial RAG system is organized around two primary components: **Data Integration** and **Retrieval & Generation**. Its design emphasizes modularity, scalability, and adaptability to heterogeneous data sources and formats.

## Data Integration Component

The modular architecture allows seamless integration of diverse data sources (e.g., issue trackers, code repositories) and data types (e.g., emails, files, wiki pages, project artifacts).

In the initial stage of the pipeline, raw artifacts are converted into Markdown format (e.g., from Textile). Markdown was chosen due to its high interpretability by LLMs, lightweight syntactic markers, and extensive support for transformation into other formats (e.g., PDF → Markdown).  
During this conversion process, redundant or repetitive sequences (e.g., excessive white spaces, dashed table headers) are removed to reduce document size. This optimization minimizes the storage footprint and accelerates subsequent chunking and loading processes.

Each Markdown document is segmented by headings to preserve document hierarchy. Within each section, the contents are divided into paragraphs, tables, lists, and code blocks — an approach supported by prior research. The resulting text blocks are then paired with their corresponding headings and tokenized to fit within the embedding model’s context window (e.g., multilingual-e5-large-instruct).  
If a block exceeds the model’s limit, it is subdivided while retaining headings and table headers to maintain semantic context. Each chunk is augmented with its source metadata and timestamp before dense and sparse representations (e.g., cosine embeddings and BM25 vectors) are computed and stored in **Parquet** format for efficient retrieval.

## Retrieval and Generation Component

The system’s backend and frontend are implemented in a performant language, enabling shared data structures and logic. The frontend transmits the chat history to the backend, which communicates via HTTP with registered RAG microservices to retrieve relevant chunks.  
Each RAG source employs a **hybrid retrieval** approach combining BM25 and semantic similarity search (cosine distance), leveraging their complementary strengths.

After all sources return their results, duplicate chunks are removed. The remaining results are reranked using a **small language model (SLM)** reranker (e.g., bge-reranker-v2-m3) to filter out irrelevant information.  
Due to the reranker’s context limitations, chunks are grouped by token length and processed iteratively until the desired subset remains.

A key capability of the system is its **function-calling mechanism** for ChatOps. A specialized SLM (e.g., Qwen3-4B-Thinking-2507) dynamically constructs prompts to include recent chat history, available function descriptions, and selected chunks while respecting the model’s context constraints.  
Older messages are progressively pruned to maximize relevant input. Once the model determines which functions to invoke, registered endpoints (e.g., project management systems) are called, and their responses are appended to the retrieved context.

Finally, the user-facing response is generated through a similar iterative prompting process. The answer is streamed token by token, and the list of contributing chunks is transmitted alongside it to support verification and transparency.

---

*Note:* All specific model names, system identifiers have been anonymized or generalized for publication purposes.
