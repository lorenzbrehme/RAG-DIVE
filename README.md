# RAG-DIVE

 **A Dynamic Approach for Multi-Turn Dialogue Evaluation in Retrieval-Augmented Generation**

This repository accompanies the results and raw data from our paper **“RAG-DIVE:  A Dynamic Approach for Multi-Turn Dialogue Evaluation in Retrieval-Augmented Generation”**, which explores conversation generation, validation, and evaluation within a Retrieval-Augmented Generation (RAG) framework.


---

## Repository Overview

This repository contains the **data, code, and configurations** used for all experiments presented in the paper.

### Folder Structure

| Folder | Description |
|--------|-------------|
| **`data/`** | Contains all generated conversations and corresponding logs. Each experiment folder includes:<br>• One JSON file per run (conversation data)<br>• Log file (Conversation Validator output)<br>• Metric files:<br> – *Multiturn metrics*: forgetfulness, context retention<br> – *Single-turn metrics*: correctness, faithfulness, context precision, context recall |
| **`data/industrial_usecase/`** | Includes the industrial use case experiments:<br>• `SQuAD_evaluation/` – SQuAD-based RAG evaluation<br>• `RAG-DIVE_evaluation/` – Evaluation results of our RAG-DIVE setup |
| **`human_validation/`** | Contains Excel sheets with human validation results of generated conversations.<br>The main dataset used: `data/data_experiment_1_10x5turn/conversation_data_gemini-2.0-flash_turns_5_conversation_100_0.json`.<br>To view the conversation, open `viewer.html` in your browser. |
| **`conversation_generator/`** | Code used for **synthetic conversation generation** with multiple model configurations and personas. |
| **`conversation_validator/`** | Code for **automatic validation** of generated conversations. |
| **`conversation_evaluator/`** | Evaluation scripts for **RAG performance metrics**, including RAGAS-based assessments. |
| **`Models/`** | Contains model configuration and integration scripts:<br>• `chat_gpt.py` – OpenAI GPT integration<br>• `gemini.py` – Google Gemini integration |
| **`rag_to_be_tested/`** | The RAG system under test. To initialize:<br>• Build the PGVector database (`vectordb.ipynb`)<br>• Start the RAG FastAPI service (`main.py`) |
| **`industrial_use_case/`** | Code for the industrial use case experiments, including Conversation Generator (CG), Conversation Validator (CV), and evaluation scripts inside for SQuAD evalaution `RAG_Evaluation_Standard/`. Data is stored in `single-hop-RAG-dataset/`. |


---

## Getting Started

### Prerequisites

- **Docker**
- **Python 3.9+**
- **Uvicorn**

---

### Environment Setup

1. **Copy the environment template**
  
   ```bash
      cp .env.example .env
   ````

2. **Configure API keys**

   Edit `.env` to include your API credentials:

   ```bash
   GOOGLE_API_KEY=your_google_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

---

### Start the RAG System

1. **Start the Docker container**

   ```bash
   sudo docker start <container_id>   # container_id obtained after running vectordb.ipynb
   ```

2. **Launch the RAG FastAPI application**

   ```bash
   cd rag_to_be_tested/
   uvicorn main:app --reload
   ```

3. **Start the Conversation Generator in a new terminal**

   ```bash
   cd conversation_generator/
   python3 conversation_generation.py
   ```

4. **After finishing, stop the Docker container**

   ```bash
   sudo docker stop <container_id>
   ```

---

## Components Overview



### Conversation Generator ([`conversation_generator/`](conversation_generator/))

Generates **synthetic conversations** for RAG evaluation.

* Supports multiple models (GPT, Gemini)
* Role-based generation (e.g., “precise/expert”, “confused” personas)

---

### Conversation Validator ([`conversation_validator/`](conversation_validator/))

Automatically checks generated conversations

---

### Conversation Evaluator ([`conversation_evaluator/`](conversation_evaluator/))

Implements **RAGAS-based evaluation** and custom metrics:

* *Single-turn*: correctness, faithfulness, context precision, context recall
* *Multi-turn*: forgetfulness, context retention

---

### Models ([`Models/`](Models/))

Integrations for large language models:

* **`chat_gpt.py`** – OpenAI GPT configuration
* **`gemini.py`** – Google Gemini API configuration

---

### RAG System ([`rag_to_be_tested/`](rag_to_be_tested/))

Implements the syntetic RAG pipeline.

* **`main.py`** – FastAPI application serving QA endpoints
* **`qa_chain.py`** – RAG implementation using LangGraph, integrating:

  * Google Generative AI embeddings
  * PGVector for document storage
  * Multi-step retrieval and generation pipeline
  * Contains ClapNQ data

---

###  Industrial Use Case ([`industrial_use_case/`](industrial_use_case/))

Demonstrates the RAG-DIVE framework in a real-world industrial setting.

* Includes code for CG, CV, and evaluation
* Uses data from `single-hop-RAG-dataset/`
* Contains SQuAD and RAG-Evaluation pipelines

---

## Metrics Summary

| Metric Type     | Metrics                                                      |
| --------------- | ------------------------------------------------------------ |
| **Single-Turn** | Correctness, Faithfulness, Context Precision, Context Recall |
| **Multi-Turn**  | Forgetfulness, Context Retention                             |

---

