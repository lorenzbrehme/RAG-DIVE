# qwen 30B 
- embedding: bme-large-v2-q8_0.gguf
- generation: qwen-30b
- context-size: 2050
- evaluation: open-AI 4o-mini

# qwen 30B Qwen3-30B-A3B-Instruct-2507-Q6_K 
- embedding: multi lang
- {'answer_relevancy': 0.8617, 'context_precision': 0.5363, 'faithfulness': 0.7838, 'context_recall': 0.7900}

# Qwen3-30B-A3B-Thinking 
- embedding: : multi lang
- {'answer_relevancy': 0.8131, 'context_precision': 0.5336, 'faithfulness': 0.7430, 'context_recall': 0.7900}


# Qwen_Qwen3-4B-Instruct-2507-Q8
- embedding:  multi lang
- {'answer_relevancy': 0.7966, 'context_precision': 0.5209, 'faithfulness': 0.7912, 'context_recall': 0.7900}


# qwen 30B Qwen3-30B-A3B-Instruct-2507-Q6_K (neues Chunking)
- embedding: multi lang
{'answer_relevancy': 0.8905, 'context_precision': 0.7020, 'faithfulness': 0.7646, 'context_recall': 0.8200}

# qwen 30B Qwen3-30B-A3B-Instruct-2507-Q6_K (neues Chunking)
- embedding: bm25 + reranker 
{'answer_relevancy': 0.8769, 'context_precision': 0.9187, 'faithfulness': 0.8965, 'context_recall': 0.9700}