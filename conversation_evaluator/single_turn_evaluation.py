from ragas.metrics import AspectCritic, SimpleCriteriaScore
from ragas.dataset_schema import MultiTurnSample, EvaluationDataset, SingleTurnSample
from ragas.messages import HumanMessage, AIMessage
from ragas import evaluate
import json
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import os
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
load_dotenv(override=True)
# print(os.environ['GOOGLE_API_KEY'])

# evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Choose the appropriate import based on your API:
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize with Google AI Studio
evaluator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
    model="gemini-2.0-flash"
))

paths = [
# add your paths here
]

for path in paths[:3]:
    print(path)
    with open(path) as f:
        data = json.load(f)
    
    list_cconversatiuons = []
    list_singleturn_conversations = []
    for conversation in data:
        conversationlist = conversation['conversation']
        userInput = []
        for turn in conversationlist:
            userInput.append(HumanMessage(content=turn['rag_input']))
            userInput.append(AIMessage(content=turn['rag_answer']))
            retrieved_contexts = []
            for doc in turn['context']:
                retrieved_contexts.append(doc['page_content'])
            list_singleturn_conversations.append(
                SingleTurnSample(
                    user_input=turn['question'],
                    reference=turn['answer'],
                    response=turn['rag_answer'],
                    retrieved_contexts=retrieved_contexts,
                    reference_contexts=[conversation['document']]
                )
            )
        sample_conversation = MultiTurnSample(
            user_input=userInput
            
        )
        list_cconversatiuons.append(sample_conversation)
        
    definition_correctness = "Return 1 if the AI answers the question correct; otherwise, return 0."
    correctness_aspect_critic = AspectCritic(
        name="correctness_aspect_critic",
        definition=definition_correctness,
        llm=evaluator_llm,
    )

    result = evaluate(
        dataset=EvaluationDataset(samples=list_singleturn_conversations),
        metrics=[correctness_aspect_critic,faithfulness, context_precision, context_recall],
        llm=evaluator_llm,
    )
    result_json = []
    results = result.to_pandas()
    for _, row in results.iterrows():
        user_input = row["user_input"]      
        correctness_aspect_critic = row["correctness_aspect_critic"]
        faithfulness_critic = row["faithfulness"]
        context_precision_critic = row["context_precision"]
        context_recall_critic = row["context_recall"]

    
        
        result_json.append({
            "content": user_input,
            "correctness_aspect_critic": correctness_aspect_critic,
            "faithfulness_aspect_critic": faithfulness_critic,
            "context_precision_critic": context_precision_critic,
            "context_recall_critic": context_recall_critic
        })
    avg_correctness = sum([entry["correctness_aspect_critic"] for entry in result_json]) / len(result_json)
    avg_faithfulness = sum([entry["faithfulness_aspect_critic"] for entry in result_json]) / len(result_json)
    avg_context_precision = sum([entry["context_precision_critic"] for entry in result_json]) / len(result_json)
    avg_context_recall = sum([entry["context_recall_critic"] for entry in result_json]) / len(result_json)
    result_json = {
        "file_path": path,
        "average_scores": {
            "correctness_aspect_critic": avg_correctness,
            "faithfulness_aspect_critic": avg_faithfulness,
            "context_precision_critic": avg_context_precision,
            "context_recall_critic": avg_context_recall
        },
        "detailed_results": result_json
    }
    print(avg_correctness, avg_faithfulness, avg_context_precision, avg_context_recall)
    #modify path to save the results
    final_path = path.replace("conversation_data", "single_turn_evaluation_results").replace(".json", "_evaluation.json")
    with open(final_path, "w") as f:
        json.dump(result_json, f, indent=4)
    import gc
    gc.collect()