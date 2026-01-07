from ragas.metrics import AspectCritic, SimpleCriteriaScore
from ragas.dataset_schema import MultiTurnSample, EvaluationDataset, SingleTurnSample
from ragas.messages import HumanMessage, AIMessage
from ragas import evaluate
import json
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import os
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

for path in paths:
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
        

    definition_forgetfulness = "Return 1 if the AI forgets relevant information from earlier in the conversation, showing loss of context or failure to follow up on prior turns; otherwise, return 0."
    forgetfulness_aspect_critic = AspectCritic(
        name="forgetfulness_aspect_critic",
        definition=definition_forgetfulness,
        llm=evaluator_llm,
    )
    definition_contextretention = "Return 1 if the AI clearly retains relevant information from earlier in the conversation, demonstrating strong understanding and continuity across turns; otherwise, return 0."
    context_retention_aspect_critic = AspectCritic(
        name="context_retention_aspect_critic",
        definition=definition_contextretention,
        llm=evaluator_llm,
    )
    result = evaluate(
        dataset=EvaluationDataset(samples=list_cconversatiuons),
        metrics=[forgetfulness_aspect_critic, context_retention_aspect_critic],
    )
    result_json = []
    results = result.to_pandas()
    for _, row in results.iterrows():
        user_input = row["user_input"]    
        forgetfullness_critic = row["forgetfulness_aspect_critic"]
        context_critic = row["context_retention_aspect_critic"]

        for entry in user_input:
            result_json.append({
                "content": entry["content"],
                "forgetfulness_aspect_critic": forgetfullness_critic,
                "context_retention_aspect_critic": context_critic
            })
    avg_forgetfulness = sum([entry["forgetfulness_aspect_critic"] for entry in result_json]) / len(result_json)
    avg_context_retention = sum([entry["context_retention_aspect_critic"] for entry in result_json]) / len(result_json)
    result_json = {
        "file_path": path,
        "average_forgetfulness_aspect_critic": avg_forgetfulness,
        "average_context_retention_aspect_critic": avg_context_retention,
        "detailed_results": result_json
    }
    print(avg_forgetfulness, avg_context_retention)
    #modify path to save the results
    final_path = path.replace("conversation_data", "evaluation_results").replace(".json", "_evaluation.json")
    with open(final_path, "w") as f:
        json.dump(result_json, f, indent=4)