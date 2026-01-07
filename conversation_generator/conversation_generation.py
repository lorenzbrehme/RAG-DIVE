import requests
import sys
import os

# Füge das übergeordnete Verzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Models.gemini as gemini
import Models.chat_gpt as chat_gpt
import templates as templates
import utils.parser as parser
import time
import json
import conversation_validator.conversation_validator as conversation_validator
# API Endpoint
API_URL = "http://localhost:8000/rag"
API_URL_THREAD_ID = "http://localhost:8000/getThreadID"
# model_name = 'gpt-5'  # Change to 'gemini-2.0-flash' if needed'
# model = chat_gpt.ChatGPT(model_name)
model_name = 'gemini-2.0-flash'  
# model_name = 'gemini-2.5-flash-lite'
# model_name = 'gemini-2.5-flash'  
model = gemini.GEMINI(model_name)
parser = parser.LLMResponseParser()
# file_path = "../rag_to_be_tested-Data/clapnq.jsonl"
max = 100 # number conversations'
n=5 # number turns
file_path = "../rag_to_be_tested/RAG-Data/clapnq_dev_answerable_orig.jsonl"
# data_storage_path = "./data/conversation_data2.csv"
output_file = "./data/conversation_data_" + model_name + "_turns_" + str(n) + "_conversation_" +str(max)+ ".jsonl"
log_file = "./data/conversation_data_" + model_name + "_turns_" + str(n) + "_conversation_" +str(max)+ ".log"
Role = "Your questions are very short and precise"
# Role = "You are a highly attentive conversationalist who asks context-aware questions. Your questions should build naturally on previous exchanges, using referring expressions like 'this', 'that', or 'it' to maintain coherence and continuity."
# Role = "You are a very confused and forgetful person who always misunderstands what has been said. You repeatedly ask the same questions as if you never heard the answer, often mixing up details and getting things wrong. Your questions are unclear or off-topic, and you struggle to follow the flow of conversation, causing you to constantly reask and seek clarification."
def getAllTitles(file_path):
    # with open(file_path, "r") as f:
    #     json_lines = [json.loads(line) for line in f]
    
    # # Extract titles from all entries
    # titles = [entry.get("document_title") for entry in json_lines if "document_title" in entry]
    
    # return list(set(titles))
    with open('./data/titles.json', "r") as f:
        titles = json.load(f)  # json.load() statt json.loads() in Schleife
    return titles

def getNextDocument(title):
    with open(file_path, "r") as f:
        json_lines = [json.loads(line) for line in f]

    # Filter entries matching the given title
    texts = [entry["document_plaintext"] for entry in json_lines if entry.get("document_title") == title]

    # Combine texts into one string
    text = " ".join(texts) if texts else None
    # cut text to 1000 characters
    # if text:
    #     text = text[:5000]
    return text if text else "No document found for this title."



def send_request_to_LLM_conversation(prompt):
    success = False
    while not success:
        try:
            llm_response = model.chat_with_model(prompt)
            # print(f"LLM Response: {llm_response}")
            response = parser.parse_and_validate(llm_response)
            # print(f"Parsed Response: {response}")
            if response != "":
                success = True
        except Exception as e:
            if '429' in str(e):
                print(f"Rate limit exceeded. Waiting for 60 seconds before evaluating next batch")
                time.sleep(60)
                # try again
            elif '503' in str(e):
                print(f"Service Unavailable. Waiting for 60 seconds before evaluating next batch")
                time.sleep(60)
                # try again
            else:
                success = True
                print(f"Error generating prompt data: {e}")
                return None
    return response

def get_prompt_data(title):
    """
    Generates input for >rag_to_be_tested with LLM
    """
    prompt = templates.CONVERSATION_PROMPTS['init_prompt'].format(
        Role=Role,
        Document=getNextDocument(title)
        
    )
    # print(f"Prompt: {prompt}")
    answer = send_request_to_LLM_conversation(prompt)
    if answer is None:
        return None
    if (answer.get('rag_input') is None or answer.get('question') is None or answer.get('answer') is None):
            return None
    # print(f"Answer from prompt: {answer}")
    validation = conversation_validator.valaidate_init_prompt_all_in_one(answer, getNextDocument(title))
    if validation and not validation['correct']:
        print(f"Initial prompt validation failed: {validation['reason']}")
        # safe to log file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Validation failed for title: {title}\n")
            f.write(f"Initial Answer: {answer}\n")
            f.write(f"Reason: {validation['reason']}\n\n")
        
        answer = send_request_to_LLM_conversation(templates.CONVERSATION_PROMPTS['rephrase_init_prompt'].format(
            reason=validation['reason']
        ))
        if answer is None:
            return None
        if (answer.get('rag_input') is None or answer.get('question') is None or answer.get('answer') is None):
            return None
        # validation = conversation_validator.valaidate_init_prompt_all_in_one(answer, getNextDocument(title))
        # if validation and not validation['correct']:
        #     return None
    return answer


def get_follow_up_question(answer, title):
    """    Generates a follow-up question based on the answer provided by the rag_to_be_tested system.
    """
    follow_up_prompt = templates.CONVERSATION_PROMPTS['follow_up_prompt'].format(
        RAG_answer=answer
    )
    history = model.get_chat_history()
    response = send_request_to_LLM_conversation(follow_up_prompt)
    if response is None:
        return None
    if (response.get('rag_input') is None or response.get('type') is None or response.get('question') is None or response.get('answer') is None):
            return None
    validation = conversation_validator.validate_follow_up_question_all_in_one(response, history)
    if validation and not validation['correct']:
        print(f"Follow-up validation failed: {validation['reason']}")
         # safe to log file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Validation failed for title: {title}\n")
            f.write(f"Initial Answer: {response}\n")
            f.write(f"Reason: {validation['reason']}\n\n")
        response = send_request_to_LLM_conversation(templates.CONVERSATION_PROMPTS['rephrase_follow_up_prompt'].format(
            reason=validation['reason']
        ))
        if response is None:
            return None
        if (response.get('rag_input') is None or response.get('type') is None or response.get('question') is None or response.get('answer') is None):
            return None
        # validation = conversation_validator.validate_follow_up_question_all_in_one(response, history)
        #  # safe to log file
        # with open(log_file, 'a', encoding='utf-8') as f:
        #     f.write(f"Validation failed for title: {title}\n")
        #     f.write(f"Initial Answer: {response}\n")
        #     f.write(f"Reason: {validation['reason']}\n\n")
        # if validation and not validation['correct']:
        #     return None
        
    
    return response
        

    
# Evaluation loop
def generate_conversation():
    list_titles = getAllTitles(file_path)
    counter = 0
    # data = []
    failed_counter = 0
    while counter < len(list_titles):
        title = list_titles[counter]
        conv = []
        try:
            res = requests.get(API_URL_THREAD_ID)
            thread_id = res.json().get("thread_id", "")
        except Exception as e:
            print(f"Error getting thread ID: {e}")
            return
        print(f"Evalauting Document {counter} Evaluating title: {title}")
        question = get_prompt_data(title)
        failed = question is None
        if not failed:
            for i in range(n):
                if failed:
                    print(f"Failed to generate valid question, stopping this conversation.")
                    break
                print(f"Question {i+1}")
                try:
                    res = requests.post(API_URL, json={"question": question['rag_input'], "thread_id": thread_id})
                    answer = res.json().get("answer", "")
                    context = res.json().get("context", "")
                    # print(f"Question: {question['rag_input']}")
                    # print(f"Answer: {answer}")
                    conv.append({
                        "rag_input": question['rag_input'],
                        "question": question['question'],
                        "answer": question['answer'],
                        "type": question.get('type') or "Initial",
                        # "context": question['context'],
                        "rag_answer": answer,
                        "context": context,
                        "index": i
                    })
                except Exception as e:
                    print(question)
                    print(f"Error with question: {question} -> {e}")
                    return

                # generate follow up question, only if it is not the last loop
                if i < n-1:
                    question = get_follow_up_question(answer, title) 
                    failed = question is None 
        if not failed:
            #save data as csv
            # data.append({
            #     "title": title,
            #     "document": getNextDocument(title),
            #     "role": Role,
            #     "conversation": conv
            # })
            
            # with open(output_file, 'w', encoding='utf-8') as f:
            #     json.dump(data, f, ensure_ascii=False, indent=2)
            
            data_item = { "title": title,
                "document": getNextDocument(title),
                "role": Role,
                "conversation": conv}
            #add dataitem as jsonl
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data_item, ensure_ascii=False) + "\n")
            counter += 1
            failed_counter = 0
        model.reset_chat()
        if failed:
            failed_counter += 1
            print(f"Failed conversations: {failed_counter}")
        if failed_counter >= 3:
            print(f"Stopping evaluation after {failed_counter} failed conversations.")
            break
        if counter >= max:
            return
    
    # with open(output_file, 'w', encoding='utf-8') as f:
    #         json.dump(data, f, ensure_ascii=False, indent=2)
import gc
if __name__ == "__main__":
    # for durchlauf in range(4,10):
    #     print(f"Generating Durchlauf {durchlauf}")
    #     output_file = "./data/conversation_data_" + model_name + "_turns_" + str(n) + "_conversation_" +str(max)+ "_" +str(durchlauf) +".jsonl"
    #     log_file = "./data/conversation_data_" + model_name + "_turns_" + str(n) + "_conversation_" +str(max)+ "_" +str(durchlauf)+".log"
        # gc.collect()
        generate_conversation()
        # with open(output_file, "r") as f:
        #     lines = f.readlines()
        # data = [json.loads(line) for line in lines]
       
        # output_file2 = "./data/conversation_data_" + model_name + "_turns_" + str(n) + "_conversation_" +str(max)+ "_" +str(durchlauf) +".json"
        # with open(output_file2, 'w', encoding='utf-8') as f:
        #         json.dump(data, f, ensure_ascii=False, indent=2)
       