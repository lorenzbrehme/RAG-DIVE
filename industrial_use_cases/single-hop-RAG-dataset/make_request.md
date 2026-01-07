```python
import os, json, requests, time, subprocess
from dotenv import load_dotenv
load_dotenv()
# Set your dataset folder here (e.g. 'single-hop-RAG-dataset', 'multi-hop-RAG-dataset', 'lang-RAG-dataset')
dataset_folder = 'casablanca-use-case'  # Change as needed
input_json = ('dataset.json')  # Change as needed
output_json = 'casablanca_qwen4B_gemma_bge_old.json'  # data_type, generator, embedding, reranker
log_filename = output_json.replace('.json', '_log.csv')
input_json_path = os.path.join(dataset_folder, input_json)
output_json_path = os.path.join(dataset_folder, output_json)
answer_file_path = os.path.abspath('agent_files/answer.json')
answer_archive_path = os.path.join(dataset_folder, 'answer_generated_all.json')
logging_path = os.path.join(dataset_folder, 'logging.json')
NUM_SAMPLES = 100  # Change as needed
RAG_API_URL = os.environ.get('RAG_API_URL', 'http://127.0.0.1:10000/v1/chat/completions')
RAG_API_METHOD = os.environ.get('RAG_API_METHOD', 'GET').upper()


def call_rest_api(question):

    # Archive and remove any existing local agent answer file
    if os.path.exists(answer_file_path):
        with open(answer_file_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        with open(answer_archive_path, 'a', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
            f.write('\n')
        os.remove(answer_file_path)
    headers = {'Content-Type': 'application/json'}
    payload = {'messages': [{'role': 'user', 'content': question}]}
    start_time = time.time()
    try:
        response = requests.request(
            method=RAG_API_METHOD,
            url=RAG_API_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=60
        )
        response.raise_for_status()
        response_json = response.json()
    except Exception as e:
        print(f'Request failed: {e}')
        return {'response': '', 'retrieved_contexts': [], 'response_time': None}
    end_time = time.time()
    response_time = end_time - start_time
    try:
        with open(logging_path, 'a', encoding='utf-8') as log_file:
            json.dump(response_json, log_file, indent=2, ensure_ascii=False)
            log_file.write('\n')
    except Exception:
        pass
    # Updated parsing for nested 'Result' key
    message = response_json.get('content', {}).get('Result', {})
    answer = message.get('content', '') if isinstance(message, dict) else ''
    rag_entries = message.get('rag_entries', []) if isinstance(message, dict) else []
    retrieved_contexts = [entry.get('text', '') for entry in rag_entries if isinstance(entry, dict)]
    print(f'Retrieved {len(retrieved_contexts)} chunks for question\n')
    return {'response': answer, 'retrieved_contexts': retrieved_contexts, 'response_time': response_time}
```

Input

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<<< the question string you pass to call_rest_api(question) >>>"
    }
  ]
}
```

Antwort

````json
{
    "id": "5be2727d-563d-42a3-8be4-65693df174e6",
    "content": {
        "Result": {
            "role": "assistant",
            "content": "Sandflies themselves do not cause infections that \"go away over time\" in the way that some common viral infections do. Instead, sandflies are vectors — they transmit diseases to humans and animals through bites.\n\nThe most well-known disease transmitted by sandflies is **leishmaniasis**, caused by protozoan parasites of the genus *Leishmania*. When a sandfly bites a person, it injects *Leishmania* parasites into the skin. These parasites then multiply in the body and can lead to a variety of clinical outcomes, including:\n\n- **Cutaneous leishmaniasis** (skin sores or ulcers)\n- **Mucocutaneous leishmaniasis** (involving mucous membranes)\n- **Visceral leishmaniasis** (a more severe, potentially life-threatening form affecting internal organs)\n\nThese infections **do not typically resolve on their own** without treatment. While some mild cases of cutaneous leishmaniasis may heal spontaneously over months or years, this is not guaranteed and often depends on the individual's immune response. In contrast, visceral leishmaniasis requires medical treatment with antiparasitic drugs.\n\nSo, to answer your question:  \n👉 **No, an infection from a sandfly bite (like leishmaniasis) generally does not go away on its own and usually requires treatment.** It is not a self-limiting infection in most cases.\n\nIf you suspect you've been bitten by a sandfly or have symptoms, it's important to consult a healthcare provider for proper diagnosis and care.",
            "rag_entries": [
                {
                    "score": 9.713783264160156,
                    "text": "# Infection\n\nPersistent infections occur because the body is unable to clear the organism after the initial infection. Persistent infections are characterized by the continual presence of the infectious organism, often as latent infection with occasional recurrent relapses of active infection. There are some viruses that can maintain a persistent infection by infecting different cells of the body. Some viruses once acquired never leave the body. A typical example is the herpes virus, which tends to hide in nerves and become reactivated when specific circumstances arise.\nResistance to infection (immunity) may be acquired following a disease, by asymptomatic carriage of the pathogen, by harboring an organism with a similar structure (crossreacting), or by vaccination. Knowledge of the protective antigens and specific acquired host immune factors is more complete for primary pathogens than for opportunistic pathogens. There is also the phenomenon of herd immunity which offers a measure of protection to those otherwise vulnerable people when a large enough proportion of the population has acquired immunity from certain infections.",
                    "source": "./data/evaluation/corpus-single/Infection.md"
                },
                {
                    "score": 0.8028257745385342,
                    "text": "# Infection\n\nInfection begins when an organism successfully enters the body, grows and multiplies. This is referred to as colonization. Most humans are not easily infected. Those who are weak, sick, malnourished, have cancer or are diabetic have increased susceptibility to chronic or persistent infections. Individuals who have a suppressed immune system are particularly susceptible to opportunistic infections. Entrance to the host at host-pathogen interface, generally occurs through the mucosa in orifices like the oral cavity, nose, eyes, genitalia, anus, or the microbe can enter through open wounds. While a few organisms can grow at the initial site of entry, many migrate and cause systemic infection in different organs. Some pathogens grow within the host cells (intracellular) whereas others grow freely in bodily fluids.",
                    "source": "./data/evaluation/corpus-single/Infection.md"
                },
                {
                    "score": 9.572257041931152,
                    "text": "# Antibiotics\n\nThe effects of some types of mold on infection had been noticed many times over the course of history (see: History of penicillin). In 1928, Alexander Fleming noticed the same effect in a Petri dish, where a number of disease-causing bacteria were killed by a fungus of the genus Penicillium. Fleming postulated that the effect is mediated by an antibacterial compound he named penicillin, and that its antibacterial properties could be exploited for chemotherapy. He initially characterized some of its biological properties, and attempted to use a crude preparation to treat some infections, but he was unable to pursue its further development without the aid of trained chemists.",
                    "source": "./data/evaluation/corpus-single/Antibiotics.md"
                }
            ]
        }
    }
}```


````
