from fastapi import FastAPI
from pydantic import BaseModel
from qa_chain import get_rag_graph
from langchain_core.messages import HumanMessage

app = FastAPI()
rag_graph = get_rag_graph()

class QuestionRequest(BaseModel):
    question: str
    thread_id: str

@app.post("/rag")
def ask_question(request: QuestionRequest):
    # Retrieve the current state for the given thread_id
    # get_state returns a Checkpoint object, which has a .values attribute
    thread_id = request.thread_id
    current_checkpoint = rag_graph.get_state(config={"configurable": {"thread_id": thread_id}})
    
    # Extract existing messages from the checkpoint, default to empty list if not found
    # current_checkpoint.values will be None if no state exists for the thread_id yet
    current_messages = current_checkpoint.values.get("messages", []) if current_checkpoint and current_checkpoint.values else []

    # Append the new human message to the existing messages
    updated_messages = current_messages + [HumanMessage(content=request.question)]

    # Invoke the graph with the updated messages
    # The graph will automatically load the full state from the checkpointer based on thread_id
    # and then update it with the new messages provided in the input.
    result = rag_graph.invoke(
        {"messages": updated_messages},
        config={"configurable": {"thread_id": thread_id}}
    )
    
    # The 'result' will contain the final state after the graph execution.
    # The answer is the content of the last AIMessage in the updated messages list.
    return {"answer": result["messages"][-1].content, "context":result["context"]}

@app.get("/getThreadID")
def get_thread_id():
    """
    Returns a unique thread ID for the current conversation.
    This can be used to maintain state across multiple requests.
    """
    import uuid
    return {"thread_id": str(uuid.uuid4())}




# from fastapi import Request
# @app.post("/v1/chat/completions")
# def chat_completions(request: Request):
#     print(request)
#     answer = {
#         "id": "5be2727d-563d-42a3-8be4-65693df174e6",
#         "content": {
#             "Result": {
#                 "role": "assistant",
#                 "content": "Sandflies themselves do not cause infections that \"go away over time\" in the way that some common viral infections do. Instead, sandflies are vectors — they transmit diseases to humans and animals through bites.\n\nThe most well-known disease transmitted by sandflies is **leishmaniasis**, caused by protozoan parasites of the genus *Leishmania*. When a sandfly bites a person, it injects *Leishmania* parasites into the skin. These parasites then multiply in the body and can lead to a variety of clinical outcomes, including:\n\n- **Cutaneous leishmaniasis** (skin sores or ulcers)\n- **Mucocutaneous leishmaniasis** (involving mucous membranes)\n- **Visceral leishmaniasis** (a more severe, potentially life-threatening form affecting internal organs)\n\nThese infections **do not typically resolve on their own** without treatment. While some mild cases of cutaneous leishmaniasis may heal spontaneously over months or years, this is not guaranteed and often depends on the individual's immune response. In contrast, visceral leishmaniasis requires medical treatment with antiparasitic drugs.\n\nSo, to answer your question:  \n👉 **No, an infection from a sandfly bite (like leishmaniasis) generally does not go away on its own and usually requires treatment.** It is not a self-limiting infection in most cases.\n\nIf you suspect you've been bitten by a sandfly or have symptoms, it's important to consult a healthcare provider for proper diagnosis and care.",
#                 "rag_entries": [
#                     {
#                         "score": 9.713783264160156,
#                         "text": "# Infection\n\nPersistent infections occur because the body is unable to clear the organism after the initial infection. Persistent infections are characterized by the continual presence of the infectious organism, often as latent infection with occasional recurrent relapses of active infection. There are some viruses that can maintain a persistent infection by infecting different cells of the body. Some viruses once acquired never leave the body. A typical example is the herpes virus, which tends to hide in nerves and become reactivated when specific circumstances arise.\nResistance to infection (immunity) may be acquired following a disease, by asymptomatic carriage of the pathogen, by harboring an organism with a similar structure (crossreacting), or by vaccination. Knowledge of the protective antigens and specific acquired host immune factors is more complete for primary pathogens than for opportunistic pathogens. There is also the phenomenon of herd immunity which offers a measure of protection to those otherwise vulnerable people when a large enough proportion of the population has acquired immunity from certain infections.",
#                         "source": "./data/evaluation/corpus-single/Infection.md"
#                     },
#                     {
#                         "score": 0.8028257745385342,
#                         "text": "# Infection\n\nInfection begins when an organism successfully enters the body, grows and multiplies. This is referred to as colonization. Most humans are not easily infected. Those who are weak, sick, malnourished, have cancer or are diabetic have increased susceptibility to chronic or persistent infections. Individuals who have a suppressed immune system are particularly susceptible to opportunistic infections. Entrance to the host at host-pathogen interface, generally occurs through the mucosa in orifices like the oral cavity, nose, eyes, genitalia, anus, or the microbe can enter through open wounds. While a few organisms can grow at the initial site of entry, many migrate and cause systemic infection in different organs. Some pathogens grow within the host cells (intracellular) whereas others grow freely in bodily fluids.",
#                         "source": "./data/evaluation/corpus-single/Infection.md"
#                     },
#                     {
#                         "score": 9.572257041931152,
#                         "text": "# Antibiotics\n\nThe effects of some types of mold on infection had been noticed many times over the course of history (see: History of penicillin). In 1928, Alexander Fleming noticed the same effect in a Petri dish, where a number of disease-causing bacteria were killed by a fungus of the genus Penicillium. Fleming postulated that the effect is mediated by an antibacterial compound he named penicillin, and that its antibacterial properties could be exploited for chemotherapy. He initially characterized some of its biological properties, and attempted to use a crude preparation to treat some infections, but he was unable to pursue its further development without the aid of trained chemists.",
#                         "source": "./data/evaluation/corpus-single/Antibiotics.md"
#                     }
#                 ]
#             }
#         }
#     }
#     return answer