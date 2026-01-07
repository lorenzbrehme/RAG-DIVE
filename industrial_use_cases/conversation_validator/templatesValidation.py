"""
Prompt templates for rag_to_be_tested evaluation
"""



VALIDATION_PROMPTS = {     'validate_init_prompt': '''
            Validate the following initial prompt.  
            It contains a `RAG_input` used to test a Retrieval-Augmented Generation (rag_to_be_tested) system.  
            This should represent the start of a conversation with the rag_to_be_tested system.  

            Requirements:  
            1. The context of the RAG_input should be related to the provided document.  
            2. The question must be exactly the same as the `RAG_input`.  
            3. The answer must be correct.  

            Input to validate:  
            {question}  
            Provided Document: {document}

            If any of the above requirements are not met, return the reason and set `"correct"` to `false`.  
            If all requirements are met, set `"correct"` to `true`.  

            Return the result as JSON in the following format:
         
                "correct": true/false,
                "reason": "Explanation if incorrect, otherwise empty"
           
        ''',
    
    'validate_follow_up_prompt': '''
             Validate the input against the rules below.
              1. RAG_input

                
                Must be clear, specific, and contextually connected.
                It must clearly build on the prior turn (not standalone)
                Never mention that a document was provided.

              2.  Question

                Must be a standalone rephrasing of the RAG_input (self-contained, understandable without history).
                Must include necessary context but not refer to “a document.”
                Must not be identical to RAG_input.

            3. Type

               Must be one of: Follow-up, Clarification, Correction, Comparative.
                Must match its declared type:
                Follow-up: continues or builds on the previous answer.
                Clarification: asks for clarification about the previous answer.
                Correction: points out or requests a correction to the previous answer.
                Comparative: asks to compare or contrast items from earlier in the conversation.


            4. Answer

                Must be factually correct and relevant to the RAG_input / standalone Question.
            Input: {question}
            History: {conversation_history}

                **Output Format (exact JSON):** 
                "correct": true/false, 
                "reason": "Explanation of which requirement failed and why (leave empty if correct)"
                    
        
        '''
}