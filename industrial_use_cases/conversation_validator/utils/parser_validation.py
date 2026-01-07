"""
Parser for LLM responses in rag_to_be_tested evaluation
"""
import re
from typing import Dict, Optional

class LLMResponseParser:

    def parse_and_validate_validation(self, response: str) -> Optional[Dict[str, str]]:
        try:
            import json, re
            if response.strip().startswith("```json"):
                response = re.sub(r"^```json\s*", "", response.strip())
                response = re.sub(r"\s*```$", "", response.strip())

            json_data = json.loads(response)
            if isinstance(json_data, dict):
                normalized = {}
                for k, v in json_data.items():
                    key = k.lower().replace(" ", "_")
                    if isinstance(v, str):
                        normalized[key] = v.strip()
                    elif isinstance(v, bool):
                        normalized[key] = v
                    else:
                        normalized[key] = str(v)

                required_fields = ['correct', 'reason']
                # Check for presence, not truthiness
                if all(field in normalized for field in required_fields):
                    return {field: normalized[field] for field in required_fields}

        except json.JSONDecodeError:
            pass

        print("Warning: Incomplete validation response received")
        print("Response text:", response)
        try:
            missing = [field for field in ['correct', 'reason'] if field not in normalized]
            print(f"Missing fields: {missing}")
        except NameError:
            pass

        return None

# Usage example
if __name__ == "__main__":
    # Test the parser
    sample_response = """
    **rag_to_be_tested input:**
    What is CARE and how does it evaluate rag_to_be_tested systems?

    **Question:**
    Can you explain the Context-Aware Retriever Evaluation method mentioned in the document?

    **Answer:**
    CARE (Context-Aware Retriever Evaluation) is a novel method for evaluating the retriever component of rag_to_be_tested systems in multi-hop queries.

    **Context:**
    In this research, we propose Context-Aware Retriever Evaluation (CARE), a novel method for evaluating the retriever component of rag_to_be_tested systems in multi-hop queries.
    """
    
    parser = LLMResponseParser()
    parsed = parser.parse_and_validate(sample_response)
    
    if parsed:
        for key, value in parsed.items():
            print(f"{key}: {value}\n")