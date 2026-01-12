
from llm_parser import get_parser
import json

text = "I am a super fit marathon runner who has difficulty walking and uses a wheelchair"

print(f"Testing input: '{text}'")

try:
    parser = get_parser()
    result = parser.parse(text)
    
    print("\n--- LLM RESULT ---")
    print(json.dumps(result, indent=2))
    
    if result.get("llm_used"):
        print("\n✅ GEMINI WAS USED")
    else:
        print("\n❌ FALLBACK PARSER USED (Gemini failed/not configured)")
        
except Exception as e:
    print(f"\nERROR: {e}")
