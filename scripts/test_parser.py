from llm_parser import get_parser

print("Testing Parser Initialization...")
parser = get_parser(force_recreate=True)

if parser.model:
    print(f"SUCCESS: Parser initialized with model.")
else:
    print("FAILURE: Parser could not find a valid model. Check API key and available models.")
