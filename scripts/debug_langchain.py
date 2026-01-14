
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

api_key = os.environ.get("GEMINI_API_KEY")
print(f"Testing Key: {api_key[:10]}...{api_key[-5:] if api_key else 'None'}")

if not api_key:
    print("No API Key found!")
    exit(1)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
    exit(1)


print("\n--- Testing Direct google-generativeai ---")
try:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Hi")
    print(f"DIRECT GENAI SUCCESS: {response.text}")
except Exception as e:
    print(f"DIRECT GENAI FAILED: {e}")

print("\n--- Testing LangChain ---")

