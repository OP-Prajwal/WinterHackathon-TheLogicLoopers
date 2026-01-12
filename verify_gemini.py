
import os
import sys

# Try loading dotenv
try:
    from dotenv import load_dotenv, find_dotenv
    env_file = find_dotenv()
    print(f"Loading .env from: {env_file}")
    load_dotenv(env_file)
except ImportError:
    print("python-dotenv not installed.")

# Check Key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not found in environment variables.")
    # Try reading .env file manually if dotenv failed or key missing
    if os.path.exists(".env"):
        print("Checking .env file manually...")
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY"):
                    print("Found GEMINI_API_KEY in .env file (raw text check passed).")
                    # Manually set it for this invalid connection
                    parts = line.strip().split("=", 1)
                    if len(parts) == 2:
                        os.environ["GEMINI_API_KEY"] = parts[1].strip()
                        api_key = parts[1].strip()
    
    if not api_key:
        print("FATAL: Could not locate API Key.")
        sys.exit(1)
else:
    print(f"API Key found: {api_key[:5]}...{api_key[-4:]}")

# Try GenAI
try:
    import google.generativeai as genai
    print("google.generativeai imported successfully.")
    
    genai.configure(api_key=api_key)
    
    print("Attempting to list models...")
    try:
        models = list(genai.list_models())
        print(f"Found {len(models)} models.")
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                print(f" - {m.name}")
    except Exception as e:
        print(f"List models failed: {e}")

    print("\nAttempting generation with gemini-1.5-flash...")
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Hello, can you hear me?")
    print(f"Response: {response.text}")
    print("\nSUCCESS: Gemini is connected and working.")

except Exception as e:
    print(f"\nGenerative AI Error: {e}")
    print("Ensure 'google-generativeai' is installed: pip install google-generativeai")
