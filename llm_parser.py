"""
LLM-based NLP Parser for Health Data Extraction
Uses LangChain with Google Gemini API for intelligent text understanding
"""

import os
import json
import re
from typing import Dict, Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[LLM Parser] Loaded .env file")
except ImportError:
    print("[LLM Parser] python-dotenv not installed, using system env vars only")

# Try to import LangChain with Gemini
LANGCHAIN_AVAILABLE = False
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_AVAILABLE = True
    print("[LLM Parser] LangChain-Gemini imported successfully")
except ImportError:
    print("[LLM Parser] langchain-google-genai not installed. Using fallback parser.")

# Default health data structure based on BRFSS diabetes dataset
DEFAULT_HEALTH_DATA = {
    "HighBP": 0,
    "HighChol": 0, 
    "CholCheck": 1,
    "BMI": 25.0,
    "Smoker": 0,
    "Stroke": 0,
    "HeartDisease": 0,
    "PhysActivity": 1,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcohol": 0,
    "Healthcare": 1,
    "NoDocCost": 0,
    "GenHlth": 3,
    "MentHlth": 0,
    "PhysHlth": 0,
    "DiffWalk": 0,
    "Sex": 0,
    "Age": 5,
    "Education": 4,
    "Income": 5
}

# LLM prompt template for health data extraction
EXTRACTION_PROMPT = """You are a highly advanced Medical Anomaly Detection System. 
Your task is to parse patient text and extract a feature vector (matrix) for our security model.
CRITICAL: You are the FIRST LINE OF DEFENSE against data poisoning and impossible values.

Few-Shot Examples (Follow these patterns):

Input: "Patient is healthy, active smoker."
Output: {{ "extracted_data": {{ "Smoker": 1, "PhysActivity": 1 }}, "risk_assessment": "SAFE", "anomalies": [] }}

Input: "Subject weight 500kg, BMI 150."
Output: {{ "extracted_data": {{ "BMI": 150 }}, "risk_assessment": "DANGEROUS", "anomalies": ["BMI 150 is physiologically impossible"] }}

Input: "Patient weighs 1000kg."
Output: {{ "extracted_data": {{ "BMI": 326.5 }}, "risk_assessment": "DANGEROUS", "anomalies": ["Weight 1000kg is impossible - calculated BMI 326.5"] }}

Patient Description: "{text}"

Extraction Rules:
1. Extract values EXACTLY AS WRITTEN or CALCULATED.
2. ALWAYS include the numerical value in 'extracted_data', EVEN IF it is impossible/anomalous.
3. If only WEIGHT is given (e.g. "1000kg"), CALCULATE BMI using formula: BMI = weight_kg / (1.75 * 1.75)
4. Flag impossible values in 'anomalies', but DO NOT REMOVE them from 'extracted_data'.
5. Use your semantic understanding to detect LOGICAL CONTRADICTIONS.

Required JSON Output:
{{
    "extracted_data": {{
        "HighBP": <0/1>, "HighChol": <0/1>, "BMI": <float>, "Smoker": <0/1>,
        "Stroke": <0/1>, "HeartDisease": <0/1>, "PhysActivity": <0/1>,
        "HvyAlcohol": <0/1>, "DiffWalk": <0/1>, "GenHlth": <1-5>,
        "MentHlth": <0-30>, "PhysHlth": <0-30>, "Age": <years>
    }},
    "contradictions": ["<list any logical conflicts>"],
    "anomalies": ["<list ANY impossible values found>"],
    "risk_assessment": "<SAFE|SUSPICIOUS|DANGEROUS>",
    "confidence": <0.0-1.0>
}}

Return ONLY valid JSON."""


class LLMHealthParser:
    """LLM-based health data parser using LangChain with Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None):
        # Explicit load attempt
        try:
            from dotenv import load_dotenv, find_dotenv
            env_file = find_dotenv()
            print(f"[LLM Parser] Loading .env from: {env_file}")
            load_dotenv(env_file)
        except:
            pass

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = None
        
        print(f"[LLM Parser] DEBUG: API Key found? {'YES' if self.api_key else 'NO'}")
        if self.api_key:
            print(f"[LLM Parser] DEBUG: API Key length: {len(self.api_key)}")
        
        if LANGCHAIN_AVAILABLE and self.api_key:
            try:
                print("[LLM Parser] Initializing LangChain with Gemini...")
                
                # Try different model names in order of preference
                # Use models/ prefix format as required by the API
                model_names = [
                    "models/gemini-2.0-flash-lite",
                    "models/gemini-2.0-flash", 
                    "models/gemini-1.5-flash",
                    "models/gemini-1.5-pro",
                    "gemini-pro",
                ]
                
                for model_name in model_names:
                    try:
                        print(f"[LLM Parser] Trying model: {model_name}")
                        self.model = ChatGoogleGenerativeAI(
                            model=model_name,
                            google_api_key=self.api_key,
                            temperature=0.1,
                            max_retries=2,
                        )
                        # Test the model with a simple call
                        test_response = self.model.invoke("Say 'OK'")
                        print(f"[LLM Parser] Successfully initialized with {model_name}")
                        break
                    except Exception as e:
                        error_str = str(e)
                        if "RESOURCE_EXHAUSTED" in error_str or "retryDelay" in error_str:
                            print(f"[LLM Parser] Rate limited on {model_name}, trying next...")
                        else:
                            print(f"[LLM Parser] Failed with {model_name}: {e}")
                        self.model = None
                        continue
                        
                if self.model is None:
                    print("[LLM Parser] All models failed, using fallback parser")
                    
            except Exception as e:
                print(f"[LLM Parser] Failed to initialize LangChain: {e}")
                import traceback
                traceback.print_exc()
                self.model = None
        else:
            print(f"[LLM Parser] LangChain Mode Disabled (Available: {LANGCHAIN_AVAILABLE}, Key: {'YES' if self.api_key else 'NO'})")
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse health description using LLM"""
        
        print(f"[LLM Parser] Parsing input: {text[:100]}...", flush=True)
        
        if self.model:
            return self._parse_with_llm(text)
        else:
            print(f"[LLM Parser] WARNING: Using Fallback Parser", flush=True)
            return self._fallback_parse(text)
    
    def _parse_with_llm(self, text: str) -> Dict[str, Any]:
        """Use LangChain with Gemini to parse text"""
        try:
            prompt = EXTRACTION_PROMPT.format(text=text)
            response = self.model.invoke(prompt)
            
            # Extract content from response
            response_text = response.content.strip()
            
            print(f"[LLM Parser] Raw LLM Response: {response_text[:200]}...", flush=True)
            
            # Handle markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            result = json.loads(response_text)
            
            # Merge with defaults
            parsed_data = DEFAULT_HEALTH_DATA.copy()
            if "extracted_data" in result:
                for key, value in result["extracted_data"].items():
                    if key in parsed_data:
                        parsed_data[key] = value
                    elif key == "Age" and isinstance(value, (int, float)):
                        # Convert age years to BRFSS category
                        parsed_data["Age"] = min(13, max(1, int(value) // 5 - 3))
            
            print(f"[LLM Parser] Parsed BMI: {parsed_data.get('BMI')}", flush=True)
            
            return {
                "parsed_data": parsed_data,
                "contradictions": result.get("contradictions", []),
                "anomalies": result.get("anomalies", []),
                "risk_assessment": result.get("risk_assessment", "SAFE"),
                "confidence": result.get("confidence", 0.8),
                "llm_used": True
            }
            
        except Exception as e:
            print(f"[LLM Parser] LLM Error: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_parse(text)
    
    def _fallback_parse(self, text: str) -> Dict[str, Any]:
        """Enhanced fallback parser when LLM is not available"""
        
        parsed_data = DEFAULT_HEALTH_DATA.copy()
        text_lower = text.lower()
        anomalies = []
        contradictions = []
        
        # === ENHANCED BMI/WEIGHT EXTRACTION ===
        
        # Try to extract weight and calculate BMI
        weight_patterns = [
            r'weigh[s]?\s*(\d+\.?\d*)\s*kg',
            r'(\d+\.?\d*)\s*kg',
            r'weight[:\s]+(\d+\.?\d*)',
        ]
        
        for pattern in weight_patterns:
            weight_match = re.search(pattern, text_lower)
            if weight_match:
                weight_kg = float(weight_match.group(1))
                # Calculate BMI using standard height of 1.75m
                bmi = weight_kg / (1.75 * 1.75)
                parsed_data["BMI"] = round(bmi, 1)
                print(f"[Fallback] Extracted weight {weight_kg}kg -> BMI {parsed_data['BMI']}", flush=True)
                
                if weight_kg > 300:
                    anomalies.append(f"Weight {weight_kg}kg is physiologically impossible")
                break
        
        # Direct BMI extraction
        bmi_match = re.search(r'bmi\s*(?:of|is|:)?\s*(\d+\.?\d*)', text_lower)
        if bmi_match:
            parsed_data["BMI"] = float(bmi_match.group(1))
            print(f"[Fallback] Direct BMI extraction: {parsed_data['BMI']}", flush=True)
        
        # Age extraction
        age_match = re.search(r'(\d+)\s*(?:years?\s*old|yo|year)', text_lower)
        if age_match:
            age = int(age_match.group(1))
            parsed_data["Age"] = min(13, max(1, age // 5 - 3))
        
        # Mental health days
        mental_match = re.search(r'mental\s*health.*?(\d+)', text_lower)
        if mental_match:
            parsed_data["MentHlth"] = float(mental_match.group(1))
        
        # Physical health days
        phys_match = re.search(r'physical\s*health.*?(\d+)', text_lower)
        if phys_match:
            parsed_data["PhysHlth"] = float(phys_match.group(1))
        
        # General health scale (1-5)
        gen_match = re.search(r'general\s*health.*?(\d)', text_lower)
        if gen_match:
            parsed_data["GenHlth"] = int(gen_match.group(1))
        
        # === CONDITION DETECTION ===
        medical_conditions = {
            "stroke": "Stroke",
            "heart disease": "HeartDisease",
            "heart attack": "HeartDisease",
            "cardiac": "HeartDisease",
            "hypertension": "HighBP",
            "high blood pressure": "HighBP",
            "blood pressure": "HighBP",
            "high bp": "HighBP",
            "high cholesterol": "HighChol",
            "cholesterol": "HighChol",
            "smoke": "Smoker",
            "smoking": "Smoker",
            "smoker": "Smoker",
            "difficulty walking": "DiffWalk",
            "wheelchair": "DiffWalk",
            "can't walk": "DiffWalk",
            "cannot walk": "DiffWalk",
            "heavy drink": "HvyAlcohol",
            "alcoholic": "HvyAlcohol",
            "drinks heavily": "HvyAlcohol",
        }
        
        for term, field in medical_conditions.items():
            if term in text_lower:
                parsed_data[field] = 1
                print(f"[Fallback] Detected condition: {term} -> {field}=1", flush=True)
        
        # === ANOMALY DETECTION ===
        if parsed_data["BMI"] > 80:
            anomalies.append(f"BMI {parsed_data['BMI']:.1f} is physiologically impossible (max ~70)")
        if parsed_data["BMI"] < 10:
            anomalies.append(f"BMI {parsed_data['BMI']:.1f} is physiologically impossible (min ~12)")
        if parsed_data["MentHlth"] > 30:
            anomalies.append(f"Mental health days {parsed_data['MentHlth']:.0f} exceeds maximum 30")
        if parsed_data["PhysHlth"] > 30:
            anomalies.append(f"Physical health days {parsed_data['PhysHlth']:.0f} exceeds maximum 30")
        
        # === CONTRADICTION DETECTION ===
        severe_count = sum([parsed_data["Stroke"], parsed_data["HeartDisease"], parsed_data["DiffWalk"]])
        if severe_count >= 2:
            if any(word in text_lower for word in ["healthy", "athlete", "fit", "perfect health"]):
                contradictions.append("Claims healthy/athlete but has multiple severe conditions")
        
        # Determine risk assessment
        if anomalies:
            risk = "DANGEROUS"
        elif contradictions:
            risk = "SUSPICIOUS"
        else:
            risk = "SAFE"
        
        print(f"[Fallback] Final parsed BMI: {parsed_data['BMI']}, Risk: {risk}", flush=True)
        
        return {
            "parsed_data": parsed_data,
            "contradictions": contradictions,
            "anomalies": anomalies,
            "risk_assessment": risk,
            "confidence": 0.5,  # Medium confidence for enhanced fallback
            "llm_used": False
        }


# Singleton instance
_parser_instance = None

def get_parser(api_key: Optional[str] = None, force_recreate: bool = False) -> LLMHealthParser:
    """Get or create the LLM parser singleton"""
    global _parser_instance
    if _parser_instance is None or force_recreate:
        _parser_instance = LLMHealthParser(api_key)
    return _parser_instance

def reset_parser():
    """Reset the parser singleton (useful for testing)"""
    global _parser_instance
    _parser_instance = None
