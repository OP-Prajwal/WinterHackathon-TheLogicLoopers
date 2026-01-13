"""
LLM-based NLP Parser for Health Data Extraction
Uses Google Generative AI SDK for intelligent text understanding
"""

import os
import json
import re
from typing import Dict, Any, Optional, List
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
print("[LLM Parser] Loaded .env file")

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
    """LLM-based health data parser using Google Generative AI SDK"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model = None
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            
            # List of model names to try in order of preference
            # Added gemini-2.0-flash-exp and others that might be in the list
            model_names = [
                "gemini-1.5-flash",
                "gemini-2.0-flash-exp",
                "gemini-1.5-pro",
                "gemini-pro",
                "models/gemini-1.5-flash",
                "models/gemini-1.5-pro",
                "models/gemini-pro",
                "gemini-1.0-pro"
            ]
            
            # Try to list models and find something if the defaults fail
            try:
                available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                print(f"[LLM Parser] Found {len(available)} models with generateContent support.")
                if available:
                    # Prepend discovered models to the search list
                    model_names = [name.replace('models/', '') for name in available] + model_names
            except Exception as e:
                print(f"[LLM Parser] Error listing models: {e}")

            for m_name in model_names:
                try:
                    print(f"[LLM Parser] Attempting to init model: {m_name}")
                    test_model = genai.GenerativeModel(m_name)
                    # Small test call as some models exist but won't work on the key
                    test_model.generate_content("hello", generation_config={"max_output_tokens": 5})
                    self.model = test_model
                    print(f"[LLM Parser] Successfully initialized with: {m_name}")
                    break
                except Exception as e:
                    print(f"[LLM Parser] Model {m_name} failed: {e}")
                    continue
                    
            if not self.model:
                print("[LLM Parser] Failed to initialize any LLM model. Falling back to rule-based parser.")
        else:
            print("[LLM Parser] No API key found, using rule-based parser only")

    def parse(self, text: str, features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Parse health description using LLM with rule-based fallback"""
        
        print(f"[Parser] Parsing input: {text[:100]}...", flush=True)
        
        if self.model:
            try:
                return self._parse_with_llm(text, features)
            except Exception as e:
                print(f"[Parser] LLM Runtime Error: {e}, falling back to rule-based")
                return self._parse_rule_based(text, features)
        else:
            return self._parse_rule_based(text, features)

    def _parse_with_llm(self, text: str, features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Use Google Generative AI SDK to parse text"""
        # Dynamically adjust target features if provided
        if features:
            feature_json = ", ".join([f'"{f}": <value>' for f in features])
            dynamic_prompt = EXTRACTION_PROMPT.replace(
                '"HighBP": <0/1>, "HighChol": <0/1>, "BMI": <float>, "Smoker": <0/1>,\n        "Stroke": <0/1>, "HeartDisease": <0/1>, "PhysActivity": <0/1>,\n        "HvyAlcohol": <0/1>, "DiffWalk": <0/1>, "GenHlth": <1-5>,\n        "MentHlth": <0-30>, "PhysHlth": <0-30>, "Age": <years>',
                feature_json
            )
            prompt = dynamic_prompt.format(text=text)
        else:
            prompt = EXTRACTION_PROMPT.format(text=text)
            
        response = self.model.generate_content(prompt)
        response_text = response.text.strip()
        
        print(f"[LLM Parser] Raw Response: {response_text[:200]}...", flush=True)
        
        # Handle markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        result = json.loads(response_text)
        
        # Merge with defaults or custom feature list
        if features:
            parsed_data = {f: 0 for f in features}
            if "extracted_data" in result:
                for key, value in result["extracted_data"].items():
                    if key in parsed_data:
                        parsed_data[key] = value
        else:
            parsed_data = DEFAULT_HEALTH_DATA.copy()
            if "extracted_data" in result:
                for key, value in result["extracted_data"].items():
                    if key in parsed_data:
                        parsed_data[key] = value
                    elif key == "Age" and isinstance(value, (int, float)):
                        # Convert age years to BRFSS category
                        parsed_data["Age"] = min(13, max(1, int(value) // 5 - 3))
        
        return {
            "parsed_data": parsed_data,
            "contradictions": result.get("contradictions", []),
            "anomalies": result.get("anomalies", []),
            "risk_assessment": result.get("risk_assessment", "SAFE"),
            "confidence": result.get("confidence", 0.8),
            "llm_used": True
        }

    def _parse_rule_based(self, text: str, features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Dynamic rule-based parser that maps text to features using regex and aliases"""
        
        text_lower = text.lower()
        anomalies = []
        contradictions = []
        
        # Use provided features or default set
        target_features = features if features else list(DEFAULT_HEALTH_DATA.keys())
        parsed_data = {f: (DEFAULT_HEALTH_DATA.get(f, 0.0) if not features else 0.0) for f in target_features}
        
        # Common aliases for features
        aliases = {
            "HighBP": ["blood pressure", "bp", "hypertension", "hypertensive"],
            "HighChol": ["cholesterol", "chol", "lipids"],
            "BMI": ["bmi", "body mass index", "weight", "mass"],
            "Smoker": ["smoke", "smoking", "smoker", "tobacco"],
            "Stroke": ["stroke", "brain attack"],
            "HeartDisease": ["heart disease", "heart attack", "cardiac", "coronary"],
            "PhysActivity": ["activity", "exercise", "active", "workout"],
            "Fruits": ["fruit", "fruits"],
            "Veggies": ["veggies", "vegetables", "greens"],
            "HvyAlcohol": ["alcohol", "drink", "drinking", "alcoholic"],
            "AnyHealthcare": ["healthcare", "health insurance", "covered"],
            "GenHlth": ["general health", "health status", "genhlth"],
            "MentHlth": ["mental health", "mental status", "menthlth"],
            "PhysHlth": ["physical health", "physhlth"],
            "DiffWalk": ["walk", "walking", "mobility", "diffwalk"],
            "Sex": ["sex", "gender", "male", "female"],
            "Age": ["age", "years", "yo"],
        }
        
        # 1. Extract Numeric Values
        for feature in target_features:
            search_terms = [feature] + aliases.get(feature, [])
            for term in search_terms:
                pattern = rf'{re.escape(term.lower())}\s*(?:is|of|:)?\s*(\d+\.?\d*)'
                match = re.search(pattern, text_lower)
                if match:
                    val = float(match.group(1))
                    if term == "weight" and "kg" in text_lower:
                         val = val / (1.75 * 1.75)
                    if feature == "Age" and not features:
                         parsed_data[feature] = min(13, max(1, int(val) // 5 - 3))
                    else:
                         parsed_data[feature] = val
                    break

        # 2. Extract Binary Flags
        for feature in target_features:
            if parsed_data[feature] == 0:
                search_terms = [feature] + aliases.get(feature, [])
                for term in search_terms:
                    if term.lower() in text_lower:
                        negators = ["no ", "never ", "doesn't ", "not ", "without "]
                        context = text_lower[max(0, text_lower.find(term.lower())-10):text_lower.find(term.lower())]
                        if any(neg in context for neg in negators):
                            parsed_data[feature] = 0.0
                        else:
                            parsed_data[feature] = 1.0
                        break

        # 3. Anomaly Detection
        if "BMI" in parsed_data:
            if parsed_data["BMI"] > 100:
                anomalies.append(f"BMI {parsed_data['BMI']:.1f} is physiologically impossible")
        
        # Risk Assessment
        risk = "SAFE"
        if anomalies: risk = "DANGEROUS"
        elif contradictions: risk = "SUSPICIOUS"

        return {
            "parsed_data": parsed_data,
            "contradictions": contradictions,
            "anomalies": anomalies,
            "risk_assessment": risk,
            "confidence": 0.7,
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
    """Reset the parser singleton"""
    global _parser_instance
    _parser_instance = None
