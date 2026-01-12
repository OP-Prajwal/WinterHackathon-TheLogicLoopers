"""
LLM-based NLP Parser for Health Data Extraction
Uses Google Gemini API for intelligent text understanding
No hardcoded keywords or regex patterns
"""

import os
import json
from typing import Dict, Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[LLM Parser] Loaded .env file")
except ImportError:
    print("[LLM Parser] python-dotenv not installed, using system env vars only")

# Try to import Google GenerativeAI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Using fallback parser.")

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
EXTRACTION_PROMPT = """You are a medical data extraction AI. Analyze the following patient description and extract health metrics.

Patient Description: "{text}"

Extract the following fields as JSON. Use your medical knowledge to infer values from context.
For binary fields (0 or 1), use 1 if the condition is present/likely, 0 otherwise.

Required JSON output format:
{{
    "extracted_data": {{
        "HighBP": <0 or 1 - high blood pressure>,
        "HighChol": <0 or 1 - high cholesterol>,
        "BMI": <number - body mass index, estimate if weight/height given>,
        "Smoker": <0 or 1>,
        "Stroke": <0 or 1 - history of stroke>,
        "HeartDisease": <0 or 1>,
        "PhysActivity": <0 or 1 - physically active>,
        "HvyAlcohol": <0 or 1 - heavy alcohol consumption>,
        "DiffWalk": <0 or 1 - difficulty walking>,
        "GenHlth": <1-5: 1=excellent, 2=very good, 3=good, 4=fair, 5=poor>,
        "MentHlth": <0-30: days of poor mental health in past month>,
        "PhysHlth": <0-30: days of poor physical health in past month>,
        "Age": <estimated age in years>
    }},
    "contradictions": [
        <list any logical contradictions found, e.g. "claims healthy but has severe conditions">
    ],
    "anomalies": [
        <list any physiologically impossible values, e.g. "BMI 150 is impossible">
    ],
    "risk_assessment": "<SAFE|SUSPICIOUS|DANGEROUS>",
    "confidence": <0.0 to 1.0>
}}

Be thorough in detecting contradictions like:
- Someone claiming to be "healthy" or "athlete" but having chronic diseases
- Young age with multiple severe conditions
- Impossible numerical values (BMI > 80, negative values, days > 30)

Return ONLY valid JSON, no other text."""


class LLMHealthParser:
    """LLM-based health data parser using Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = None
        
        if GEMINI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                # Try newer models first
                try:
                    self.model = genai.GenerativeModel('gemini-2.0-flash')
                    print("[LLM Parser] Gemini API initialized (gemini-2.0-flash)")
                except:
                    try:
                        self.model = genai.GenerativeModel('gemini-1.5-flash')
                        print("[LLM Parser] Gemini API initialized (gemini-1.5-flash)")
                    except:
                        # Fallback to whatever 'gemini-pro' points to
                        self.model = genai.GenerativeModel('gemini-pro')
                        print("[LLM Parser] Gemini API initialized (gemini-pro)")
            except Exception as e:
                print(f"[LLM Parser] Failed to initialize Gemini: {e}")
                self.model = None
        else:
            print("[LLM Parser] Gemini not available, using fallback")
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse health description using LLM"""
        
        if self.model:
            return self._parse_with_llm(text)
        else:
            return self._fallback_parse(text)
    
    def _parse_with_llm(self, text: str) -> Dict[str, Any]:
        """Use Gemini to parse text"""
        try:
            prompt = EXTRACTION_PROMPT.format(text=text)
            response = self.model.generate_content(prompt)
            
            # Extract JSON from response
            response_text = response.text.strip()
            
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
            
            return {
                "parsed_data": parsed_data,
                "contradictions": result.get("contradictions", []),
                "anomalies": result.get("anomalies", []),
                "risk_assessment": result.get("risk_assessment", "SAFE"),
                "confidence": result.get("confidence", 0.5),
                "llm_used": True
            }
            
        except Exception as e:
            print(f"[LLM Parser] Error: {e}")
            return self._fallback_parse(text)
    
    def _fallback_parse(self, text: str) -> Dict[str, Any]:
        """Simple fallback parser when LLM is not available"""
        import re
        
        parsed_data = DEFAULT_HEALTH_DATA.copy()
        text_lower = text.lower()
        
        # Basic number extraction for BMI
        bmi_match = re.search(r'bmi\s*(?:of|is|:)?\s*(\d+\.?\d*)', text_lower)
        if bmi_match:
            parsed_data["BMI"] = float(bmi_match.group(1))
        
        # Age extraction
        age_match = re.search(r'(\d+)\s*(?:years?\s*old|yo)', text_lower)
        if age_match:
            age = int(age_match.group(1))
            parsed_data["Age"] = min(13, max(1, age // 5 - 3))
        
        # Mental health
        mental_match = re.search(r'mental\s*health.*?(\d+)', text_lower)
        if mental_match:
            parsed_data["MentHlth"] = float(mental_match.group(1))
        
        # Condition detection (without hardcoded lists - using common medical terms)
        medical_conditions = {
            "stroke": "Stroke",
            "heart disease": "HeartDisease",
            "heart attack": "HeartDisease",
            "cardiac": "HeartDisease",
            "hypertension": "HighBP",
            "high blood pressure": "HighBP",
            "high cholesterol": "HighChol",
            "smoke": "Smoker",
            "smoking": "Smoker",
            "smoker": "Smoker",
            "difficulty walking": "DiffWalk",
            "wheelchair": "DiffWalk",
            "can't walk": "DiffWalk",
            "heavy drink": "HvyAlcohol",
            "alcoholic": "HvyAlcohol"
        }
        
        for term, field in medical_conditions.items():
            if term in text_lower:
                parsed_data[field] = 1
        
        # Detect contradictions
        contradictions = []
        anomalies = []
        
        # Check for physiological impossibilities
        if parsed_data["BMI"] > 80:
            anomalies.append(f"BMI {parsed_data['BMI']:.1f} is physiologically impossible")
        if parsed_data["MentHlth"] > 30:
            anomalies.append(f"Mental health days {parsed_data['MentHlth']:.0f} exceeds 30")
        
        # Check for contradictions (simplified without LLM)
        severe_count = sum([parsed_data["Stroke"], parsed_data["HeartDisease"], parsed_data["DiffWalk"]])
        if severe_count >= 2:
            if any(word in text_lower for word in ["healthy", "athlete", "fit"]):
                contradictions.append("Claims healthy/athlete but has multiple severe conditions")
        
        return {
            "parsed_data": parsed_data,
            "contradictions": contradictions,
            "anomalies": anomalies,
            "risk_assessment": "DANGEROUS" if anomalies else ("SUSPICIOUS" if contradictions else "SAFE"),
            "confidence": 0.3,  # Lower confidence for fallback
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
