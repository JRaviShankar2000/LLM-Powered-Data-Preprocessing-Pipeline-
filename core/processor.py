import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from core.prompts import (
    normalization_prompt,
    entity_extraction_prompt,
    label_suggestion_prompt,
    feature_augmentation_prompt
)
from dotenv import load_dotenv

load_dotenv()

class TextProcessor:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            google_api_key=api_key,
            temperature=0.1
        )
        
        self.output_parser = StrOutputParser()
        
        self.chains = {
            "normalize": normalization_prompt | self.llm | self.output_parser,
            "entities": entity_extraction_prompt | self.llm | self.output_parser,
            "labels": label_suggestion_prompt | self.llm | self.output_parser,
            "augment": feature_augmentation_prompt | self.llm | self.output_parser
        }

    def process_text(self, text, tasks=None):
        if tasks is None:
            tasks = ["normalize", "entities", "labels", "augment"]
        
        results = {}
        
        if "normalize" in tasks:
            try:
                normalized_text = self.chains["normalize"].invoke({"text": text})
                results["normalized_text"] = normalized_text.strip()
            except Exception as e:
                results["normalized_text_error"] = str(e)

        if "entities" in tasks:
            try:
                entities_json = self.chains["entities"].invoke({"text": text})
                # Clean up markdown code blocks if present
                entities_json = entities_json.replace("```json", "").replace("```", "").strip()
                results["entities"] = json.loads(entities_json)
            except Exception as e:
                results["entities_error"] = str(e)
                results["entities"] = {}

        if "labels" in tasks:
            try:
                labels_json = self.chains["labels"].invoke({"text": text})
                labels_json = labels_json.replace("```json", "").replace("```", "").strip()
                results["labels"] = json.loads(labels_json)
            except Exception as e:
                results["labels_error"] = str(e)
                results["labels"] = []

        if "augment" in tasks:
            try:
                augment_json = self.chains["augment"].invoke({"text": text})
                augment_json = augment_json.replace("```json", "").replace("```", "").strip()
                results["augmentation"] = json.loads(augment_json)
            except Exception as e:
                results["augmentation_error"] = str(e)
                results["augmentation"] = {}

        return results
