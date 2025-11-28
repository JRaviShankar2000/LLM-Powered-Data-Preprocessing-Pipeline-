from langchain_core.prompts import PromptTemplate

# Normalization Prompt
normalization_template = """
You are an expert text preprocessor. Your task is to normalize the following text.
Specific instructions:
1. Fix grammatical errors.
2. Remove unnecessary whitespace.
3. Expand contractions (e.g., "don't" -> "do not").
4. Convert to standard English if it's slang.
5. Do NOT change the meaning of the text.

Text: {text}

Normalized Text:
"""

normalization_prompt = PromptTemplate(
    input_variables=["text"],
    template=normalization_template
)

# Entity Extraction Prompt
entity_extraction_template = """
You are an expert Named Entity Recognition (NER) system.
Extract the following entities from the text if present:
- PERSON
- ORGANIZATION
- LOCATION
- DATE
- PRODUCT

Return the result strictly as a JSON object where keys are the entity types and values are lists of extracted entities.
If no entities of a type are found, return an empty list for that key.
Do not include any markdown formatting or explanations, just the raw JSON string.

Text: {text}

JSON Output:
"""

entity_extraction_prompt = PromptTemplate(
    input_variables=["text"],
    template=entity_extraction_template
)

# Label Suggestion Prompt
label_suggestion_template = """
Analyze the following text and suggest up to 3 relevant category labels or tags.
Return the result strictly as a JSON list of strings.
Example: ["Technology", "AI", "Machine Learning"]

Text: {text}

JSON Output:
"""

label_suggestion_prompt = PromptTemplate(
    input_variables=["text"],
    template=label_suggestion_template
)

# Feature Augmentation Prompt
feature_augmentation_template = """
Analyze the text and generate a brief summary (max 1 sentence) and a sentiment score (Positive, Negative, Neutral).
Return the result strictly as a JSON object with keys "summary" and "sentiment".

Text: {text}

JSON Output:
"""

feature_augmentation_prompt = PromptTemplate(
    input_variables=["text"],
    template=feature_augmentation_template
)
