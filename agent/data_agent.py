import os
import pandas as pd
import io
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

class DataAgent:
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

    def analyze_metadata(self, df: pd.DataFrame) -> str:
        """Extracts metadata from the dataframe for the LLM."""
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        
        description = df.describe(include='all').to_string()
        head = df.head().to_string()
        
        metadata = f"""
        Dataframe Info:
        {info_str}
        
        First 5 rows:
        {head}
        
        Statistical Description:
        {description}
        """
        return metadata

    def generate_strategy(self, metadata: str, user_intent: str) -> str:
        """Generates a preprocessing strategy based on metadata and user intent."""
        template = """
        You are a Senior Data Scientist.
        
        **Dataset Metadata:**
        {metadata}
        
        **User Intent (Target Model):**
        {user_intent}
        
        **Task:**
        List the specific preprocessing steps required to prepare this data for the target model.
        
        **Considerations:**
        1. **Missing Values**: Choose imputation strategies suitable for the model (e.g., mean/median for linear models, constant for trees).
        2. **Categorical Encoding**: 
           - Use OneHotEncoder for low-cardinality nominal variables.
           - Use TargetEncoder or LabelEncoder for high-cardinality variables to avoid sparsity.
        3. **Feature Scaling**: CRITICAL for distance-based models (Linear Regression, KNN, SVM). Use StandardScaler or MinMaxScaler.
        4. **Performance**: If the dataset is large, prefer efficient transformers.
        5. **Dropping Columns**: Remove IDs, high-cardinality text, or columns with >50% missing values if not imputable.
        
        **Output Format:**
        Provide a concise, numbered list of steps with a brief rationale for each. Do not write code yet.
        """
        
        prompt = PromptTemplate(
            input_variables=["metadata", "user_intent"],
            template=template
        )
        
        chain = prompt | self.llm | self.output_parser
        return chain.invoke({"metadata": metadata, "user_intent": user_intent})

    def generate_code(self, metadata: str, strategy: str, previous_error: str = None) -> str:
        """Generates Python code to implement the strategy."""
        template = """
        You are an expert Python Data Engineer.
        
        **Dataset Metadata:**
        {metadata}
        
        **Preprocessing Strategy:**
        {strategy}
        
        **Task:**
        Write a robust Python function `def clean_data(df):` that takes a pandas DataFrame `df` and returns the processed DataFrame.
        
        **Constraints:**
        1. Use ONLY `pandas` and `scikit-learn`.
        2. **Prefer `sklearn.pipeline.Pipeline` and `ColumnTransformer`** for cleaner, more robust code.
        3. Handle missing values and categorical encoding as specified.
        4. The function MUST return the modified DataFrame.
        5. Import all necessary libraries inside the function or at the top.
        6. Be defensive: check if columns exist before dropping or modifying them.
        7. Return ONLY the valid Python code. Do not include markdown formatting like ```python.
        
        {error_context}
        
        **Code:**
        """
        
        error_context = ""
        if previous_error:
            error_context = f"**PREVIOUS ERROR TO FIX:**\n{previous_error}\n\nPlease fix the code to resolve this error."
            
        prompt = PromptTemplate(
            input_variables=["metadata", "strategy", "error_context"],
            template=template
        )
        
        chain = prompt | self.llm | self.output_parser
        code = chain.invoke({
            "metadata": metadata, 
            "strategy": strategy, 
            "error_context": error_context
        })
        
        # Clean up markdown if present
        code = code.replace("```python", "").replace("```", "").strip()
        return code

    def execute_code(self, df: pd.DataFrame, code: str):
        """Executes the generated code on the dataframe."""
        # Create a shared scope for execution to ensure imports are visible to functions
        # We pass this as both globals and locals to exec
        execution_scope = {'df': df.copy()}
        
        try:
            # Execute the code definition
            # Passing the same dict for globals and locals ensures that imports (which go to locals)
            # are visible to functions defined in the same exec call (which look in globals).
            exec(code, execution_scope, execution_scope)
            
            # Check if function exists
            if 'clean_data' not in execution_scope:
                raise ValueError("Function 'clean_data' was not defined in the generated code.")
            
            # Run the function
            clean_data_func = execution_scope['clean_data']
            processed_df = clean_data_func(execution_scope['df'])
            
            if not isinstance(processed_df, pd.DataFrame):
                raise ValueError("The 'clean_data' function did not return a pandas DataFrame.")
                
            return processed_df, None
            
        except Exception as e:
            return None, str(e)
