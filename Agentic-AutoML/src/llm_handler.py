import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class preprocessing_agent:
    def __init__(self, api_key: str, provider: str = "gemini"):
        """
        Initializes the agent with the specified LLM provider.
        
        Args:
            api_key (str): The API key for the provider.
            provider (str): 'gemini' or 'openai'. Defaults to 'gemini'.
        """
        self.provider = provider.lower()
        self.api_key = api_key
        
        if self.provider == "gemini":
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.api_key,
                temperature=0.1
            )
        elif self.provider == "openai":
            self.llm = ChatOpenAI(
                model="gpt-4o",
                api_key=self.api_key,
                temperature=0.1
            )
        else:
            raise ValueError("Invalid provider. Choose 'gemini' or 'openai'.")
            
        self.output_parser = StrOutputParser()

    def generate_cleaning_code(self, df_head: str, df_dtypes: str, user_goal: str) -> str:
        """
        Generates Python code to clean the data based on the user's goal.
        
        Args:
            df_head (str): String representation of df.head()
            df_dtypes (str): String representation of df.dtypes
            user_goal (str): The user's specific objective (e.g., "Train Linear Regression")
            
        Returns:
            str: Executable Python code.
        """
        template = """
        You are a Senior Data Engineer and Machine Learning Expert.
        
        **Data Profile:**
        - **First 5 Rows:**
        {df_head}
        
        - **Data Types:**
        {df_dtypes}
        
        **User Goal:**
        {user_goal}
        
        **Task:**
        Write a robust Python script to preprocess this dataset for the user's goal.
        The script must define a function `def clean_data(df):` that takes the DataFrame as input and returns the cleaned DataFrame.
        
        **Requirements:**
        1. **Explainability**: Add comments explaining WHY you are performing each step (e.g., "Imputing age with median due to skew").
        2. **Robustness**: Handle missing values, encode categorical variables, and scale numerical features as appropriate for the target model.
        3. **Safety**: Do NOT use `os.system`, `subprocess`, or any dangerous system calls.
        4. **Libraries**: Use ONLY `pandas`, `numpy`, and `scikit-learn`. Import them inside the function or assume they are available.
        5. **Output**: Return ONLY the valid Python code. Do not include markdown formatting like ```python.
        
        **Code:**
        """
        
        prompt = PromptTemplate(
            input_variables=["df_head", "df_dtypes", "user_goal"],
            template=template
        )
        
        chain = prompt | self.llm | self.output_parser
        
        try:
            code = chain.invoke({
                "df_head": df_head,
                "df_dtypes": df_dtypes,
                "user_goal": user_goal
            })
            
            # Clean up markdown if present
            code = code.replace("```python", "").replace("```", "").strip()
            return code
            
        except Exception as e:
            return f"# Error generating code: {str(e)}"
