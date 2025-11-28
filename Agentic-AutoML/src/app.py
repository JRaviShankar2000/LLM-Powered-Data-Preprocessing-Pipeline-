import streamlit as st
import pandas as pd
from llm_handler import preprocessing_agent
import io

st.set_page_config(page_title="Agentic AutoML Preprocessing", layout="wide")

# --- Sidebar: Configuration ---
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter API Key", type="password")
provider = st.sidebar.selectbox("Select Provider", ["Gemini", "OpenAI"])

st.sidebar.markdown("---")
st.sidebar.info(
    "This agent uses an LLM to generate a preprocessing pipeline for your dataset."
)

# --- Main Window ---
st.title("🤖 Agentic AutoML Data Preprocessing")

# 1. File Uploader
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        # 2. Data Profiling
        st.subheader("📊 Data Profiling")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**First 5 Rows**")
            st.dataframe(df.head())
            
        with col2:
            st.write("**Missing Values**")
            nulls = df.isnull().sum()
            st.dataframe(nulls[nulls > 0], use_container_width=True)
            
        with st.expander("View Statistical Description"):
            st.dataframe(df.describe())

        # 3. User Intent
        st.subheader("🎯 User Intent")
        user_goal = st.text_input(
            "What model are you training?",
            placeholder="e.g., I am training a Linear Regression model to predict house prices."
        )
        
        # 4. Generate Pipeline
        if st.button("✨ Generate Cleaning Pipeline", type="primary"):
            if not api_key:
                st.error("Please enter your API Key in the sidebar.")
            elif not user_goal:
                st.warning("Please describe your goal.")
            else:
                try:
                    # Initialize Agent
                    agent = preprocessing_agent(api_key, provider)
                    
                    with st.spinner("Generating preprocessing code..."):
                        # Prepare metadata
                        buffer = io.StringIO()
                        df.info(buf=buffer)
                        df_dtypes = buffer.getvalue()
                        df_head = df.head().to_string()
                        
                        # Generate Code
                        generated_code = agent.generate_cleaning_code(df_head, df_dtypes, user_goal)
                        
                    st.subheader("💻 Generated Pipeline")
                    st.code(generated_code, language="python")
                    
                    # 5. Execution
                    with st.spinner("Executing pipeline..."):
                        # Shared scope for execution
                        execution_scope = {'df': df.copy(), 'pd': pd}
                        
                        try:
                            exec(generated_code, execution_scope, execution_scope)
                            
                            if 'clean_data' not in execution_scope:
                                raise ValueError("Function 'clean_data' was not defined in the generated code.")
                            
                            clean_data_func = execution_scope['clean_data']
                            cleaned_df = clean_data_func(execution_scope['df'])
                            
                            st.success("✅ Data Cleaned Successfully!")
                            
                            st.subheader("✨ Cleaned Data Preview")
                            st.dataframe(cleaned_df.head())
                            
                            # 6. Downloads
                            col1, col2 = st.columns(2)
                            with col1:
                                csv = cleaned_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download Cleaned CSV",
                                    data=csv,
                                    file_name="cleaned_data.csv",
                                    mime="text/csv",
                                )
                            with col2:
                                st.download_button(
                                    label="📜 Download Pipeline Script",
                                    data=generated_code,
                                    file_name="preprocessing_pipeline.py",
                                    mime="text/x-python",
                                )
                                
                        except Exception as e:
                            st.error(f"❌ Execution Failed: {str(e)}")
                            st.error("Please check the generated code above for errors.")
                            
                except Exception as e:
                    st.error(f"Agent Error: {str(e)}")
                    
    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    st.info("Please upload a CSV file to begin.")
