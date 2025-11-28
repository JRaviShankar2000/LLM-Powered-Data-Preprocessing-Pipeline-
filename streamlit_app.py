import streamlit as st
import pandas as pd
from agent.data_agent import DataAgent
import time

st.set_page_config(page_title="Smart Data Preprocessing Agent", layout="wide")

st.title("🤖 Smart Data Preprocessing Agent")
st.markdown("""
This agent helps you prepare your data for Machine Learning. 
Upload a CSV, tell us your target model, and let the agent handle the rest!
""")

# Initialize Agent
@st.cache_resource
def get_agent():
    try:
        return DataAgent()
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        return None

agent = get_agent()

# Sidebar
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

if 'df' in st.session_state:
    df = st.session_state['df']
    
    # Tabs Layout
    tab1, tab2, tab3 = st.tabs(["📊 Data Inspection", "🤖 Agent Workflow", "📥 Results & Export"])
    
    with tab1:
        st.subheader("Data Overview")
        
        st.info("👉 **Next Step**: Go to the '🤖 Agent Workflow' tab to configure and run preprocessing.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**First 5 Rows:**")
            st.dataframe(df.head())
            
        with col2:
            st.write("**Missing Values:**")
            nulls = df.isnull().sum()
            st.dataframe(nulls[nulls > 0], use_container_width=True)
            st.write(f"**Shape:** {df.shape}")
            st.write(f"**Columns:** {list(df.columns)}")

    with tab2:
        st.subheader("Configure & Run")
        
        # Model Selection using Radio Buttons (compatible with older Streamlit)
        st.write("**Select Target Model:**")
        model_options = ["Custom Input", "Linear Regression", "Logistic Regression", "Random Forest", "XGBoost", "K-Means Clustering"]
        selected_model = st.radio("Quick Select", model_options, horizontal=True)
        
        if selected_model == "Custom Input":
            user_intent = st.text_input(
                "Describe your custom intent:", 
                placeholder="e.g., Predict housing prices using a neural network."
            )
        else:
            user_intent = selected_model
            st.caption(f"Selected: **{selected_model}**")
        
        if st.button("🚀 Run Preprocessing Agent", type="primary"):
            if not user_intent:
                st.warning("Please select a model or describe your intent.")
            else:
                with st.status("Agent is working...", expanded=True) as status:
                    # Step 1: Analyze
                    st.write("🔍 Analyzing metadata...")
                    metadata = agent.analyze_metadata(df)
                    
                    # Step 2: Strategy
                    st.write("🧠 Generating strategy...")
                    strategy = agent.generate_strategy(metadata, user_intent)
                    st.markdown(f"**Strategy:**\n{strategy}")
                    
                    # Step 3: Code Generation
                    st.write("💻 Generating code...")
                    code = agent.generate_code(metadata, strategy)
                    st.caption("Generated Code")
                    st.code(code, language='python')
                    
                    # Step 4: Execution
                    st.write("⚙️ Executing code...")
                    processed_df, error = agent.execute_code(df, code)
                    
                    # Self-Correction Loop
                    if error:
                        st.error(f"Execution failed: {error}")
                        st.write("🔧 Attempting self-correction...")
                        code = agent.generate_code(metadata, strategy, previous_error=error)
                        st.caption("Corrected Code")
                        st.code(code, language='python')
                        processed_df, error = agent.execute_code(df, code)
                        
                        if error:
                            st.error(f"Self-correction failed: {error}")
                            status.update(label="Agent failed!", state="error")
                        else:
                            st.success("Self-correction successful!")
                            st.session_state['processed_df'] = processed_df
                            st.session_state['processing_code'] = code
                            status.update(label="Agent finished successfully!", state="complete")
                            
                            # Celebration and guidance
                            st.balloons()
                            st.success("🎉 **Processing Complete!** Your dataset has been preprocessed and is ready for ML model training.")
                            st.info("👉 **Go to the '📥 Results & Export' tab** to download your processed data and Python script.")
                    else:
                        st.session_state['processed_df'] = processed_df
                        st.session_state['processing_code'] = code
                        status.update(label="Agent finished successfully!", state="complete")
                        
                        # Celebration and guidance
                        st.balloons()
                        st.success("🎉 **Processing Complete!** Your dataset has been preprocessed and is ready for ML model training.")
                        st.info("👉 **Go to the '📥 Results & Export' tab** to download your processed data and Python script.")

    with tab3:
        if 'processed_df' in st.session_state:
            st.subheader("Results")
            st.success("Data processed successfully!")
            st.write("**Processed Data Preview:**")
            st.dataframe(st.session_state['processed_df'].head())
            
            st.write("**Download Options:**")
            col1, col2 = st.columns(2)
            with col1:
                csv = st.session_state['processed_df'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Processed CSV",
                    data=csv,
                    file_name="processed_data.csv",
                    mime="text/csv",
                )
                
            with col2:
                st.download_button(
                    label="📜 Download Python Script",
                    data=st.session_state['processing_code'],
                    file_name="preprocessing_script.py",
                    mime="text/x-python",
                )
        else:
            st.info("Run the agent in the 'Agent Workflow' tab to see results here.")
else:
    st.info("Please upload a CSV file to begin.")
