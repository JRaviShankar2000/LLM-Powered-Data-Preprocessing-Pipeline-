# Agentic-AutoML Data Preprocessing Pipeline

This repository contains an intelligent, LLM-powered data preprocessing pipeline that automatically generates and executes Python cleaning scripts based on your specific machine learning goals.

## 🚀 Features

- **LLM-Powered Logic**: Uses Google Gemini (or OpenAI) to reason about your data and goal.
- **Dynamic Code Generation**: Writes custom `pandas` and `scikit-learn` code for every dataset.
- **Secure Execution**: Runs generated code in a controlled scope.
- **Interactive UI**: Built with Streamlit for easy file upload, configuration, and visualization.
- **Full Transparency**: View, download, and audit the generated Python pipeline.

## 📂 Structure

```
Agentic-AutoML/
├── data/
│   ├── raw/          # Place raw CSVs here
│   └── processed/    # Cleaned data is saved here
├── src/
│   ├── __init__.py
│   ├── app.py        # Main Streamlit application
│   └── llm_handler.py # LLM interaction logic
├── .gitignore
├── README.md
└── requirements.txt
```

## 🛠️ Setup & Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run src/app.py
   ```

3. **Configure**:
   - Enter your **Google Gemini API Key** (or OpenAI Key) in the sidebar.
   - Upload your CSV file.
   - Describe your ML goal (e.g., "Train a Random Forest to predict churn").

4. **Generate & Download**:
   - Click **Generate Cleaning Pipeline**.
   - Review the generated code and cleaned data.
   - Download the processed CSV and the Python script.

## 🛡️ Security Note

This tool uses `exec()` to run LLM-generated code. While it includes safety prompts to avoid system calls, always review the generated code before execution, especially in production environments.
