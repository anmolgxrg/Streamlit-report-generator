from openai import OpenAI
client = OpenAI(api_key=${{ secrets.PSYCH_AR }})
import logging
import streamlit as st
import pandas as pd
from io import StringIO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up the API key

def get_psychar_report(collected_data, model="gpt-3.5-turbo", temperature=0.7):
    try:
        messages = [
        {
            "role": "system", 
            "content": "You are an expert in psychology and data analysis, focusing on ADHD diagnosis using AR testing."
        },
        {
            "role": "user", 
            "content": f"Here is the collected data from an ADHD CPT Test for IVA-2 CPT and TOVA CPT: {collected_data}. Please generate a detailed report with insights and some interpretations for the clinician, without providing recommendations."
        }
        ]

        response = client.chat.completions.create(model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)

        # Extract the generated text
        report = response.choices[0].message.content.strip()
        return report

    except openai.OpenAIError as e:
        logging.error(f"An error occurred: {e}")
        return "An error occurred while generating the ADHD report. Please try again later."
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return "An unexpected error occurred. Please try again later."

# Streamlit front end
def main():
    st.title("Psych-AR Report Generator")
    st.write("Upload the raw data from the Psych-AR.")

    # Text area for manual data input
    data_input = st.text_area("Enter Psych-AR data", height=200)

    # File uploader for data files
    uploaded_file = st.file_uploader("Upload Psych-AR data file", type=["txt", "csv"])
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            data_input = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        elif uploaded_file.type == "text/csv":
            data_df = pd.read_csv(uploaded_file)
            data_input = data_df.to_string(index=False)

    # Generate the report
    if st.button("Generate Report"):
        if data_input:
            with st.spinner("Generating report..."):
                psychar_report = get_psychar_report(data_input)
                if isinstance(psychar_report, str):
                    st.write("Psych-AR Report:")
                    st.text_area("Generated Report", psychar_report, height=300)
                else:
                    st.error(psychar_report)
        else:
            st.warning("Please enter or upload the Psych-AR data to generate the report.")

if __name__ == "__main__":
    main()
