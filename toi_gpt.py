import streamlit as st
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI
import openai




# Function to load CSV file into DataFrame
if 'df' not in st.session_state:
    df =pd.read_excel('Summary2.xlsx')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop(columns='Link')
    st.session_state['df'] = df.copy(deep = True)
    

df = st.session_state['df'].copy(deep = True)
# Define LangChain prompt template to generate Python code
prompt_template = """
You are an AI assistant. You have access to a global dataframe `df`,so you dont need to do 'pd.read_csv' and you will generate Python code that answers the user's query using the dataset. 
The dataset contains columns such as 'Section', 'Date', 'Link', 'Content', and 'Summary'. Extract the Keyords given in user's query
The goal is to create Python code to extract relevant information based on the user's Keywords. 

E.g  If user asks about 'Ukraine Russia War' extract Key words of 'Ukraine', 'Russia' and 'War'.
. And then find those rows in 'Content' which have all the Keywords.
Ensure that all the Keywrods are present in the content. I dont want rows having only few of the Keywords. For any date related query, convert the date into 'yyyy-mm-dd' format by using 
query_date = pd.to_datetime('2025-06-01', errors='coerce') etc

E.g for user's question of 'Give me news about 'India', look into 'Content' column and find which row contains 'India'.
Similarly, for user's question of 'Give me news about 'Sports', 'Nation, 'World', 'Opnion' or 'Editorial, look into below columns:
'Sports' in 'Section' column containing 'Sports', 'Nation' in 'Nation' Section, 'World' in 'World' Section', 'Opninion' or 'Editorial' 
in Section of 'Feature', etc and do filtering etc.

For date related queries, look into 'Date' Section and apply filter.
The user's question: {user_query}

Please write a Python script that can answer the query.
 Don't give any description, just write relevant and correct and error-free Python code and store output in a variable called result.
        Ignore the case in 'df' and also ignore case in the question the user asks.
        Please generate a Python script using this 'df' as input dataframe and pandas to answer this question: "{user_query}".
       
        Write only the correct and error free code
         with exception handing read-only Python script and import streamlit as st. While using any column having Date values use 'dt' and not 'str'.
        Dont give any explanation while executing the python code.
        **Do not include any descriptions, explanations, or comments.**
. Ensure that the script reads the dataset and filters data accordingly. Return the Python script.
"""
# st.secrets = '6FlSZ2qDwe9kfsTaEIPb8PRUCRmNyhS2AIIDduT1hsi54ap2dYeAJQQJ99BFACHYHv6XJ3w3AAAAACOGV2L9'
# st.azure_endpoint = 'https://learn-mcboa439-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview'
               
llm = AzureChatOpenAI(
api_key = st.secrets['api_key'],
                azure_endpoint = st.secrets['azure_endpoint'],
                model = "gpt-4o",
                api_version="2024-02-15-preview",
                temperature = 0.
# other params...
) 

template = PromptTemplate(
        input_variables=["user_query"],
        template=prompt_template,
    )
    # When asked about plotting a graph, use st.pyplot(fig) where fig = plt.figure(figsize=(10,10)) 
            # While drawing the graphs, put x axis labels rotated by 90.    
    # # Create the LLMChain to manage the model and prompt interaction
llm_chain = LLMChain(prompt=template, llm=llm)

# template = PromptTemplate(input_variables=["text"], template=prompt_template)

# chain = LLMChain(llm=llm, prompt=PromptTemplate(template=prompt_template, input_variables=["user_query"]))

st.title("Times of India GPT")
st.markdown("You may want to watch the video before using the tool")
st.video("https://youtu.be/TSD2R9mntfg")
st.markdown("You can ask questions for any news item from 1st of June 2025 to 12th of July 2025")


# Show a preview of the dataset
st.write("Here is a preview of your dataset:")
st.dataframe(df.head())

user_query = st.text_input("Enter your query", "News about India")

if user_query :
# Query LangChain to generate Python code based on user query
    python_script = llm_chain.invoke({"user_query": user_query})

# Display the generated Python code
st.write("Generated Python script to answer the query:")
# st.code(python_script, language="python")

# Dynamically execute the Python script using exec()
try:
    # Define a safe environment for execution
    exec_globals = {"df": df}  # Only the df is available in the execution context
    python_script['text'] = python_script['text'].strip('`').replace('python', '')
    st.code(python_script['text'], language='python')
    # Execute the Python script
    exec(python_script['text'], exec_globals)
    
    # Retrieve and display the result from the execution context
    result = exec_globals.get('result', None)
    
    if result is not None:
        st.write("Here is the result based on your query:")
        st.dataframe(result)  # Display the filtered data as a dataframe
    else:
        st.write("No matching results found.")
except Exception as e:
    st.write(f"An error occurred while executing the code: {e}")
