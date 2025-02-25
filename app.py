import streamlit as st
import os
from mistralai import Mistral, UserMessage

# Set your Mistral API key (replace with your actual API key)
os.environ["MISTRAL_API_KEY"] = "gQeOXgiZv5437TMVhKrChGMQ98rtNyfc"
api_key = os.getenv("MISTRAL_API_KEY")

# Define a generic function to call Mistral AI
def mistral(user_message, model="mistral-large-latest", is_json=False):
    client = Mistral(api_key=api_key)
    messages = [UserMessage(content=user_message)]
    chat_response = client.chat.complete(model=model, messages=messages)
    return chat_response.choices[0].message.content

# Classification task: Categorize bank customer inquiries
def classify_customer_query(query):
    prompt = """
    You are a bank customer service bot.
    Your task is to assess customer intent and categorize the inquiry after <<<>>> into one of the following predefined categories:
    
    card arrival
    change pin
    exchange rate
    country support
    cancel transfer
    charge dispute
    
    If the text doesn't fit any of the above categories, classify it as:
    customer service
    
    You will only respond with the predefined category. Do not provide explanations or notes.
    
    <<<
    Inquiry: {inquiry}
    >>>
    Category:
    """
    final_prompt = prompt.format(inquiry=query)
    return mistral(final_prompt)

# Information Extraction task: Extract structured data from medical notes
def extract_medical_info(note):
    prompt = f"""
    Extract information from the following medical notes:
    {note}

    Return json format with the following JSON schema: 

    {{
        "age": {{
            "type": "integer"
        }},
        "gender": {{
            "type": "string",
            "enum": ["male", "female", "other"]
        }},
        "diagnosis": {{
            "type": "string",
            "enum": ["migraine", "diabetes", "arthritis", "acne"]
        }},
        "weight": {{
            "type": "integer"
        }},
        "smoking": {{
            "type": "string",
            "enum": ["yes", "no"]
        }}
    }}
    """
    return mistral(prompt, is_json=True)

# Personalized Email Response task: Generate a professional email reply
def generate_email_response(email_text):
    prompt = f"""
    You are a mortgage lender customer service bot, and your task is to create personalized email responses to address customer questions.
    Answer the customer's inquiry using the provided facts below. Ensure that your response is clear, concise, and directly addresses the customer's question. Address the customer in a friendly and professional manner. Sign the email with "Lender Customer Support."
    
    # Facts
    30-year fixed-rate: interest rate 6.403%, APR 6.484%
    20-year fixed-rate: interest rate 6.329%, APR 6.429%
    15-year fixed-rate: interest rate 5.705%, APR 5.848%
    10-year fixed-rate: interest rate 5.500%, APR 5.720%
    7-year ARM: interest rate 7.011%, APR 7.660%
    5-year ARM: interest rate 6.880%, APR 7.754%
    3-year ARM: interest rate 6.125%, APR 7.204%
    30-year fixed-rate FHA: interest rate 5.527%, APR 6.316%
    30-year fixed-rate VA: interest rate 5.684%, APR 6.062%
    
    # Email
    {email_text}
    """
    return mistral(prompt)

# Summarization task: Summarize a news article
def summarize_article(article_text):
    prompt = f"""
    Summarize the following news article in a few concise paragraphs:
    {article_text}
    
    Summary:
    """
    return mistral(prompt)

# Streamlit UI
st.title("Mistral AI Chatbot - Streamlit Deployment")
st.sidebar.header("Select Task")
task = st.sidebar.selectbox("Task", ["Classification", "Information Extraction", "Email Response", "Summarization"])

if task == "Classification":
    st.header("Bank Customer Inquiry Classification")
    inquiry = st.text_area("Enter customer inquiry:")
    if st.button("Classify Inquiry"):
        if inquiry:
            category = classify_customer_query(inquiry)
            st.write("Predicted Category:", category)
        else:
            st.warning("Please enter an inquiry.")

elif task == "Information Extraction":
    st.header("Medical Notes Information Extraction")
    note = st.text_area("Enter medical note:")
    if st.button("Extract Information"):
        if note:
            info = extract_medical_info(note)
            st.write("Extracted Information:", info)
        else:
            st.warning("Please enter a medical note.")

elif task == "Email Response":
    st.header("Personalized Email Response")
    email_text = st.text_area("Enter customer's email inquiry:")
    if st.button("Generate Email Response"):
        if email_text:
            email_response = generate_email_response(email_text)
            st.write("Email Response:")
            st.write(email_response)
        else:
            st.warning("Please enter an email inquiry.")

elif task == "Summarization":
    st.header("Article Summarization")
    article_text = st.text_area("Enter news article text:")
    if st.button("Summarize Article"):
        if article_text:
            summary = summarize_article(article_text)
            st.write("Article Summary:")
            st.write(summary)
        else:
            st.warning("Please enter an article.")

