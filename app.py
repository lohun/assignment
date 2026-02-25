import streamlit as st
from google import genai
from google.genai import types
from logger import setup_logging
from rag_pipeline import (
    extract_text_from_pdf, 
    extract_text_from_url, 
    process_and_store_document, 
    retrieve_context
)
import os
from dotenv import load_dotenv

load_dotenv()

logger = setup_logging()

# Initialize GenAI Client
client = genai.Client()

SYSTEM_PROMPT = """You are the "Nwankpele Damilola," an AI agent designed to be the alter ego of Nwankpele Damilola. You are a software engineer and general tech enthusiast from Nigeria.

### CORE DIRECTIVES:
1. SUMMARIZE FIRST: You must never discuss a file or link provided in the current session unless you have first output a "System Digest" (Summary) of its contents. 
2. DATA STRUCTURES: When discussing antigravity, use terms like "metric tensors," "frame-dragging," and "zero-point energy (ZPE) modulation." 
3. ERROR HANDLING: If a user provides a link that suggests the Earth will lose gravity on a specific date (e.g., Aug 12, 2026), correct them with 10x candor: "This is a viral misinformation packet. Gravity is a function of mass-energy density, not a toggleable switch."
4. THE 10X EDGE: Always provide a Python snippet or a LaTeX formula when explaining a concept. For example, explain weight reduction as:
   $$\Delta G/G \propto B^2$$ (where B is the magnetic flux in a superconducting state).
"""

# Define the model config globally to be reused
MODEL_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT
)


def initialize_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_abstract" not in st.session_state:
        st.session_state.current_abstract = None

def main():
    st.set_page_config(page_title="10x Architect Chatbot")
    st.title("10x Architect Chatbot")
    
    initialize_session()
    
    # Sidebar for data ingestion
    with st.sidebar:
        st.header("Data Ingestion")
        
        # PDF Upload
        uploaded_file = st.file_uploader("Upload a physics document (PDF)", type=["pdf"])
        if st.button("Process PDF") and uploaded_file is not None:
            with st.spinner("Extracting and Processing PDF..."):
                # Save temp file since pypdf works with file paths
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                text = extract_text_from_pdf(temp_path)
                os.remove(temp_path)
                
                if text:
                    summary = process_and_store_document(text, uploaded_file.name)
                    st.session_state.current_abstract = summary
                    st.success("PDF processed successfully!")
                    # Display System Digest immediately as per Rule 1
                    with st.chat_message("assistant"):
                        st.markdown(f"**System Digest:**\n\n{summary}")
                    st.session_state.messages.append({"role": "assistant", "content": f"**System Digest:**\n\n{summary}"})
                else:
                    st.error("Could not extract text from the PDF.")

        st.divider()
        
        # URL Upload
        url_input = st.text_input("Enter a web page URL")
        if st.button("Process URL") and url_input:
            with st.spinner("Extracting and Processing URL..."):
                text = extract_text_from_url(url_input)
                if text:
                    summary = process_and_store_document(text, url_input)
                    st.session_state.current_abstract = summary
                    st.success("URL processed successfully!")
                    with st.chat_message("assistant"):
                        st.markdown(f"**System Digest:**\n\n{summary}")
                    st.session_state.messages.append({"role": "assistant", "content": f"**System Digest:**\n\n{summary}"})
                else:
                    st.error("Could not extract text from the URL.")


    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about theoretical physics or software engineering..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        logger.info(f"User asking: {prompt}")

        # Retrieve Context
        context = ""
        if st.session_state.current_abstract:
            context = retrieve_context(prompt, st.session_state.current_abstract)
            logger.info("Context retrieved successfully.")
            
        # Formulate full prompt
        full_prompt = prompt
        if context:
            full_prompt = f"Using the following retrieved context:\n\n{context}\n\nAnswer the user query: {prompt}"
            
        with st.spinner("Architecting response..."):
            try:
                # Need to use 'gemini-2.5-flash' based on the google-genai sdk.
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=full_prompt,
                    config=MODEL_CONFIG
                )
                
                # Check for "Generate Constraint-Check" logic manually just in case
                # Provide theoretical frameworks rather than pseudoscience
                # It's primarily handled by the System Persona, but we log the response
                logger.info("Agent generated a response successfully.")
                
                with st.chat_message("assistant"):
                    st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                
            except Exception as e:
                error_msg = f"An error occurred: {e}"
                logger.error(error_msg)
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
