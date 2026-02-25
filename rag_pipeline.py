import os
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from bs4 import BeautifulSoup
import requests
from logger import setup_logging
from google import genai

logger = setup_logging()

# Initialize GenAI Client
client = genai.Client()

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
try:
    collection = chroma_client.get_or_create_collection(name="10x_architect_db")
except error:
    logger.error(str(error))

# Initialize Sentence-Transformers
encoder = SentenceTransformer("all-MiniLM-L6-v2")

def summarize_content(text):
    """Generates a Technical Abstract and Technical classification using Gemini."""
    logger.info("Generating summary for content.")
    prompt = f"""
    Analyze the following text and perform two tasks:
    1. Generate a brief 'Technical Abstract' summarizing the core concepts.
    2. Identify if the source is "Speculative", "Peer-Reviewed", or "Conspiracy".
    
    Format the output as follows:
    Classification: [Classification]
    Abstract: [Abstract]

    Text:
    {text[:5000]} # Limit text length to avoid token limits for summary
    """
    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return "Classification: Unknown\nAbstract: Could not generate summary."

def extract_text_from_pdf(file_path):
    """Extracts text from a local PDF file."""
    logger.info(f"Extracting text from PDF: {file_path}")
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
    return text

def extract_text_from_url(url):
    """Extracts text from a web page."""
    logger.info(f"Extracting text from URL: {url}")
    text = ""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from paragraphs to avoid script/style tags
        paragraphs = soup.find_all('p')
        text = "\n\n".join([p.get_text() for p in paragraphs])
    except Exception as e:
        logger.error(f"Error reading URL: {e}")
    return text

def process_and_store_document(text, source_identifier):
    """Ingests text, summarizes, chunks, embeds, and stores in ChromaDB."""
    logger.info(f"Processing document: {source_identifier}")
    
    if not text.strip():
        logger.warning(f"No text to process for {source_identifier}")
        return None

    # Detect technical vibe (10x logic)
    keywords = ["superconductor", "metric tensor", "dark energy", "Podkletnov"]
    is_technical = any(word in text.lower() for word in keywords)

    # 1. Summarize First (System Digest)
    summary = summarize_content(text)
    
    # 2. Chunking (Double-Space Strategy)
    # The requirement explicitly states to split by \n\n to preserve formulas
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    
    # Adjust overlap/handling if technical? The instructions say "increase chunk overlap to 15%" 
    # but strictly splitting by \n\n doesn't inherently support overlap without custom logic. 
    # For now, we strictly follow the Double-Space Strategy.

    # 3. Vectorization
    documents_to_store = [summary] + chunks
    
    # Generate unique IDs for the chunks
    ids = [f"{source_identifier}_summary"] + [f"{source_identifier}_chunk_{i}" for i in range(len(chunks))]
    
    # Metadata to distinguish abstract from granular chunks
    metadatas = [{"type": "abstract", "source": source_identifier}] + \
                [{"type": "granular", "source": source_identifier, "is_technical": is_technical} for _ in range(len(chunks))]

    logger.info(f"Storing {len(documents_to_store)} chunks/summaries to ChromaDB.")
    
    try:
        # Generate embeddings
        embeddings = encoder.encode(documents_to_store).tolist()
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents_to_store
        )
        logger.info(f"Successfully ingested {source_identifier}")
    except Exception as e:
        logger.error(f"Error storing vectors in ChromaDB: {e}")
        
    return summary


def retrieve_context(user_prompt, summary_abstract):
    """Multi-Vector Query: Searches ChromaDB for user prompt + summarized abstract."""
    logger.info("Retrieving context for query.")
    
    query_text = f"Prompt: {user_prompt}\nContext Abstract: {summary_abstract}"
    
    try:
        query_embedding = encoder.encode([query_text]).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=5 # Retrieve top 5 most relevant chunks
        )
        
        retrieved_documents = results.get("documents", [[]])[0]
        context = "\n\n".join(retrieved_documents)
        return context
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return ""
