import os
import logging
import traceback
from typing import List, Dict, Any, Optional
from functools import lru_cache
from datetime import datetime
from contextlib import asynccontextmanager

import algosdk
from algosdk.v2client import algod
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Updated LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("algorand-assistant")

# Lifespan context manager (replacing on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Algorand Assistant API...")
    
    # Pre-initialize services to warm up the cache
    try:
        get_algod_client()
        logger.info("Algod client initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize Algod client: {str(e)}")
    
    try:
        get_llm()
        logger.info("LLM initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize LLM: {str(e)}")
    
    try:
        setup_vector_store()
        logger.info("Vector store initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize vector store: {str(e)}")
    
    logger.info("Startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Algorand Assistant API...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Algorand Assistant API",
    description="A backend service for an Algorand-focused chatbot with RAG capabilities",
    version="1.0.0",
    lifespan=lifespan,
)

# Use a relative path for static files (ensures portability)
app.mount("/static", StaticFiles(directory=r"C:\Users\mail2\OneDrive\Documents\GitHub\Algo-assistant-lohit\Static"), name="static")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    query: str = Field(..., description="User's question or query")
    history: List[Dict[str, str]] = Field(
        default_factory=list, description="Chat history (optional)"
    )

class TealCodeRequest(BaseModel):
    code: str = Field(..., description="TEAL code to compile")

class AlgorandTransaction(BaseModel):
    sender: str = Field(..., description="Transaction sender address")
    receiver: str = Field(..., description="Transaction receiver address")
    amount: int = Field(..., description="Transaction amount in microAlgos")
    note: Optional[str] = Field(None, description="Optional note for the transaction")


# Global variables for caching
ALGORAND_DOCS = [
    "https://www.algorand.foundation/about-us",
    "https://algorand.foundation/algorand-protocol",
    "https://www.algorand.com/technology",
    "https://developer.algorand.org/docs/get-started/basics/why_algorand/",
    "https://developer.algorand.org/docs/get-details/accounts/",
    "https://developer.algorand.org/docs/get-details/transactions/",
    "https://developer.algorand.org/docs/get-details/dapps/smart-contracts/",
    "https://developer.algorand.org/docs/get-details/asa/",
    "https://developer.algorand.org/docs/get-details/atomic_transfers/",
    "https://developer.algorand.org/docs/get-details/algorand-consensus/",
    "https://developer.algorand.org/docs/sdks/python/",
    "https://developer.algorand.org/docs/get-started/dapps/pyteal/",
]

ALGORAND_KNOWLEDGE_BASE = {
    "what is algorand": """
# What is Algorand?

Algorand is a blockchain platform founded by Turing Award-winning cryptographer Silvio Micali in 2017. It's designed to solve the blockchain trilemma by providing security, scalability, and decentralization without compromise.

## Core Features

Algorand uses a Pure Proof-of-Stake (PPoS) consensus mechanism that randomly and secretly selects validators from all token holders, ensuring high security and true decentralization.

Key strengths of Algorand include:
- Fast transaction finality (under 5 seconds)
- Low transaction costs (fraction of a cent)
- Carbon-negative blockchain
- Smart contract capabilities
- Asset tokenization through Algorand Standard Assets (ASAs)
- No forking, providing instant finality for transactions

## Technology Highlights

The Algorand blockchain achieves its performance through:
- Two-phase block production process
- Verifiable random function (VRF) for validator selection
- Byzantine agreement protocol for consensus
- Layer-1 smart contracts with developer-friendly languages

Algorand serves use cases across DeFi, NFTs, CBDCs, and enterprise blockchain solutions.
""",

    "algorand history": """
# Algorand History

Algorand was founded in 2017 by Silvio Micali, an MIT professor and Turing Award-winning cryptographer known for fundamental contributions to cryptography.

## Key Milestones

2017: Algorand founded by Silvio Micali
2019: Mainnet launch and initial token sale
2020: Introduction of Algorand 2.0 with ASAs and Atomic Transfers
2021: Major ecosystem growth with DeFi and NFT platforms
2022: State Proofs feature added for enhanced interoperability
2023: TPS upgrades and major institutional partnerships
## Foundation

The Algorand Foundation, based in Singapore, oversees the ecosystem development, while Algorand Inc. focuses on protocol development. The platform has grown to support hundreds of applications across DeFi, gaming, NFTs, and enterprise solutions.

The ALGO token is the native cryptocurrency of the Algorand blockchain, used for transaction fees, governance, and staking.
""",

    "algorand consensus": """
# Algorand Consensus Mechanism

Algorand uses Pure Proof-of-Stake (PPoS), a unique consensus mechanism designed to be secure, scalable, and decentralized.

## How PPoS Works

Validator Selection: Users are secretly and randomly selected to propose and validate blocks proportional to their stake
Two-Phase Process:
Block proposal phase (single user proposes a block)
Block certification phase (committee votes on proposed block)
VRF Technology: Cryptographic sortition using Verifiable Random Functions selects validators
Byzantine Agreement: Modified Byzantine agreement ensures consensus even with malicious participants
## Key Advantages

Energy efficient compared to Proof-of-Work
No forking capability (instant finality)
High degree of decentralization (any token holder can participate)
Strong security guarantees even with significant percentage of malicious users
Performance efficiency (thousands of transactions per second)
The consensus mechanism is mathematically proven to provide security guarantees as long as 2/3 of the voting power is honest.
"""
}

def normalize_query(query: str) -> str:
    """Normalize a query by removing extra spaces, punctuation and converting to lowercase"""
    import re
    # Remove punctuation and extra spaces
    query = re.sub(r'[^\w\s]', '', query.lower())
    query = re.sub(r'\s+', ' ', query).strip()
    return query

def match_knowledge_base(query: str) -> Optional[str]:
    """Match a normalized query against the knowledge base with improved matching"""
    normalized = normalize_query(query)
    
    # Direct matches
    if normalized in ALGORAND_KNOWLEDGE_BASE:
        return ALGORAND_KNOWLEDGE_BASE[normalized]
    
    # Check for key phrases/words - more flexible matching
    key_phrases = {
        "what is algorand": ["what is algorand", "what's algorand", "tell me about algorand", 
                           "algorand is", "explain algorand", "describe algorand"],
        "algorand history": ["algorand history", "history of algorand", "when was algorand", 
                           "algorand timeline", "algorand founded", "algorand creation"],
        "algorand consensus": ["algorand consensus", "algorand pos", "algorand proof of stake",
                             "how does algorand work", "algorand mechanism"]
    }
    
    # Check each key phrase for a match
    for kb_key, phrases in key_phrases.items():
        if any(phrase in normalized for phrase in phrases):
            return ALGORAND_KNOWLEDGE_BASE[kb_key]
    
    # Very basic check for just "algorand" by itself
    if normalized == "algorand":
        return ALGORAND_KNOWLEDGE_BASE["what is algorand"]
        
    return None

def categorize_query(query: str) -> str:
    """Categorize the query to determine appropriate response type"""
    query_lower = normalize_query(query)
    
    # Informational queries
    info_keywords = ["what", "who", "explain", "tell", "history", "about", 
                    "define", "learn", "introduction", "detail", "describe"]
    
    # Technical queries
    tech_keywords = ["how", "code", "implement", "build", "develop", "create", 
                    "program", "deploy", "api", "sdk", "setup", "install"]
    
    if any(keyword in query_lower for keyword in info_keywords):
        return "informational"
    elif any(keyword in query_lower for keyword in tech_keywords):
        return "technical"
    else:
        return "general"

@lru_cache(maxsize=1)
def get_algod_client():
    """Initialize and return an Algod client."""
    algod_address = os.getenv("ALGOD_ADDRESS", "https://testnet-api.algonode.cloud")
    algod_token = os.getenv("ALGOD_TOKEN", "")
    
    try:
        # Set user agent to avoid warning
        headers = {"User-Agent": "algorand-assistant/1.0"}
        return algod.AlgodClient(algod_token, algod_address, headers)
    except Exception as e:
        logger.error(f"Failed to initialize Algod client: {str(e)}")
        raise


@lru_cache(maxsize=1)
def get_llm():
    """Initialize and return the Gemini 1.5 Flash model with optimized parameters."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.3,  # Slightly higher for better creative explanations
        top_p=0.92,
        top_k=30,         # Better focus on high-quality tokens
        max_output_tokens=2048,
        additional_kwargs={
            "safety_settings": {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
            }
        }
    )

@lru_cache(maxsize=1)
def get_embeddings():
    """Initialize and return the embeddings model."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )

def split_documents(documents):
    """Split documents into smaller chunks with better overlap for context."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Smaller chunks for more precise retrieval
        chunk_overlap=200,     # Sufficient overlap to maintain context
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks")
    return chunks

@lru_cache(maxsize=1)
def setup_vector_store():
    """Initialize the RAG vector store with enhanced document processing."""
    logger.info("Setting up vector store with Algorand documentation...")
    
    try:
        # Use persistent storage to avoid reloading on every restart
        vector_store_path = "algorand_faiss_index"
        
        # FORCE REGENERATION FOR TESTING (remove this for production)
        if os.path.exists(vector_store_path):
            import shutil
            shutil.rmtree(vector_store_path)
            logger.info(f"Deleted existing vector store at {vector_store_path} to force refresh")
        
        # Load documents from URLs
        documents = []
        for url in ALGORAND_DOCS:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded documentation from {url}")
            except Exception as e:
                logger.error(f"Failed to load {url}: {str(e)}")
        
        # Add our knowledge base directly to the documents
        for key, content in ALGORAND_KNOWLEDGE_BASE.items():
            documents.append(Document(page_content=content, metadata={"source": f"knowledge_base_{key}"}))
            logger.info(f"Added knowledge base entry: {key}")
        
        if not documents:
            logger.warning("No documents were loaded for the vector store")
            # Fallback to a minimal document to prevent crashes
            documents = [Document(page_content="Algorand is a blockchain platform.")]
        
        # Split documents and create vector store
        chunks = split_documents(documents)
        embeddings = get_embeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Save vector store for future use
        vector_store.save_local(vector_store_path)
        logger.info(f"Successfully created and saved FAISS vector store to {vector_store_path}")
        
        return vector_store
    except Exception as e:
        logger.error(f"Failed to setup vector store: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def create_prompt_template(query: str, query_type: str, context: str = "", history: List[Dict[str, str]] = None) -> str:
    """Create an optimized prompt template for Gemini 1.5 Flash based on query type"""
    
    # Format chat history if provided
    history_text = ""
    if history:
        for msg in history:
            if msg.get("role") == "user":
                history_text += f"Human: {msg.get('content')}\n"
            elif msg.get("role") == "assistant":
                history_text += f"Assistant: {msg.get('content')}\n"
    
    # Base system instruction
    system_instruction = """You are AlgoAssist, an Algorand blockchain specialist with comprehensive knowledge about the Algorand ecosystem."""
    
    # Customize instructions based on query type
    if query_type == "informational":
        instructions = """
INSTRUCTIONS:
- Answer questions about Algorand clearly and directly
- Provide comprehensive educational content for informational questions
- Begin with a clear, concise definition or explanation
- Explain Algorand's core concepts, history, and unique value proposition
- Use headers and well-structured explanations
- Cover key points before details
- Don't list technical services you can provide unless specifically asked
"""
    elif query_type == "technical":
        instructions = """
INSTRUCTIONS:
- Provide practical, actionable technical guidance
- Include working code samples when appropriate
- Explain code and technical concepts thoroughly
- Reference official documentation when relevant
- Be specific rather than general in technical advice
- Provide step-by-step instructions for complex processes
"""
    else:  # general
        instructions = """
INSTRUCTIONS:
- Answer questions about Algorand clearly and directly
- Balance informational content with practical guidance
- Determine if the query needs educational content, technical help, or both
- For mixed queries, provide education first, then technical assistance
- Be conversational but precise in your response
"""
    
    # Create the prompt with improved system instruction
    prompt = f"""
{system_instruction}

CONTEXT INFORMATION:
{context}

{instructions}

CONVERSATION HISTORY:
{history_text}
Human: {query}
Assistant: """

    return prompt

@app.get("/")
async def root():
    """Root endpoint for API health check."""
    return {"status": "online", "service": "Algorand Assistant API"}


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Process user queries about Algorand using enhanced RAG retrieval."""
    try:
        query = request.query
        history = request.history
        
        logger.info(f"Received query: '{query}'")

        if query.lower().strip() == "what is algorand":
            logger.info("Direct match for 'what is algorand' query detected")
        
        # First check if this matches our knowledge base
        kb_response = match_knowledge_base(query)
        if kb_response:
            logger.info("Found match in knowledge base")
            return {
                "response": kb_response,
                "timestamp": datetime.now().isoformat(),
                "source": "knowledge_base",
            }
        else:
            logger.info("No match found in knowledge base")
        
        # Categorize the query
        query_type = categorize_query(query)
        logger.info(f"Query type: {query_type}")
        
        # Initialize RAG components
        vector_store = setup_vector_store()
        
        # Retrieve relevant context with improved parameters
        retriever = vector_store.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": 6,         # Return more documents
                "fetch_k": 15,  # Consider more candidates
                "lambda_mult": 0.7  # Balance between relevance and diversity
            }
        )
        relevant_docs = retriever.get_relevant_documents(query)
        
        # Process and format the context better
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            # Add source metadata if available
            source = doc.metadata.get("source", f"Source {i+1}")
            context_parts.append(f"[{source}]\n{doc.page_content.strip()}")
        
        # Join with clear separation
        context = "\n\n---\n\n".join(context_parts)
        
        # Create the prompt with context and history
        prompt = create_prompt_template(query, query_type, context, history)
        
        # Generate response using LLM
        llm = get_llm()
        response = llm.invoke(prompt)
        
        # Return response with metadata
        return {
            "response": response.content,
            "timestamp": datetime.now().isoformat(),
            "source_count": len(relevant_docs),
            "query_type": query_type
        }
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    

@app.post("/compile-teal")
async def compile_teal_endpoint(request: TealCodeRequest):
    """Compile TEAL code using Algorand SDK."""
    try:
        algod_client = get_algod_client()
        compiled_result = algod_client.compile(request.code)
        
        return {
            "result": compiled_result,
            "timestamp": datetime.now().isoformat(),
        }
    
    except Exception as e:
        logger.error(f"Error compiling TEAL: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={
                "error": str(e),
                "detail": traceback.format_exc(),
            },
        )


@app.get("/algorand/status")
async def algorand_status():
    """Get current Algorand network status."""
    try:
        algod_client = get_algod_client()
        status = algod_client.status()
        
        return {
            "network": os.getenv("ALGOD_ADDRESS", "testnet"),
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }
    
    except Exception as e:
        logger.error(f"Error getting Algorand status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )

@app.get("/algorand/account/{address}")
async def account_info(address: str):
    """Get information about an Algorand account."""
    try:
        algod_client = get_algod_client()
        account_info = algod_client.account_info(address)
        
        return {
            "account_info": account_info,
            "timestamp": datetime.now().isoformat(),
        }
    
    except Exception as e:
        logger.error(f"Error getting account info: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"error": str(e)},
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "algoing:app",  # Use the filename and app variable
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "False").lower() == "true",
    )
