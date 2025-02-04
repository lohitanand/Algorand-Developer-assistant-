from dotenv import load_dotenv
import os
import logging
import traceback
from typing import Optional

load_dotenv()  # Load environment variables first

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from algosdk.v2client import algod
from algosdk.error import AlgodHTTPError
from fastapi import Request
from starlette.responses import JSONResponse
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set user agent for Algorand API
os.environ["USER_AGENT"] = "AlgorandAssistant/1.0"

# Load API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logging.error("⚠️ GOOGLE_API_KEY not found in .env file!")
    raise ValueError("⚠️ GOOGLE_API_KEY not found in .env file!")

# --- Configuration ---
ALGOD_ENDPOINT = os.getenv("ALGOD_ENDPOINT", "https://testnet-api.algonode.cloud")

@lru_cache()
def get_algod_client():
    """Create and cache the Algorand client."""
    return algod.AlgodClient("", ALGOD_ENDPOINT)

app = FastAPI(
    title="Algorand Assistant",
    description="An API for interacting with Algorand blockchain and providing assistance",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Knowledge Base Initialization ---
@lru_cache()
def initialize_knowledge_base():
    """Initialize and cache the knowledge base retriever."""
    urls = [
        "https://developer.algorand.org/docs/get-started/smart-contracts/",
        "https://developer.algorand.org/docs/rest-apis/algod/v2/",
        "https://pyteal.readthedocs.io/en/stable/"
    ]
    
    try:
        loader = WebBaseLoader(urls)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)
        
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=GOOGLE_API_KEY,
            model="embedding-001"  # Updated to correct embedding model
        )
        vector_store = FAISS.from_documents(split_docs, embeddings)
        return vector_store.as_retriever()
    except Exception as e:
        logging.error(f"Failed to initialize knowledge base: {e}")
        raise RuntimeError("Failed to initialize knowledge base")

# Initialize the LLM with the correct model
@lru_cache()
def get_llm():
    """Create and cache the LLM instance."""
    return ChatGoogleGenerativeAI(
        model="gemini-pro",  # Updated to current model name
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7
    )

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str = Field(..., description="The message to process")
    address: Optional[str] = Field(None, description="Optional Algorand address")

class TealCodeRequest(BaseModel):
    teal_code: str = Field(..., description="TEAL code to compile")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The response message")

class TealCompileResponse(BaseModel):
    hash: str = Field(..., description="Compiled TEAL hash")
    bytecode: str = Field(..., description="Compiled TEAL bytecode")

# --- Exception Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler with detailed logging."""
    logging.error(f"HTTP Exception: {exc.detail} (status_code={exc.status_code})")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unexpected errors."""
    logging.error(f"Unexpected error: {str(exc)}")
    logging.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status_code": 500}
    )

# --- API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    algod_client: algod.AlgodClient = Depends(get_algod_client),
    llm: ChatGoogleGenerativeAI = Depends(get_llm)
):
    try:
        message = request.message.lower()
        
        if "balance" in message and request.address:
            try:
                account_info = algod_client.account_info(request.address)
                return ChatResponse(response=f"Balance: {account_info['amount']/1e6:.6f} ALGO")
            except AlgodHTTPError as e:
                raise HTTPException(status_code=400, detail="Invalid Algorand Address")

        if "network status" in message:
            status = algod_client.status()
            last_round = status.get("last-round", "Unknown")
            time = status.get("time-since-last-round", "Unknown")
            return ChatResponse(response=f"Network Status - Last Round: {last_round}, Time Since Last Round: {time}s")
            
        if "create asset" in message:
            return ChatResponse(response="""PyTeal ASA Template:
from pyteal import *

def asa_creation():
    return Seq([
        Assert(Txn.application_args.length() == Int(1)),
        App.globalPut(Bytes("total_supply"), Btoi(Txn.application_args[0])),
        App.globalPut(Bytes("creator"), Txn.sender()),
        Return(Int(1))
    ])

def clear_state():
    return Return(Int(1))

def approval_program():
    program = Cond(
        [Txn.application_id() == Int(0), asa_creation()],
        [Txn.on_completion() == OnComplete.DeleteApplication, Return(Int(1))],
        [Txn.on_completion() == OnComplete.UpdateApplication, Return(Int(1))],
        [Txn.on_completion() == OnComplete.CloseOut, Return(Int(1))],
        [Txn.on_completion() == OnComplete.OptIn, Return(Int(1))]
    )
    return program""")
            
        # General questions using LLM
        result = llm.invoke(request.message)
        return ChatResponse(response=str(result.content))

    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/compile-teal", response_model=TealCompileResponse)
async def compile_teal(
    request: TealCodeRequest,
    algod_client: algod.AlgodClient = Depends(get_algod_client)
):
    try:
        compile_response = algod_client.compile(request.teal_code)
        return TealCompileResponse(
            hash=compile_response["hash"],
            bytecode=compile_response["result"]
        )
    except AlgodHTTPError as e:
        logging.error(f"TEAL Compilation error: {str(e)}")
        raise HTTPException(status_code=400, detail="TEAL compilation failed")
    except Exception as e:
        logging.error(f"Error during TEAL compilation: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    