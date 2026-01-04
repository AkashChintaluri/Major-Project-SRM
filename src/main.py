import sys
import torch
import chromadb
from pydantic import BaseModel

def system_check():
    print(f"Python Path: {sys.executable}")
    print(f"Torch CUDA Available: {torch.cuda.is_available()}")
    print(f"ChromaDB Version: {chromadb.__version__}")
    print("✅ Environment is ready for RAG Pipeline Construction.")

if __name__ == "__main__":
    system_check()