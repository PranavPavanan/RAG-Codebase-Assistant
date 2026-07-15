import asyncio
import sys
from src.services.rag_service import RAGService
import logging

logging.basicConfig(level=logging.INFO)

async def main():
    r = RAGService()
    res = await r._load_indices()
    print("Result:", res)

if __name__ == "__main__":
    asyncio.run(main())
