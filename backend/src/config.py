"""Configuration management using environment variables."""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_SYSTEM_PROMPT = """You are an expert code analysis assistant helping developers understand their codebase.

Guidelines:
1. Provide clear, conversational answers - avoid mentioning "code snippets" or "provided files"
2. Answer questions based on what you observe in the repository
3. Be specific about file names, functions, and implementations when relevant
4. If something isn't found, say so clearly rather than being vague
5. Focus on answering the actual question directly and concisely
6. Reference the repository/codebase naturally (e.g., "In this repository..." or "The codebase uses...")

Special Instructions:
- When asked about API endpoints, LIST them explicitly with their HTTP methods (GET, POST, PUT, DELETE)
- Look for decorators like @app.get(), @app.post(), @router.get(), @router.post()
- For FastAPI/Flask apps, identify routes by their decorator patterns
- When you see a clear pattern (like multiple @app.post decorators), enumerate them confidently
- Don't say "endpoints are not explicitly listed" if you can see decorator patterns in the code

Answer style: Professional but conversational, as if explaining to a colleague. Be confident when the evidence is clear."""

# Model configurations - easily switch models by changing MODEL_NAME
MODEL_CONFIGS = {
    "phi3-mini": {
        "filename": "Q4_K_M-00001-of-00001.gguf",  # Phi-3 Mini 128k context
        "context_length": 8192,  # Using 8k for practical purposes (full 128k is available but slower)
        "max_tokens": 512,  # Increased for better responses
        "temperature": 0.7,
        "top_p": 0.95,
        "n_threads": 4,
        "n_gpu_layers": 0,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "stop_tokens": ["<|end|>", "<|endoftext|>", "<|user|>", "<|assistant|>", "\n\nUser Question:", "\n\nFile:"],
        "prompt_format": "phi3"  # Phi-3 uses special format
    },
    "codellama-7b": {
        "filename": "codellama-7b-merged-Q4_K_M.gguf",
        "context_length": 2048,
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.95,
        "n_threads": 2,
        "n_gpu_layers": 0,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "stop_tokens": ["</s>", "[INST]", "<|endoftext|>", "\n\nUser Question:", "\n\nFile:"],
        "prompt_format": "llama2"
    },
    "llama3.1-8b": {
        "filename": "llama-3.1-8b-instruct-q4_k_m.gguf",
        "context_length": 8192,
        "max_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.95,
        "n_threads": 4,
        "n_gpu_layers": 0,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "stop_tokens": ["<|end_of_text|>", "<|eot_id|>", "\n\nUser Question:", "\n\nFile:"],
        "prompt_format": "llama3"
    },
    "qwen2.5-coder-3b": {
        "filename": "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        "context_length": 8192,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95,
        "n_threads": 4,
        "n_gpu_layers": 0,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "stop_tokens": ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "\n\nUser Question:", "\n\nFile:"],
        "prompt_format": "chatml"
    }
}

# Default model - change this to switch models easily
DEFAULT_MODEL = "phi3-mini"


class Settings:
    """Application settings loaded from environment variables."""

    # GitHub Configuration
    GITHUB_TOKEN: Optional[str] = os.getenv("GITHUB_TOKEN")

    # Model Configuration - easily switch by changing MODEL_NAME
    MODEL_NAME: str = os.getenv("MODEL_NAME", DEFAULT_MODEL)

    # Indexing Configuration
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "1048576"))  # 1MB
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # RAG Configuration
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "4000"))

    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    @classmethod
    def get_github_token(cls) -> Optional[str]:
        """Get GitHub token from environment."""
        return cls.GITHUB_TOKEN

    @classmethod
    def get_model_path(cls) -> Path:
        """Get model path as Path object."""
        model_config = MODEL_CONFIGS.get(cls.MODEL_NAME, MODEL_CONFIGS[DEFAULT_MODEL])
        model_filename = model_config["filename"]
        full_path = BASE_DIR / "models" / model_filename
        return full_path
    
    @classmethod
    def get_model_config(cls):
        """Get model configuration dictionary."""
        return MODEL_CONFIGS.get(cls.MODEL_NAME, MODEL_CONFIGS[DEFAULT_MODEL])
    
    @classmethod
    def get_available_models(cls):
        """Get list of available model names."""
        return list(MODEL_CONFIGS.keys())


# Global settings instance
settings = Settings()
