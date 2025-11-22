"""
LLM Service - Multi-provider LLM abstraction with SSL Bypass
Supports: OpenAI, Anthropic, HuggingFace, Ollama, Azure
Token counting with tiktoken cache for performance
"""

import os
import ssl
import json
import httpx
import urllib3
import requests
from typing import List, Dict, Optional, Any
from functools import lru_cache
from langchain_core.language_models import BaseChatModel

# ==============================
# AGGRESSIVE SSL BYPASS (MUST BE AT TOP)
# ==============================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

original_request = requests.Session.request
def patched_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return original_request(self, method, url, **kwargs)
requests.Session.request = patched_request

original_get = requests.get
def patched_get(url, **kwargs):
    kwargs['verify'] = False
    return original_get(url, **kwargs)
requests.get = patched_get

os.environ['POSTHOG_DISABLE_GZIP'] = 'true'
os.environ['POSTHOG_DEBUG'] = 'false'

# ==============================
# TIKTOKEN
# ==============================
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class LLMService:
    """Unified interface for multiple LLM providers with SSL bypass"""
    
    def __init__(self, config: dict):
        """
        Initialize LLM service with configuration
        
        Args:
            config: Dictionary from llm_config.json
        """
    
        self.config = config
        self.provider = config.get('default_provider', 'openai')
        self.providers_config = config.get('llm_providers', {})
        self.embeddings_config = config.get('embedding_providers', {})
        
        self.llm = self._initialize_llm()
        self.embeddings = self._initialize_embeddings()

        print(f"[LLMService] Initialized with provider: {self.provider}")
    
    def _initialize_llm(self) -> BaseChatModel:
        """Initialize chat model based on configured provider"""
        provider_config = self.providers_config.get(self.provider, {})
        
        if not provider_config.get('enabled', False):
            # Find first enabled provider
            for name, cfg in self.providers_config.items():
                if cfg.get('enabled', False):
                    self.provider = name
                    provider_config = cfg
                    break
        
        try:
            print(self.provider,"provider type")
            if self.provider == 'azure':
                return self._create_azure(provider_config)
            elif self.provider == 'openai':
                return self._create_openai(provider_config)
            elif self.provider == 'anthropic':
                return self._create_anthropic(provider_config)
            elif self.provider == 'huggingface':
                return self._create_huggingface(provider_config)
            elif self.provider == 'ollama':
                return self._create_ollama(provider_config)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize {self.provider}: {e}")
            # Fallback to Ollama
            print("[INFO] Falling back to Ollama")
            return self._create_ollama(self.providers_config.get('ollama', {}))
    
    def _create_azure(self, config: dict) -> BaseChatModel:
        """Create Azure OpenAI chat model with SSL bypass"""
        from langchain_openai import ChatOpenAI
        
        api_key = config.get('api_key_env', 'AZURE_API_KEY')
        if not api_key:
            raise ValueError("Azure API key not found")
        
        # Create httpx client with SSL verification disabled
        client = httpx.Client(verify=False)
        
        return ChatOpenAI(
            base_url=config.get('api_endpoint'),
            api_key=api_key,
            model=config.get('model', 'azure/genailab-maas-gpt-4o'),
            temperature=config.get('temperature', 0.3),
            max_tokens=config.get('max_tokens', 2000),
            http_client=client
        )
    
    def _create_openai(self, config: dict) -> BaseChatModel:
        """Create OpenAI chat model with SSL bypass"""
        from langchain_openai import ChatOpenAI
        
        api_key = os.getenv(config.get('api_key_env', 'OPENAI_API_KEY'))
        if not api_key:
            raise ValueError("OpenAI API key not found")
        
        client = httpx.Client(verify=False)
        
        return ChatOpenAI(
            model=config.get('model', 'gpt-4'),
            temperature=config.get('temperature', 0.7),
            max_tokens=config.get('max_tokens', 2000),
            api_key=api_key,
            http_client=client
        )
    
    def _create_anthropic(self, config: dict) -> BaseChatModel:
        """Create Anthropic chat model with SSL bypass"""
        from langchain_anthropic import ChatAnthropic
        
        api_key = os.getenv(config.get('api_key_env', 'ANTHROPIC_API_KEY'))
        if not api_key:
            raise ValueError("Anthropic API key not found")
        
        client = httpx.Client(verify=False)
        
        return ChatAnthropic(
            model=config.get('model', 'claude-3-sonnet-20240229'),
            temperature=config.get('temperature', 0.7),
            max_tokens=config.get('max_tokens', 2000),
            api_key=api_key,
            http_client=client
        )
    
    def _create_huggingface(self, config: dict) -> BaseChatModel:
        """Create HuggingFace chat model with SSL bypass"""
        from langchain_openai import ChatOpenAI
        
        api_token = config.get('api_key_env')
        
        if api_token and api_token.startswith('hf_'):
            pass
        elif api_token:
            api_token = os.getenv(api_token)
        
        if not api_token:
            api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN') or os.getenv('HF_TOKEN')
        
        if not api_token:
            raise ValueError("HuggingFace API token not found")
        
        client = httpx.Client(verify=False)
        
        return ChatOpenAI(
            model=config.get('model', 'meta-llama/Llama-3.3-70B-Instruct'),
            base_url="https://router.huggingface.co/v1",
            api_key=api_token,
            temperature=config.get('temperature', 0.7),
            max_tokens=config.get('max_tokens', 512),
            http_client=client
        )
    
    def _create_ollama(self, config: dict) -> BaseChatModel:
        """Create Ollama chat model"""
        from langchain_ollama import ChatOllama
        
        return ChatOllama(
            model=config.get('model', 'gemma3:4b'),
            temperature=config.get('temperature', 0.3),
            base_url=config.get('base_url', 'http://localhost:11434')
        )
    
    def _initialize_embeddings(self):
        """Initialize embedding model with SSL bypass"""
        default_provider = self.config.get('default_embedding_provider', 'sentence_transformers')
        embed_config = self.embeddings_config.get(default_provider, {})
        
        try:
            if default_provider == 'azure_embedding':
                return self._create_azure_embedding(embed_config)
            elif default_provider == 'openai':
                return self._create_openai_embedding(embed_config)
            elif default_provider == 'sentence_transformers':
                return self._create_sentence_transformers_embedding(embed_config)
            elif default_provider == 'huggingface':
                return self._create_huggingface_embedding(embed_config)
            elif default_provider == 'ollama':
                return self._create_ollama_embedding(embed_config)
        except Exception as e:
            print(f"[ERROR] Failed to initialize embeddings with {default_provider}: {e}")
            print("[WARNING] Attempting fallback to Ollama embeddings...")
            try:
                return self._create_ollama_embedding({})
            except:
                print("[WARNING] Using fallback embeddings")
                return self._create_sentence_transformers_embedding({})
    
    def _create_azure_embedding(self, config: dict):
        """Create Azure embedding model with SSL bypass"""
        from langchain_openai import OpenAIEmbeddings
        
        api_key = config.get('api_key_env', 'AZURE_API_KEY')
        if not api_key:
            raise ValueError("Azure API key not found")
        
        client = httpx.Client(verify=False)
        
        return OpenAIEmbeddings(
            base_url=config.get('api_endpoint'),
            api_key=api_key,
            model=config.get('model', 'azure/genailab-maas-text-embedding-3-large'),
            http_client=client,
            tiktoken_enabled=True,
            tiktoken_model_name="text-embedding-3-large"
        )
    
    def _create_openai_embedding(self, config: dict):
        """Create OpenAI embedding model with SSL bypass"""
        from langchain_openai import OpenAIEmbeddings
        
        api_key = os.getenv(config.get('api_key_env', 'OPENAI_API_KEY'))
        
        client = httpx.Client(verify=False)
        
        return OpenAIEmbeddings(
            model=config.get('model', 'text-embedding-3-small'),
            api_key=api_key,
            http_client=client
        )
    
    def _create_sentence_transformers_embedding(self, config: dict):
        """Create Sentence Transformers embedding model"""
        from langchain_huggingface import HuggingFaceEmbeddings
        
        cache_folder = os.path.expanduser("~/.cache/huggingface/hub")
        device = config.get('device', 'cpu')
        
        try:
            import torch
            if device == 'cuda' and torch.cuda.is_available():
                print(f"[LLMService] Using GPU for embeddings (CUDA available)")
            else:
                device = 'cpu'
        except:
            device = 'cpu'
        
        return HuggingFaceEmbeddings(
            model_name=config.get('model', 'all-MiniLM-L6-v2'),
            cache_folder=cache_folder,
            model_kwargs={"device": device}
        )
    
    def _create_huggingface_embedding(self, config: dict):
        """Create HuggingFace embedding model"""
        from langchain_huggingface import HuggingFaceEmbeddings
        
        cache_folder = os.path.expanduser("~/.cache/huggingface/hub")
        
        return HuggingFaceEmbeddings(
            model_name=config.get('model', 'sentence-transformers/all-mpnet-base-v2'),
            cache_folder=cache_folder
        )
    
    def _create_ollama_embedding(self, config: dict):
        """Create Ollama embedding model"""
        from langchain_ollama import OllamaEmbeddings
        
        return OllamaEmbeddings(
            model=config.get('model', 'nomic-embed-text'),
            base_url=config.get('base_url', 'http://localhost:11434')
        )
    
    def get_model(self) -> BaseChatModel:
        """Get the initialized LLM model"""
        return self.llm
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate text response from prompt
        
        Args:
            prompt: Input prompt text
            
        Returns:
            Generated text response
        """
        response = self.llm.invoke(prompt)
        return response.content
    
    def generate_json(self, prompt: str) -> dict:
        """
        Generate structured JSON response
        
        Args:
            prompt: Input prompt (should request JSON output)
            
        Returns:
            Parsed JSON dictionary
        """
        response = self.llm.invoke(prompt)
        content = response.content
        
        # Extract JSON from markdown code blocks if present
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse JSON: {e}")
            print(f"Content: {content}")
            return {}
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for single text
        
        Args:
            text: Input text string
            
        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(text)
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken (cached) or fallback estimation
        
        Args:
            text: Input text
            
        Returns:
            Token count
        """
        if TIKTOKEN_AVAILABLE:
            return self._count_tokens_tiktoken(text)
        else:
            return len(text) // 4
    
    @staticmethod
    @lru_cache(maxsize=2048)
    def _count_tokens_tiktoken(text: str) -> int:
        """
        Count tokens using tiktoken with LRU cache
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Token count
        """
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            print(f"[WARNING] Tiktoken fallback: {e}")
            return len(text) // 4
    
    def estimate_cost(self, text: str, provider: str = None) -> Dict[str, float]:
        """
        Estimate token cost for text input
        
        Args:
            text: Input text
            provider: LLM provider (uses default if not specified)
            
        Returns:
            Cost breakdown dict
        """
        provider = provider or self.provider
        token_count = self.count_tokens(text)
        
        pricing = {
            'azure': {'default': {'input': 0.03, 'output': 0.06}},
            'openai': {'gpt-4': {'input': 0.03, 'output': 0.06}, 
                       'gpt-4-turbo': {'input': 0.01, 'output': 0.03}},
            'anthropic': {'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
                         'claude-3-opus': {'input': 0.015, 'output': 0.075}},
            'ollama': {'default': {'input': 0.0, 'output': 0.0}},
            'huggingface': {'default': {'input': 0.0, 'output': 0.0}}
        }
        
        provider_pricing = pricing.get(provider, {'default': {'input': 0.0, 'output': 0.0}})
        model_key = next(iter(provider_pricing.keys())) if provider_pricing else 'default'
        rates = provider_pricing.get(model_key, {'input': 0.0, 'output': 0.0})
        
        input_cost = (token_count / 1_000_000) * rates['input']
        output_estimate = (token_count * 1.5 / 1_000_000) * rates['output']
        
        return {
            'input_tokens': token_count,
            'input_cost_usd': round(input_cost, 6),
            'output_estimate_cost_usd': round(output_estimate, 6),
            'total_estimate_usd': round(input_cost + output_estimate, 6),
            'provider': provider
        }
