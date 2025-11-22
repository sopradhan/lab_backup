"""
Configuration Loader - Optimized with lazy loading, caching, and SSL bypass
"""
import os
import ssl
import json
import urllib3
import requests
from pathlib import Path
from typing import Dict, Any, Optional

# ==============================
# AGGRESSIVE SSL BYPASS
# ==============================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

original_request = requests.Session.request
def patched_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return original_request(self, method, url, **kwargs)
requests.Session.request = patched_request

# Disable PostHog SSL
os.environ['POSTHOG_DISABLE_GZIP'] = 'true'
os.environ['POSTHOG_DEBUG'] = 'false'

# ==============================
# CONFIG LOADER
# ==============================
class ConfigLoader:
    """Lazy-loading config with JSON and caching"""
    
    _cache: Dict[str, Any] = {}
    _config_dir: Optional[str] = None
    
    @classmethod
    def set_config_dir(cls, config_dir: str) -> None:
        """Set base config directory"""
        cls._config_dir = config_dir
        cls._cache.clear()
    
    @classmethod
    def _load_json_file(cls, filename: str) -> Dict[str, Any]:
        """Load single JSON file and cache result"""
        if filename in cls._cache:
            return cls._cache[filename]
        
        if not cls._config_dir:
            cls._config_dir = "config"  # Default to 'config' directory
        
        file_path = Path(cls._config_dir) / filename
        if not file_path.exists():
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            cls._cache[filename] = config
            return config
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load {filename}: {e}")
            return {}
    
    @classmethod
    def get_system_prompt(cls, agent_name: str) -> str:
        """Get specific agent system prompt"""
        config = cls._load_json_file('prompts_config.json')
        prompt = config.get(agent_name, {}).get('system_prompt', '')
        return prompt.strip() if prompt else ''
    
    @classmethod
    def get_llm_config(cls, llm_name: Optional[str] = None) -> Dict[str, Any]:
        """Get LLM configuration"""
        config = cls._load_json_file('llm_config.json')
        if llm_name:
            return config.get(llm_name, {})
        return config
    
    @classmethod
    def get_default_llm_provider(cls) -> Dict[str, Any]:
        """Get default LLM provider"""
        config = cls._load_json_file('llm_config.json')
        default_name = config.get("default_provider", "azure")
        provider = config.get("llm_providers", {}).get(default_name, {})
        provider["name"] = default_name
        return provider
    
    @classmethod
    def get_default_embedding_provider(cls) -> Dict[str, Any]:
        """Get default embedding provider"""
        config = cls._load_json_file('llm_config.json')
        default_name = config.get("default_embedding_provider", "azure_embedding")
        provider = config.get("embedding_providers", {}).get(default_name, {})
        provider["name"] = default_name
        return provider
    
    @classmethod
    def get_agent_config(cls, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get agent configuration"""
        config = cls._load_json_file('agent_config.json')
        if agent_name:
            return config.get(agent_name, {})
        return config
    
    @classmethod
    def get_rbac_config(cls) -> Dict[str, Any]:
        """Get RBAC configuration"""
        return cls._load_json_file('rbac_config.json')
    
    @classmethod
    def get_data_sources_config(cls) -> Dict[str, Any]:
        """Get data sources configuration"""
        return cls._load_json_file('data_sources.json')
    
    @classmethod
    def get_system_config(cls) -> Dict[str, Any]:
        """Get system configuration"""
        return cls._load_json_file('system_config.json')
    
    @classmethod
    def get_prompt_template(cls, category: str, template_name: str) -> str:
        """Get specific prompt template"""
        config = cls._load_json_file('prompts_config.json')
        template = config.get(category, {}).get(template_name, {}).get('template', '')
        return template.strip() if template else ''
    
    @classmethod
    def get_rbac_namespace(cls, namespace_name: str) -> Dict[str, Any]:
        """Get specific RBAC namespace config"""
        config = cls._load_json_file('rbac_config.json')
        return config.get('namespaces', {}).get(namespace_name, {})
    
    @classmethod
    def get_database_config(cls) -> Dict[str, Any]:
        """Get database configuration from LLM config"""
        config = cls._load_json_file('llm_config.json')
        return config.get('database', {})
    
    @classmethod
    def get_cache_config(cls) -> Dict[str, Any]:
        """Get cache configuration from LLM config"""
        config = cls._load_json_file('llm_config.json')
        return config.get('cache', {})
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached configs"""
        cls._cache.clear()


# ==============================
# USAGE HELPER
# ==============================
def load_all_configs(config_dir: str = "config") -> Dict[str, Dict]:
    """
    Load all configuration files (backward compatibility)
    """
    ConfigLoader.set_config_dir(config_dir)
    
    configs = {
        'agent': ConfigLoader.get_agent_config(),
        'llm': ConfigLoader.get_llm_config(),
        'rbac': ConfigLoader.get_rbac_config(),
        'system': ConfigLoader.get_system_config(),
        'data_sources': ConfigLoader.get_data_sources_config(),
        'prompts': ConfigLoader._load_json_file('prompts_config.json'),
        'database': ConfigLoader.get_database_config(),
        'cache': ConfigLoader.get_cache_config()
    }
    
    return configs


# ==============================
# INITIALIZATION
# ==============================
# Auto-initialize with default config directory
# ConfigLoader.set_config_dir("config")
