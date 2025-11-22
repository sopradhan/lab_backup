"""PromptModifyingAgent - Dynamic prompt orchestration and optimization"""
import json
import logging
from deepagents import create_deep_agent
from typing import Tuple

logger = logging.getLogger(__name__)


class PromptModifyingAgent:
    """Analyzes and optimizes prompts for downstream agents using meta-prompting"""
    
    def __init__(self, services: dict, config: dict = None):
        """Initialize the Prompt Modifying Agent
        
        Args:
            services: Dict containing 'llm' service with get_model() method
            config: Optional configuration dict
        """
        self.services = services
        self.name = config.get('name', 'PromptModifyingAgent') if config else 'PromptModifyingAgent'
        self.llm = services.get('llm')
        
        # Meta-role system prompt for orchestration specialist
        self.meta_system_prompt = (
            "You are a Prompt Orchestration Specialist. Analyze the original system prompt, "
            "user query, and technical metadata. Output ONLY a refined system prompt and query "
            "separated by '|'. Format: '<OPTIMIZED_SYSTEM_PROMPT>|<REFINED_QUERY>'. "
            "Make prompts specific, actionable, and metadata-aware."
        )
        
        # Create agent with LLM only (no external tools needed)
        try:
            self.agent = create_deep_agent(
                tools=[],
                system_prompt=self.meta_system_prompt,
                model=self.llm.get_model() if self.llm else None
            )
        except Exception as e:
            logger.error(f"Failed to create PromptModifyingAgent: {e}")
            raise
    
    def generate_optimized_prompt(
        self,
        original_system_prompt: str,
        original_query: str,
        metadata: dict = None
    ) -> Tuple[str, str]:
        """Generate optimized system prompt and query using meta-prompting
        
        Args:
            original_system_prompt: Base system prompt for downstream agent
            original_query: Original user query/incident data
            metadata: Additional context (severity, resource type, environment, etc.)
            
        Returns:
            Tuple of (optimized_system_prompt, refined_query)
        """
        try:
            # Build meta-input for prompt optimization
            meta_input = self._build_meta_input(
                original_system_prompt,
                original_query,
                metadata
            )
            
            # Call LLM directly using LLMService instead of agent.run()
            logger.debug(f"Optimizing prompt with metadata: {metadata}")
            combined_output = self.llm.generate_response(meta_input)
            
            # Parse output using pipe separator
            try:
                parts = combined_output.split('|', 1)
                if len(parts) == 2:
                    optimized_prompt = parts[0].strip()
                    refined_query = parts[1].strip()
                    logger.info(f"✓ Prompt optimized ({len(optimized_prompt)} chars prompt, {len(refined_query)} chars query)")
                    return optimized_prompt, refined_query
                else:
                    logger.warning("PMA output missing separator, using fallback")
                    return original_system_prompt, original_query
            except Exception as e:
                logger.warning(f"Failed to parse PMA output: {e}, using fallback")
                return original_system_prompt, original_query
                
        except Exception as e:
            logger.error(f"Error in generate_optimized_prompt: {e}", exc_info=True)
            return original_system_prompt, original_query
    
    def _build_meta_input(
        self,
        original_system_prompt: str,
        original_query: str,
        metadata: dict = None
    ) -> str:
        """Build structured input for the meta-agent
        
        Args:
            original_system_prompt: Base system prompt
            original_query: Original query
            metadata: Additional context
            
        Returns:
            Formatted string for meta-agent processing
        """
        meta_input = f"""# Original System Prompt
{original_system_prompt}

# User Query & Metadata
{original_query}"""
        
        if metadata:
            meta_input += f"""

# Context Metadata
{json.dumps(metadata, indent=2)}"""
        
        meta_input += """

---
INSTRUCTIONS:
1. Enhance the system prompt with specific context from metadata
2. Extract core action from the query as refined query
3. Output format: '<MODIFIED_SYSTEM_PROMPT>|<REFINED_QUERY>'
"""
        return meta_input
