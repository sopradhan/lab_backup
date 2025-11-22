"""MasterOrchestrator - Master agent coordinating 3 sub-agents (Ingestion, Retrieval, Healing)"""
import json
import sqlite3
import time
from deepagents import create_deep_agent
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from ..tools.config.loader import ConfigLoader
from ..config.env_config import EnvConfig
from .ingestion_agent import IngestionAgent
from .retrieval_agent import RetrievalAgent
from .healing_agent import HealingAgent
from .prompt_modifying_agent import PromptModifyingAgent
from ...database.models import AgentOperationModel


class MasterOrchestrator:
    """Master agent spawning and managing sub-agents"""
    
    def __init__(self, config_dir: str = None):
        self.name = "MasterOrchestrator"
        if config_dir is None:
            config_dir = EnvConfig.get_rag_config_path()
            print(config_dir)
        self.config_dir = config_dir
        ConfigLoader.set_config_dir(config_dir)
        self.db_path = EnvConfig.get_db_path()
        
        # Initialize LLM service
        from ..tools.services.llm_service import LLMService
        self.llm_service = LLMService(ConfigLoader.get_llm_config())
        
        # Initialize Vector DB service
        from ..tools.services.vectordb_service import VectorDBService
        self.vectordb_service = VectorDBService(EnvConfig.get_chroma_db_path())
        
        # Initialize database connection
        db_path = EnvConfig.get_db_path()
        self.db_conn = sqlite3.connect(db_path)
        self.db_conn.row_factory = sqlite3.Row
        
        # Create services dict
        self.services = {
            'llm': self.llm_service,
            'vectordb': self.vectordb_service,
            'db': self.db_conn,
            'rbac_config': ConfigLoader.get_rbac_config()
        }

        
        # Initialize sub-agents with master reference
        agent_config = ConfigLoader.get_agent_config()
        self.ingestion_agent = IngestionAgent(self.services, agent_config.get('ingestion_agent', {}), self)
        self.retrieval_agent = RetrievalAgent(self.services, agent_config.get('retrieval_agent', {}), self)
        self.healing_agent = HealingAgent(self.services, agent_config.get('healing_agent', {}), self)
        self.prompt_agent = PromptModifyingAgent(self.services, agent_config.get('prompt_modifying_agent', {}))
        
        # Store agent config for dynamic spawning
        self.agent_config = agent_config
        
        # Create master tools and agent
        self.tools = self._create_master_tools()
        system_prompt = ConfigLoader.get_system_prompt('orchestrator') or \
            "Master orchestrator agent. Spawn specialized sub-agents based on user tasks. " \
            "Delegate work via tools. Route ingestion, retrieval, healing tasks. Track all operations."
        
        print(system_prompt,"sys prompt in master orch")
        self.agent = create_deep_agent(
            tools=self.tools,
            system_prompt=system_prompt,
            model=self.llm_service.get_model()
        )
    
    def _create_master_tools(self):
        """Create master orchestrator tools"""
        from ..tools.ingestion_tools import record_agent_spawn_tool, record_agent_operation_tool, record_agent_memory_tool
        
        master = self
        
        class SpawnAgentInput(BaseModel):
            agent_type: str = Field(description="ingestion|retrieval|healing")
            task: str = Field(description="Task description")
            parameters: str = Field(default="{}", description="JSON parameters")
        
        class ExecuteAgentTaskInput(BaseModel):
            agent_name: str = Field(description="IngestionAgent|RetrievalAgent|HealingAgent")
            task: str = Field(description="Task to execute")
            data: str = Field(default="{}", description="JSON input data")
        
        class MonitorAgentInput(BaseModel):
            agent_name: str = Field(description="Agent name to monitor")
            metric: str = Field(description="Metric: status|operations|memory")
        
        def spawn_agent(agent_type: str, task: str, parameters: str = "{}") -> str:
            try:
                record_agent_spawn_tool.func(
                    parent_agent=master.name,
                    child_agent=f"{agent_type.capitalize()}Agent",
                    task_description=task,
                    status="spawned"
                )
                return json.dumps({"success": True, "agent": agent_type, "task": task})
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})
        
        def execute_agent_task(agent_name: str, task: str, data: str = "{}") -> str:
            try:
                data_obj = json.loads(data) if isinstance(data, str) else data
                result = {}
                
                if 'ingestion' in agent_name.lower():
                    if task == 'ingest_document':
                        result = master.ingestion_agent.ingest_document(data_obj.get('file_path', ''))
                    elif task == 'ingest_knowledge_base':
                        result = master.run_ingestion()
                else:
                    if 'retrieval' in agent_name.lower():
                        result = master.retrieval_agent.process_query(
                            data_obj.get('query', ''),
                            data_obj.get('user_id', 'system')
                        )
                    elif 'healing' in agent_name.lower():
                        result = master.healing_agent.analyze_health()
                
                record_agent_operation_tool.func(
                    agent_name=agent_name,
                    operation_type=task,
                    status="success" if result.get('success') else "failed",
                    doc_id=data_obj.get('doc_id', 'N/A'),
                    chunks_count=result.get('chunks_saved', 0)
                )
                
                record_agent_memory_tool.func(
                    agent_name=agent_name,
                    memory_key=f"task_{task}",
                    memory_value=json.dumps(result),
                    memory_type="result"
                )
                
                return json.dumps(result)
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})
        
        def monitor_agent(agent_name: str, metric: str) -> str:
            try:
                if metric == 'status':
                    return json.dumps({"status": "ready", "agent": agent_name})
                elif metric == 'operations':
                    # Use model layer instead of hardcoded SQL
                    agent_model = AgentOperationModel(master.db_conn)
                    ops = agent_model.get_operations_by_agent(agent_name, limit=10000)
                    count = len(ops)
                    return json.dumps({"operations_count": count, "agent": agent_name})
                elif metric == 'memory':
                    # Count agent_memory records for agent
                    agent_model = AgentOperationModel(master.db_conn)
                    memory_records = agent_model.raw_execute(
                        "SELECT * FROM agent_memory WHERE agent_name = ?",
                        (agent_name,)
                    )
                    count = len(memory_records)
                    return json.dumps({"memory_entries": count, "agent": agent_name})
                return json.dumps({"success": False, "error": f"Unknown metric: {metric}"})
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})
        
        return [
            StructuredTool.from_function(
                func=spawn_agent,
                name="spawn_agent",
                description="Spawn specialized sub-agent",
                    args_schema=SpawnAgentInput
            ),
            StructuredTool.from_function(
                func=execute_agent_task,
                name="execute_agent_task",
                description="Execute task on spawned agent",
                args_schema=ExecuteAgentTaskInput
            ),
            StructuredTool.from_function(
                func=monitor_agent,
                name="monitor_agent",
                description="Monitor agent status and metrics",
                args_schema=MonitorAgentInput
            ),
        ]
    
    def spawn_agent(self, agent_type: str, caller_agent: str = None):
        """
        Dynamically spawn an agent instance with optional caller context.
        
        Args:
            agent_type (str): 'ingestion', 'retrieval', or 'healing'
            caller_agent (str): Name of calling agent ('IngestionAgent', 'RetrievalAgent', etc)
                                Used to customize HealingAgent behavior
        
        Returns:
            Agent instance with appropriate configuration
        """
        agent_type = agent_type.lower()
        
        if agent_type == 'ingestion':
            return IngestionAgent(self.services, self.agent_config.get('ingestion_agent', {}), self)
        elif agent_type == 'retrieval':
            return RetrievalAgent(self.services, self.agent_config.get('retrieval_agent', {}), self)
        elif agent_type == 'healing':
            # Create HealingAgent with caller context
            return HealingAgent(
                self.services, 
                self.agent_config.get('healing_agent', {}), 
                self,
                caller_agent=caller_agent  # Pass caller context
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def get_test_user(self) -> dict:
        """Get test user from RBAC config"""
        rbac_config = ConfigLoader.get_rbac_config()
        role_mappings = rbac_config.get('role_mappings', {})
        
        if not role_mappings:
            return {'user_id': 'test_user_default', 'cdr_code': '111', 'access_level': 1}
        
        for cdr_code, role_config in role_mappings.items():
            return {
                'user_id': f"test_user_{role_config.get('company_id')}",
                'cdr_code': cdr_code,
                'access_level': role_config.get('access_level', 1)
            }
    
    def orchestrate(self, user_request: str) -> dict:
        """Route request to appropriate agent"""
        try:
            result = self.agent.invoke({"messages": [{"role": "user", "content": user_request}]})
            messages = result.get('messages', [])
            response = messages[-1].get('content', '') if messages else 'No response'
            return {"success": True, "response": response}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def process_query(self, query: str, user_id: str = None) -> dict:
        """Process query through retrieval agent"""
        if not user_id:
            user_id = self.get_test_user()['user_id']
        return self.retrieval_agent.process_query(query, user_id)
    
    def ask_question(self, query: str, user_id: str = None, enable_healing: bool = True) -> dict:
        """
        Process user question through orchestrator - spawns retrieval and optionally healing agents
        Returns: { query, answer, token_usage, tags, agents_used, execution_ms }
        """
        start_time = time.time()
        
        if not user_id:
            user_id = self.get_test_user()['user_id']
        
        try:
            # Step 0: Optimize prompt using PromptModifyingAgent (PMA)
            base_system_prompt = "You are an incident management assistant specializing in Azure cloud infrastructure."
            metadata = self._extract_metadata_from_query(query)
            
            try:
                optimized_prompt, refined_query = self.prompt_agent.generate_optimized_prompt(
                    base_system_prompt, query, metadata
                )
                print(f"[Orchestrator] PromptModifyingAgent optimized prompt")
                agents_used = ["PromptModifyingAgent"]
            except Exception as e:
                print(f"[Orchestrator] PMA failed, using original query: {e}")
                refined_query = query
                optimized_prompt = base_system_prompt
                agents_used = []
            
            # Step 1: Spawn retrieval agent to get context
            print(f"[Orchestrator] Spawning RetrievalAgent...")
            retrieval_result = self.retrieval_agent.process_query(refined_query, user_id)
            
            if not retrieval_result.get('success'):
                return {
                    "success": False,
                    "query": query,
                    "error": retrieval_result.get('error', 'Retrieval failed'),
                    "execution_ms": int((time.time() - start_time) * 1000)
                }
            
            # Get RAG context and metadata
            rag_results = retrieval_result.get('results', [])
            query_complexity = retrieval_result.get('query_complexity', 'unknown')
            namespace = retrieval_result.get('namespace', 'general')
            token_cost_retrieval = retrieval_result.get('token_cost', {})
            
            # Step 2: Generate LLM answer using optimized prompt
            print(f"[Orchestrator] Generating LLM response with {len(rag_results)} context chunks")
            
            # Build RAG prompt with optimized system prompt
            rag_context = "".join([result.get('content', '') + "\n\n" for result in rag_results])
            
            rag_prompt = f"""{optimized_prompt}

CONTEXT FROM KNOWLEDGE BASE:
{rag_context if rag_context.strip() else 'No specific context found.'}

REFINED QUERY: {refined_query}

Provide a clear, actionable answer."""
            
            # Get LLM response
            llm_answer = self.llm_service.generate_response(rag_prompt)
            
            # Estimate token cost for LLM response
            response_tokens = len(llm_answer.split())
            total_retrieval_tokens = token_cost_retrieval.get('total', 0)
            total_tokens = total_retrieval_tokens + response_tokens
            
            # Step 3: Spawn healing agent for autonomous answer optimization
            healing_analysis = None
            agents_used.append("RetrievalAgent")
            optimization_result = None
            
            if enable_healing:
                print(f"[Orchestrator] Spawning HealingAgent for autonomous answer optimization")
                try:
                    # Healing agent autonomously optimizes the answer
                    optimization_result = self.healing_agent.optimize_answer(
                        llm_answer, 
                        refined_query, 
                        token_limit=250
                    )
                    
                    if optimization_result.get('success'):
                        # Use optimized answer
                        llm_answer = optimization_result.get('optimized_answer', llm_answer)
                        response_tokens = optimization_result.get('optimized_tokens', response_tokens)
                        total_tokens = token_cost_retrieval.get('total', 0) + response_tokens
                        
                        healing_analysis = {
                            "status": "optimization_applied",
                            "original_tokens": optimization_result.get('original_tokens'),
                            "optimized_tokens": optimization_result.get('optimized_tokens'),
                            "token_reduction": optimization_result.get('token_reduction'),
                            "reduction_percentage": optimization_result.get('reduction_percentage')
                        }
                        
                        print(f"[Orchestrator] HealingAgent reduced tokens by {optimization_result.get('reduction_percentage')}%")
                        agents_used.append("HealingAgent")
                    
                    # System health check
                    health_check = self.healing_agent.analyze_health()
                    if health_check.get('success') and healing_analysis:
                        healing_analysis['system_health'] = {
                            "status": health_check.get('status'),
                            "avg_quality": health_check.get('avg_quality'),
                            "total_chunks": health_check.get('total_chunks')
                        }
                    
                except Exception as e:
                    print(f"[Orchestrator] Healing agent optimization failed: {str(e)}")
            
            # Step 4: Extract tags/topics from answer
            tags = self._extract_tags_from_answer(llm_answer, query)
            
            # Step 5: Record operation
            try:
                from ..tools.ingestion_tools import record_agent_operation_tool, record_agent_memory_tool
                
                record_agent_operation_tool.func(
                    agent_name=self.name,
                    operation_type='ask_question',
                    status='success',
                    doc_id=query[:50],
                    chunks_count=len(rag_results)
                )
                
                record_agent_memory_tool.func(
                    agent_name=self.name,
                    memory_key=f'question_{user_id}',
                    memory_value=json.dumps({
                        'query': query,
                        'complexity': query_complexity,
                        'agents_used': agents_used,
                        'tags': tags
                    }),
                    memory_type='user_question'
                )
            except:
                pass
            
            # Build final response
            exec_ms = int((time.time() - start_time) * 1000)
            
            response = {
                "success": True,
                "query": query,
                "answer": llm_answer,
                "token_usage": {
                    "retrieval": total_retrieval_tokens,
                    "response": response_tokens,
                    "total": total_tokens
                },
                "tags": tags,
                "metadata": {
                    "query_complexity": query_complexity,
                    "namespace": namespace,
                    "context_chunks": len(rag_results),
                    "agents_spawned": agents_used,
                    "healing_analysis": healing_analysis
                },
                "execution_ms": exec_ms
            }
            
            print(f"[Orchestrator] Completed in {exec_ms}ms using {', '.join(agents_used)}")
            return response
            
        except Exception as e:
            print(f"[Orchestrator] Error: {str(e)}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "execution_ms": int((time.time() - start_time) * 1000)
            }
    
    def _extract_metadata_from_query(self, query: str) -> dict:
        """Extract metadata (severity, resource type, environment) from query"""
        metadata = {}
        
        # Severity mapping
        severities = {'s1': 'Critical', 's2': 'High', 's3': 'Medium', 's4': 'Low'}
        for key, val in severities.items():
            if key in query.lower():
                metadata['severity'] = val
                break
        
        # Resource types
        resources = {
            'storageaccounts': 'Microsoft.Storage/storageAccounts',
            'database': 'Microsoft.Sql/servers',
            'appservice': 'Microsoft.Web/sites',
            'vm': 'Microsoft.Compute/virtualMachines',
            'keyvault': 'Microsoft.KeyVault/vaults'
        }
        for key, val in resources.items():
            if key in query.lower():
                metadata['resource_type'] = val
                break
        
        # Environment
        if 'prod' in query.lower():
            metadata['environment'] = 'Production'
        elif 'dev' in query.lower():
            metadata['environment'] = 'Development'
        elif 'test' in query.lower():
            metadata['environment'] = 'Testing'
        
        return metadata
    
    def _extract_tags_from_answer(self, answer: str, query: str) -> list:
        """Extract relevant tags/topics from answer and query"""
        tags = []
        
        # Keywords mapping
        keywords = {
            "network": ["network", "connection", "connectivity", "dns", "ip"],
            "database": ["database", "db", "sql", "query", "transaction", "connection pool"],
            "api": ["api", "gateway", "endpoint", "request", "response", "http"],
            "server": ["server", "instance", "vm", "host", "cpu", "memory", "disk"],
            "failover": ["failover", "failback", "redundancy", "availability", "high availability"],
            "monitoring": ["monitoring", "alert", "metrics", "logging", "trace"],
            "performance": ["performance", "latency", "throughput", "optimization", "slow"],
            "security": ["security", "authentication", "authorization", "rbac", "permission"],
            "configuration": ["configuration", "config", "setting", "parameter", "tuning"],
            "incident": ["incident", "issue", "problem", "error", "failure"]
        }
        
        combined_text = (answer + " " + query).lower()
        
        for tag, keywords_list in keywords.items():
            for keyword in keywords_list:
                if keyword in combined_text:
                    if tag not in tags:
                        tags.append(tag)
                    break
        
        return tags if tags else ["general"]
    
    def run_system_check(self) -> dict:
        """Run system health check"""
        return self.healing_agent.analyze_health()
    
    def run_ingestion(self) -> dict:
        """Ingest knowledge_base table into vector database"""
        try:
            data_sources_cfg = ConfigLoader.get_data_sources_config()
            sqlite_cfg = data_sources_cfg.get('data_sources', {}).get('sqlite', {})
            
            if not sqlite_cfg.get('enabled', False):
                return {"success": False, "error": "SQLite data source not enabled"}
            
            db_path = sqlite_cfg.get('connection_string_env', EnvConfig.get_db_path())
            source_conn = sqlite3.connect(db_path)
            source_conn.row_factory = sqlite3.Row
            
            # Get table config
            table_config = None
            tables = sqlite_cfg.get('ingestion_modes', {}).get('table_based', {}).get('tables_to_ingest', [])
            
            for t in tables:
                if t.get('name') == 'knowledge_base' and t.get('enabled', True):
                    table_config = t
                    break
            
            if not table_config:
                return {"success": False, "error": "knowledge_base table not configured"}
            
            text_columns = table_config.get('text_columns', [])
            metadata_columns = table_config.get('metadata_columns', [])
            
            # Fetch and ingest records using model layer
            from ...database.models import KnowledgeBaseModel
            kb_model = KnowledgeBaseModel(source_conn)
            rows = kb_model.all()
            
            ingested_count = 0
            for i, row in enumerate(rows, 1):
                try:
                    row_dict = dict(row)
                    text_content = " ".join([
                        str(row_dict.get(col, ""))
                        for col in text_columns
                        if col in row_dict and row_dict.get(col)
                    ])
                    
                    if not text_content.strip():
                        continue
                    
                    doc_id = f"kb_{row_dict.get('id', i)}"
                    result = self.ingestion_agent.ingest_document_text(text_content, doc_id)
                    
                    if result.get('success'):
                        ingested_count += 1
                except Exception as e:
                    continue
            
            source_conn.close()
            
            return {
                "success": ingested_count > 0,
                "total_records": len(rows),
                "ingested_count": ingested_count,
                "vector_db_status": {"documents_ingested": ingested_count}
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
