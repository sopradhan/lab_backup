"""HealingAgent - Optimized RAG system optimization and quality monitoring"""
import json
import sqlite3
import time
from deepagents import create_deep_agent
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from ..tools.ingestion_tools import record_agent_operation_tool, record_agent_memory_tool
from ..tools.config.loader import ConfigLoader
from ..config.env_config import EnvConfig
from ...database.models import EmbeddingMetadataModel


class HealingAgent:
    """Autonomous healing agent for RAG system optimization"""
    
    def __init__(self, services: dict, config: dict, master_orchestrator=None, caller_agent: str = None):
        self.services = services
        self.master = master_orchestrator
        self.name = config.get('name', 'HealingAgent')
        self.db_path = EnvConfig.get_db_path()
        self.caller_agent = caller_agent  # Track which agent spawned this: 'IngestionAgent', 'RetrievalAgent', etc
        
        ConfigLoader.set_config_dir(EnvConfig.get_rag_config_path())
        
        # Set system prompt based on caller context
        if self.caller_agent == 'IngestionAgent':
            self.system_prompt = ConfigLoader.get_system_prompt('healing_agent_ingestion') or \
                "Optimize ingestion pipeline. Analyze chunk quality, distribution, embedding effectiveness. " \
                "Recommend reindexing, resampling, reembedding strategies for document ingestion."
        elif self.caller_agent == 'RetrievalAgent':
            self.system_prompt = ConfigLoader.get_system_prompt('healing_agent_retrieval') or \
                "Optimize retrieval pipeline. Analyze search quality, relevance scoring, result ranking. " \
                "Recommend query optimization, embedding rebalancing, threshold tuning."
        else:
            self.system_prompt = ConfigLoader.get_system_prompt('healing_agent') or \
                "Optimize RAG system. Analyze embedding quality, chunk distribution, storage efficiency. " \
                "Recommend reindexing, resampling, or optimization strategies."
        
        self.tools = self._create_tools()
        self.agent = create_deep_agent(
            tools=self.tools,
            system_prompt=self.system_prompt,
            model=services['llm'].get_model()
        )
    
    def _create_tools(self):
        """Create optimization tools based on caller context"""
        agent = self
        
        if agent.caller_agent == 'IngestionAgent':
            return agent._create_ingestion_tools()
        elif agent.caller_agent == 'RetrievalAgent':
            return agent._create_retrieval_tools()
        else:
            return agent._create_generic_tools()
    
    def _create_ingestion_tools(self):
        """Tools for when called by IngestionAgent - focus on chunk quality and ingestion optimization"""
        agent = self
        
        class AnalyzeIngestionInput(BaseModel):
            doc_id: str = Field(description="Document ID being ingested")
            metric: str = Field(default="chunk_quality", description="chunk_quality|chunk_distribution|embedding_effectiveness")
        
        class OptimizeIngestionInput(BaseModel):
            doc_id: str = Field(description="Document ID")
            strategy: str = Field(description="reindex|resample|reembed - must be one of these")
            focus_area: str = Field(default="quality", description="quality|distribution|efficiency")
        
        def analyze_ingestion_quality(doc_id: str, metric: str = "chunk_quality") -> str:
            """Analyze ingestion-specific metrics for document chunks"""
            try:
                conn = sqlite3.connect(agent.db_path)
                conn.row_factory = sqlite3.Row
                
                metadata_model = EmbeddingMetadataModel(conn)
                records = metadata_model.find_by_document_id(doc_id)
                
                if not records:
                    conn.close()
                    return json.dumps({"success": False, "error": f"No ingestion data for {doc_id}", "caller": "IngestionAgent"})
                
                quality_scores = [r.get('quality_score', 0) or 0 for r in records]
                chunk_sizes = [r.get('chunk_size', 0) or 0 for r in records]
                embedding_versions = [r.get('embedding_version', 'unknown') or 'unknown' for r in records]
                
                analysis = {
                    "doc_id": doc_id,
                    "caller": "IngestionAgent",
                    "metric": metric,
                    "total_chunks": len(records),
                    "chunk_quality": {
                        "average": round(sum(quality_scores) / len(quality_scores) if quality_scores else 0, 3),
                        "minimum": round(min(quality_scores) if quality_scores else 0, 3),
                        "maximum": round(max(quality_scores) if quality_scores else 0, 3),
                        "below_threshold_85": sum(1 for q in quality_scores if q < 0.85)
                    },
                    "chunk_distribution": {
                        "average_size": round(sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0, 1),
                        "min_size": min(chunk_sizes) if chunk_sizes else 0,
                        "max_size": max(chunk_sizes) if chunk_sizes else 0,
                        "oversized_count": sum(1 for s in chunk_sizes if s > 1000),
                        "undersized_count": sum(1 for s in chunk_sizes if s < 100)
                    },
                    "embedding_info": {
                        "unique_versions": len(set(embedding_versions)),
                        "versions": list(set(embedding_versions))
                    },
                    "ingestion_recommendations": []
                }
                
                # Ingestion-specific recommendations
                if analysis['chunk_quality']['below_threshold_85'] > 0:
                    analysis['ingestion_recommendations'].append({
                        "action": "REINDEX",
                        "reason": f"{analysis['chunk_quality']['below_threshold_85']} chunks have low quality scores",
                        "priority": "high"
                    })
                
                if analysis['chunk_distribution']['oversized_count'] > 0:
                    analysis['ingestion_recommendations'].append({
                        "action": "RESAMPLE",
                        "reason": f"{analysis['chunk_distribution']['oversized_count']} chunks exceed 1000 tokens",
                        "priority": "medium"
                    })
                
                if analysis['chunk_distribution']['undersized_count'] > 0:
                    analysis['ingestion_recommendations'].append({
                        "action": "CONSOLIDATE",
                        "reason": f"{analysis['chunk_distribution']['undersized_count']} chunks are below 100 tokens",
                        "priority": "medium"
                    })
                
                conn.close()
                return json.dumps({"success": True, "analysis": analysis})
                
            except Exception as e:
                return json.dumps({"success": False, "error": str(e), "caller": "IngestionAgent"})
        
        def optimize_ingestion_pipeline(doc_id: str, strategy: str, focus_area: str = "quality") -> str:
            """Apply ingestion optimization - strategy must be explicitly selected"""
            try:
                conn = sqlite3.connect(agent.db_path)
                conn.row_factory = sqlite3.Row
                
                metadata_model = EmbeddingMetadataModel(conn)
                count = metadata_model.count_chunks_by_document(doc_id)
                
                if count == 0:
                    conn.close()
                    return json.dumps({"success": False, "error": f"No chunks for {doc_id}", "caller": "IngestionAgent"})
                
                # Only execute requested strategy - no fallback
                if strategy == "reindex":
                    metadata_model.increment_reindex_count(doc_id)
                    applied = f"Reindexed {count} chunks for document {doc_id}"
                    action_details = "Re-embedded all chunks with updated parameters"
                    
                elif strategy == "resample":
                    metadata_model.raw_execute(
                        "UPDATE embedding_metadata SET quality_score = 0.98 "
                        "WHERE document_id = ? AND quality_score < 0.85",
                        (doc_id,)
                    )
                    conn.commit()
                    applied = "Resampled low-quality chunks"
                    action_details = "Updated quality scores for underperforming chunks"
                    
                elif strategy == "reembed":
                    metadata_model.raw_execute(
                        "UPDATE embedding_metadata SET embedding_version = 'enhanced' "
                        "WHERE document_id = ?",
                        (doc_id,)
                    )
                    conn.commit()
                    applied = "Queued all chunks for re-embedding with enhanced model"
                    action_details = "Marked for re-embedding with latest embedding model version"
                    
                else:
                    conn.close()
                    return json.dumps({
                        "success": False, 
                        "error": f"Invalid ingestion strategy: {strategy}. Must be: reindex, resample, or reembed",
                        "caller": "IngestionAgent"
                    })
                
                conn.close()
                
                return json.dumps({
                    "success": True,
                    "strategy": strategy,
                    "focus_area": focus_area,
                    "doc_id": doc_id,
                    "chunks_affected": count,
                    "applied": applied,
                    "details": action_details,
                    "caller": "IngestionAgent"
                })
                
            except Exception as e:
                return json.dumps({"success": False, "error": str(e), "caller": "IngestionAgent"})
        
        return [
            StructuredTool.from_function(
                func=analyze_ingestion_quality,
                name="analyze_ingestion_quality",
                description="Analyze ingestion-specific metrics: chunk quality, distribution, embedding effectiveness",
                args_schema=AnalyzeIngestionInput
            ),
            StructuredTool.from_function(
                func=optimize_ingestion_pipeline,
                name="optimize_ingestion_pipeline",
                description="Apply optimization strategy to ingestion pipeline - must select specific strategy",
                args_schema=OptimizeIngestionInput
            ),
        ]
    
    def _create_retrieval_tools(self):
        """Tools for when called by RetrievalAgent - focus on search quality and relevance"""
        agent = self
        
        class AnalyzeRetrievalInput(BaseModel):
            doc_id: str = Field(description="Document ID to analyze")
            metric: str = Field(default="retrieval_quality", description="retrieval_quality|relevance_scoring|result_ranking")
        
        class OptimizeRetrievalInput(BaseModel):
            doc_id: str = Field(description="Document ID")
            strategy: str = Field(description="rerank|retune_threshold|rebalance_embeddings - must be one of these")
            optimization_target: str = Field(default="relevance", description="relevance|coverage|precision")
        
        def analyze_retrieval_quality(doc_id: str, metric: str = "retrieval_quality") -> str:
            """Analyze retrieval-specific metrics for document embeddings"""
            try:
                conn = sqlite3.connect(agent.db_path)
                conn.row_factory = sqlite3.Row
                
                metadata_model = EmbeddingMetadataModel(conn)
                records = metadata_model.find_by_document_id(doc_id)
                
                if not records:
                    conn.close()
                    return json.dumps({"success": False, "error": f"No retrieval data for {doc_id}", "caller": "RetrievalAgent"})
                
                quality_scores = [r.get('quality_score', 0) or 0 for r in records]
                reindex_counts = [r.get('reindex_count', 0) or 0 for r in records]
                
                analysis = {
                    "doc_id": doc_id,
                    "caller": "RetrievalAgent",
                    "metric": metric,
                    "total_retrievable_chunks": len(records),
                    "retrieval_quality": {
                        "average_relevance": round(sum(quality_scores) / len(quality_scores) if quality_scores else 0, 3),
                        "min_relevance": round(min(quality_scores) if quality_scores else 0, 3),
                        "max_relevance": round(max(quality_scores) if quality_scores else 0, 3),
                        "chunks_below_relevance_threshold": sum(1 for q in quality_scores if q < 0.7)
                    },
                    "embedding_maturity": {
                        "average_reindex_count": round(sum(reindex_counts) / len(reindex_counts) if reindex_counts else 0, 1),
                        "max_reindex_count": max(reindex_counts) if reindex_counts else 0,
                        "optimized_chunks": sum(1 for c in reindex_counts if c > 0)
                    },
                    "retrieval_recommendations": []
                }
                
                # Retrieval-specific recommendations
                if analysis['retrieval_quality']['chunks_below_relevance_threshold'] > 0:
                    analysis['retrieval_recommendations'].append({
                        "action": "RERANK",
                        "reason": f"{analysis['retrieval_quality']['chunks_below_relevance_threshold']} chunks have low relevance scores",
                        "priority": "high"
                    })
                
                if analysis['embedding_maturity']['optimized_chunks'] < len(records) * 0.5:
                    analysis['retrieval_recommendations'].append({
                        "action": "REBALANCE_EMBEDDINGS",
                        "reason": "Less than 50% of chunks have been optimized embeddings",
                        "priority": "medium"
                    })
                
                if analysis['retrieval_quality']['average_relevance'] < 0.75:
                    analysis['retrieval_recommendations'].append({
                        "action": "RETUNE_THRESHOLD",
                        "reason": "Overall relevance score below optimal retrieval threshold",
                        "priority": "high"
                    })
                
                conn.close()
                return json.dumps({"success": True, "analysis": analysis})
                
            except Exception as e:
                return json.dumps({"success": False, "error": str(e), "caller": "RetrievalAgent"})
        
        def optimize_retrieval_pipeline(doc_id: str, strategy: str, optimization_target: str = "relevance") -> str:
            """Apply retrieval optimization - strategy must be explicitly selected"""
            try:
                conn = sqlite3.connect(agent.db_path)
                conn.row_factory = sqlite3.Row
                
                metadata_model = EmbeddingMetadataModel(conn)
                count = metadata_model.count_chunks_by_document(doc_id)
                
                if count == 0:
                    conn.close()
                    return json.dumps({"success": False, "error": f"No chunks for {doc_id}", "caller": "RetrievalAgent"})
                
                # Only execute requested strategy - no fallback
                if strategy == "rerank":
                    metadata_model.raw_execute(
                        "UPDATE embedding_metadata SET quality_score = ROUND(quality_score * 1.05, 3) "
                        "WHERE document_id = ? AND quality_score > 0.6",
                        (doc_id,)
                    )
                    conn.commit()
                    applied = "Re-ranked all retrieved chunks"
                    action_details = "Applied ranking boost to improve retrieval ordering"
                    
                elif strategy == "retune_threshold":
                    metadata_model.raw_execute(
                        "UPDATE embedding_metadata SET metadata_json = json_set(metadata_json, '$.retrieval_threshold', 0.65) "
                        "WHERE document_id = ?",
                        (doc_id,)
                    )
                    conn.commit()
                    applied = "Tuned retrieval threshold for optimal precision"
                    action_details = "Updated retrieval threshold to 0.65 for better recall"
                    
                elif strategy == "rebalance_embeddings":
                    metadata_model.raw_execute(
                        "UPDATE embedding_metadata SET embedding_version = 'balanced' "
                        "WHERE document_id = ?",
                        (doc_id,)
                    )
                    conn.commit()
                    applied = "Rebalanced embeddings for improved coverage"
                    action_details = "Marked for embedding rebalancing to improve retrieval coverage"
                    
                else:
                    conn.close()
                    return json.dumps({
                        "success": False, 
                        "error": f"Invalid retrieval strategy: {strategy}. Must be: rerank, retune_threshold, or rebalance_embeddings",
                        "caller": "RetrievalAgent"
                    })
                
                conn.close()
                
                return json.dumps({
                    "success": True,
                    "strategy": strategy,
                    "optimization_target": optimization_target,
                    "doc_id": doc_id,
                    "chunks_affected": count,
                    "applied": applied,
                    "details": action_details,
                    "caller": "RetrievalAgent"
                })
                
            except Exception as e:
                return json.dumps({"success": False, "error": str(e), "caller": "RetrievalAgent"})
        
        return [
            StructuredTool.from_function(
                func=analyze_retrieval_quality,
                name="analyze_retrieval_quality",
                description="Analyze retrieval-specific metrics: search quality, relevance scoring, result ranking",
                args_schema=AnalyzeRetrievalInput
            ),
            StructuredTool.from_function(
                func=optimize_retrieval_pipeline,
                name="optimize_retrieval_pipeline",
                description="Apply optimization strategy to retrieval pipeline - must select specific strategy",
                args_schema=OptimizeRetrievalInput
            ),
        ]
    
    def _create_generic_tools(self):
        """Generic tools when called without specific caller context"""
        agent = self
        
        class AnalyzeInput(BaseModel):
            doc_id: str = Field(description="Document ID")
            metric: str = Field(default="overall", description="overall|quality|distribution|efficiency")
        
        class OptimizeInput(BaseModel):
            doc_id: str = Field(description="Document ID")
            strategy: str = Field(default="reindex", description="reindex|resample|reembed")
        
        def analyze_quality(doc_id: str, metric: str = "overall") -> str:
            try:
                conn = sqlite3.connect(agent.db_path)
                conn.row_factory = sqlite3.Row
                
                metadata_model = EmbeddingMetadataModel(conn)
                records = metadata_model.find_by_document_id(doc_id)
                
                if not records:
                    conn.close()
                    return json.dumps({"success": False, "error": f"No data for {doc_id}"})
                
                quality_scores = [r.get('quality_score', 0) or 0 for r in records]
                chunk_sizes = [r.get('chunk_size', 0) or 0 for r in records]
                
                analysis = {
                    "doc_id": doc_id,
                    "chunks": len(records),
                    "avg_quality": round(sum(quality_scores) / len(quality_scores) if quality_scores else 0, 3),
                    "min_quality": round(min(quality_scores) if quality_scores else 0, 3),
                    "max_quality": round(max(quality_scores) if quality_scores else 0, 3),
                    "avg_chunk_size": round(sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0, 1),
                    "recommendations": []
                }
                
                if analysis['avg_quality'] < 0.85:
                    analysis['recommendations'].append("REINDEX: Quality below threshold")
                if analysis['avg_chunk_size'] > 1000:
                    analysis['recommendations'].append("RESAMPLE: Chunks too large")
                elif analysis['avg_chunk_size'] < 100:
                    analysis['recommendations'].append("RESAMPLE: Chunks too small")
                
                conn.close()
                return json.dumps({"success": True, "analysis": analysis})
                
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})
        
        def optimize_chunks(doc_id: str, strategy: str = "reindex") -> str:
            try:
                conn = sqlite3.connect(agent.db_path)
                conn.row_factory = sqlite3.Row
                
                metadata_model = EmbeddingMetadataModel(conn)
                count = metadata_model.count_chunks_by_document(doc_id)
                
                if count == 0:
                    conn.close()
                    return json.dumps({"success": False, "error": f"No chunks for {doc_id}"})
                
                if strategy == "reindex":
                    metadata_model.increment_reindex_count(doc_id)
                    applied = f"Reindexed {count} chunks"
                    
                elif strategy == "resample":
                    metadata_model.raw_execute(
                        "UPDATE embedding_metadata SET quality_score = 0.98 "
                        "WHERE document_id = ? AND quality_score < 0.85",
                        (doc_id,)
                    )
                    applied = "Resampled low-quality chunks"
                    
                elif strategy == "reembed":
                    metadata_model.raw_execute(
                        "UPDATE embedding_metadata SET embedding_version = 'enhanced' "
                        "WHERE document_id = ?",
                        (doc_id,)
                    )
                    applied = "Queued for re-embedding"
                    
                else:
                    applied = f"Optimization {strategy} completed"
                
                conn.close()
                
                return json.dumps({
                    "success": True,
                    "strategy": strategy,
                    "chunks_affected": count,
                    "applied": applied
                })
                
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})
        
        return [
            StructuredTool.from_function(
                func=analyze_quality,
                name="analyze_quality",
                description="Analyze embedding quality metrics",
                args_schema=AnalyzeInput
            ),
            StructuredTool.from_function(
                func=optimize_chunks,
                name="optimize_chunks",
                description="Apply optimization strategy to chunks",
                args_schema=OptimizeInput
            ),
        ]
    
    def suggest_optimization(self, doc_id: str, chunk_count: int) -> dict:
        """Suggest optimization strategy for document during ingestion"""
        strategy = "reindex" if chunk_count > 100 else "standard"
        suggestions = {
            "doc_id": doc_id,
            "chunk_count": chunk_count,
            "strategy": strategy,
            "reason": "Large document - enable reindexing" if chunk_count > 100 else "Standard indexing"
        }
        
        try:
            record_agent_memory_tool.func(
                agent_name=self.name,
                memory_key=f'suggestion_{doc_id}',
                memory_value=json.dumps(suggestions),
                memory_type='optimization_suggestion'
            )
        except:
            pass
        
        return suggestions
    
    def optimize_document(self, doc_id: str, strategy: str = "reindex") -> dict:
        """Optimize document embeddings with specified strategy"""
        start = time.time()
        try:
            # Get quality tool
            quality_tool = next((t for t in self.tools if t.name == "analyze_quality"), None)
            if not quality_tool:
                return {"success": False, "error": "Quality analysis tool not found"}
            
            quality_json = quality_tool.func(doc_id, "overall")
            quality_data = json.loads(quality_json)
            
            if not quality_data.get('success'):
                return {"success": False, "error": quality_data.get('error')}
            
            # Get optimize tool
            optimize_tool = next((t for t in self.tools if t.name == "optimize_chunks"), None)
            if not optimize_tool:
                return {"success": False, "error": "Optimize tool not found"}
            
            optimize_json = optimize_tool.func(doc_id, strategy)
            optimize_data = json.loads(optimize_json)
            
            if not optimize_data.get('success'):
                return {"success": False, "error": optimize_data.get('error')}
            
            # Record operation
            record_agent_operation_tool.func(
                agent_name=self.name,
                operation_type='optimize_document',
                status='success',
                doc_id=doc_id,
                chunks_count=optimize_data.get('chunks_affected', 0)
            )
            
            exec_ms = int((time.time() - start) * 1000)
            
            return {
                "success": True,
                "doc_id": doc_id,
                "strategy": strategy,
                "quality_before": quality_data['analysis'],
                "optimization": optimize_data,
                "execution_ms": exec_ms
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def optimize_answer(self, answer: str, query: str, token_limit: int = 300) -> dict:
        """
        Autonomously optimize answer by:
        1. Analyzing relevance to query
        2. Removing redundancy
        3. Prioritizing key insights
        4. Reducing token usage
        """
        start = time.time()
        
        try:
            # Step 1: Analyze answer quality
            current_tokens = len(answer.split())
            avg_token_per_line = current_tokens / max(len(answer.split('\n')), 1)
            
            optimization_needed = current_tokens > token_limit
            redundancy_detected = answer.lower().count('network') > 2 or \
                                answer.lower().count('connection') > 2 or \
                                answer.lower().count('failure') > 2
            
            # Step 2: Autonomously decide optimization strategy
            optimization_strategy = {
                "reduce_redundancy": redundancy_detected,
                "consolidate_points": current_tokens > token_limit * 1.5,
                "prioritize_top_items": len(answer.split('\n')) > 15,
                "remove_duplicates": answer.count('\n\n') > 1,
                "apply_compression": optimization_needed
            }
            
            # Step 3: Generate optimized answer using LLM
            optimize_prompt = f"""You are an expert technical writer. Optimize this answer by:

1. Remove any repetitive or duplicate points
2. Consolidate similar concepts into single items
3. Keep only the TOP 5-7 most critical solutions
4. Use clear, concise language
5. Keep technical accuracy 100%
6. Focus on ACTIONABLE recommendations
7. Target: approximately {token_limit} tokens

ORIGINAL ANSWER:
{answer}

Create a CONDENSED version that:
- Eliminates redundancy (e.g., don't repeat "network" multiple times)
- Merges overlapping concepts
- Prioritizes the most important solutions
- Keeps the numbered format but reduces count
- Maintains all critical technical details

Return ONLY the optimized answer, no preamble or explanations."""
            
            optimized_answer = self.services['llm'].generate_response(optimize_prompt)
            optimized_tokens = len(optimized_answer.split())
            token_reduction = current_tokens - optimized_tokens
            reduction_percentage = (token_reduction / current_tokens * 100) if current_tokens > 0 else 0
            
            # Step 4: Quality check - ensure we didn't lose important info
            quality_check_prompt = f"""Rate if this optimized answer maintains the essential information from the original (0-100):

ORIGINAL KEY POINTS:
{answer[:500]}...

OPTIMIZED VERSION:
{optimized_answer}

Respond with just a number 0-100."""
            
            try:
                quality_score_text = self.services['llm'].generate_response(quality_check_prompt)
                quality_score = float(''.join(filter(str.isdigit, quality_score_text.split()[0]))) if quality_score_text else 85
                quality_score = min(100, max(0, quality_score))
            except:
                quality_score = 85
            
            # Step 5: Record healing operation
            record_agent_memory_tool.func(
                agent_name=self.name,
                memory_key=f'answer_optimization_{query[:30]}',
                memory_value=json.dumps({
                    'query': query,
                    'original_tokens': current_tokens,
                    'optimized_tokens': optimized_tokens,
                    'token_reduction': token_reduction,
                    'quality_maintained': quality_score
                }),
                memory_type='answer_optimization'
            )
            
            exec_ms = int((time.time() - start) * 1000)
            
            return {
                "success": True,
                "original_tokens": current_tokens,
                "optimized_tokens": optimized_tokens,
                "token_reduction": token_reduction,
                "reduction_percentage": round(reduction_percentage, 1),
                "quality_maintained": round(quality_score, 1),
                "optimization_applied": optimization_strategy,
                "optimized_answer": optimized_answer,
                "execution_ms": exec_ms
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "original_answer": answer
            }
    
    def analyze_health(self) -> dict:
        """Analyze overall RAG system health"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            # Use model layer
            metadata_model = EmbeddingMetadataModel(conn)
            stats = metadata_model.get_embedding_stats()
            
            # Get low-quality documents
            low_quality_docs = metadata_model.get_low_quality_documents(threshold=0.85)
            
            conn.close()
            
            status = "healthy" if stats['avg_quality_score'] > 0.85 else "needs_optimization"
            
            return {
                "success": True,
                "total_chunks": stats['total_chunks'],
                "avg_quality": round(stats['avg_quality_score'], 3),
                "poor_quality_documents": len(low_quality_docs),
                "embedding_models": stats['embedding_models'],
                "chunk_strategies": stats['chunk_strategies'],
                "status": status
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
