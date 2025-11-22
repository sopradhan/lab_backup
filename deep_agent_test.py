"""
Clean DeepAgent Test Suite with SSL Bypass Verification
Save this as: test_deepagent_clean.py
"""

import os
import ssl
import httpx
import urllib3
import requests
import sys
from pathlib import Path
from dotenv import load_dotenv

# ==============================
# STEP 1: AGGRESSIVE SSL BYPASS (MUST BE AT TOP)
# ==============================
print("🔐 Applying SSL bypass...")
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
print("✅ SSL bypass applied\n")

# ==============================
# STEP 2: ENV SETUP
# ==============================
load_dotenv()

API_KEY = os.getenv("AZURE_API_KEY", "sk-Wnu0C5qMv0cDuZoFy4aDpA")
API_ENDPOINT = os.getenv("API_ENDPOINT", "https://genailab.tcs.in/")
LLM_MODEL = os.getenv("LLM_MODEL", "azure/genailab-maas-gpt-4o")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "azure/genailab-maas-text-embedding-3-large")

print(f"📌 Configuration:")
print(f"   API Key: {API_KEY[:10]}...")
print(f"   Endpoint: {API_ENDPOINT}")
print(f"   LLM Model: {LLM_MODEL}")
print(f"   Embedding: {EMBEDDING_MODEL}\n")

# ==============================
# STEP 3: CHECK IMPORTS
# ==============================
print("📦 Importing libraries...")
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    print("   ✅ langchain_openai")
except ImportError as e:
    print(f"   ❌ langchain_openai: {e}")
    sys.exit(1)

try:
    from langchain_core.tools import tool
    print("   ✅ langchain_core.tools")
except ImportError as e:
    print(f"   ❌ langchain_core.tools: {e}")
    sys.exit(1)

try:
    from langgraph.graph import StateGraph
    print("   ✅ langgraph")
except ImportError as e:
    print(f"   ❌ langgraph: {e}")
    sys.exit(1)

try:
    from deepagents import create_deep_agent
    print("   ✅ deepagents")
    DEEPAGENTS_AVAILABLE = True
except ImportError:
    print("   ⚠️  deepagents (optional - will skip deepagent tests)")
    DEEPAGENTS_AVAILABLE = False

print()

# ==============================
# STEP 4: DEFINE TOOLS (ALL WITH DOCSTRINGS)
# ==============================
print("🛠️  Defining tools...")

@tool
def get_incident_summary(incident_id: str) -> str:
    """Retrieve incident summary from knowledge base by incident ID"""
    return f"Incident {incident_id}: Database connectivity issue. Status: Investigating"

@tool
def calculate_recovery_time(severity: str) -> str:
    """Calculate estimated recovery time based on incident severity"""
    times = {
        "critical": "30 minutes",
        "high": "1 hour",
        "medium": "2 hours",
        "low": "4 hours"
    }
    return f"Recovery time for {severity}: {times.get(severity, 'Unknown')}"

@tool
def log_action(action: str) -> str:
    """Log action to incident management system"""
    return f"✓ Action logged: {action}"

@tool
def get_recommendations(incident_type: str) -> str:
    """Get remediation recommendations for incident type"""
    recs = {
        "database": "1. Check connections\n2. Restart server\n3. Review logs",
        "network": "1. Check connectivity\n2. Verify firewall\n3. Ping endpoints"
    }
    return recs.get(incident_type, "No recommendations available")

print("✅ Tools defined (4 total)\n")

# ==============================
# STEP 5: INITIALIZE MODELS
# ==============================
print("🤖 Initializing models...")

def init_llm():
    """Initialize LLM with SSL bypass"""
    client = httpx.Client(verify=False)
    llm = ChatOpenAI(
        base_url=API_ENDPOINT,
        api_key=API_KEY,
        model=LLM_MODEL,
        temperature=0.3,
        http_client=client
    )
    return llm, client

def init_embedding():
    """Initialize embedding model with SSL bypass"""
    client = httpx.Client(verify=False)
    embedding = OpenAIEmbeddings(
        base_url=API_ENDPOINT,
        api_key=API_KEY,
        model=EMBEDDING_MODEL,
        http_client=client,
        tiktoken_enabled=True,
        tiktoken_model_name="text-embedding-3-large"
    )
    return embedding, client

try:
    llm, llm_client = init_llm()
    print("✅ LLM initialized")
except Exception as e:
    print(f"❌ LLM init failed: {e}")
    sys.exit(1)

try:
    embedding, emb_client = init_embedding()
    print("✅ Embedding model initialized\n")
except Exception as e:
    print(f"⚠️  Embedding init: {e}\n")
    embedding = None

# ==============================
# TEST 0: SSL BYPASS VERIFICATION
# ==============================
def test_ssl_verification():
    """Verify SSL bypass is working"""
    print("\n" + "="*60)
    print("TEST 0: SSL Bypass Verification")
    print("="*60)
    
    print("\n✓ Global SSL Bypass Status:")
    print("  • urllib3 warnings: DISABLED")
    print("  • SSL context: UNVERIFIED")
    print("  • requests patched: YES")
    
    print("\n🔍 Testing LLM Connection:")
    try:
        response = llm.invoke("Test SSL connection")
        print("  ✅ LLM connection successful (SSL bypassed)")
        print(f"     Response: {str(response.content)[:60]}...")
        return True
    except Exception as e:
        if "SSL" in str(e) or "CERTIFICATE" in str(e):
            print(f"  ❌ SSL Error: {e}")
            return False
        else:
            print(f"  ⚠️  Connection issue: {e}")
            return False

# ==============================
# TEST 1: SIMPLE TOOL-BASED AGENT
# ==============================
def test_simple_agent():
    """Test simple tool-based agent"""
    print("\n" + "="*60)
    print("TEST 1: Simple Tool-Based Agent")
    print("="*60)
    
    try:
        tools = [get_incident_summary, calculate_recovery_time, log_action]
        llm_with_tools = llm.bind_tools(tools)
        
        messages = [
            {"role": "user", "content": "What's recovery time for critical incident?"}
        ]
        
        response = llm_with_tools.invoke(messages)
        print(f"\n✅ Agent Response:")
        print(f"   {response.content[:100]}...")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

# ==============================
# TEST 2: TOOL CALLING
# ==============================
def test_tool_calling():
    """Test tool calling"""
    print("\n" + "="*60)
    print("TEST 2: Tool Calling")
    print("="*60)
    
    try:
        tools = [get_incident_summary, get_recommendations]
        llm_with_tools = llm.bind_tools(tools)
        
        messages = [
            {"role": "user", "content": "Tell me about incident INC-001 and database recommendations"}
        ]
        
        response = llm_with_tools.invoke(messages)
        print(f"\n✅ Tool Calling Response:")
        print(f"   {response.content[:100]}...")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

# ==============================
# TEST 3: STATE GRAPH
# ==============================
def test_state_graph():
    """Test LangGraph state management"""
    print("\n" + "="*60)
    print("TEST 3: LangGraph State Management")
    print("="*60)
    
    try:
        from typing import TypedDict, Annotated
        import operator
        
        class State(TypedDict):
            messages: Annotated[list, operator.add]
            data: dict
        
        workflow = StateGraph(State)
        
        def node1(state):
            incident = get_incident_summary("INC-TEST")
            return {"data": {"incident": incident}}
        
        def node2(state):
            time = calculate_recovery_time("critical")
            return {"data": {"recovery": time}}
        
        workflow.add_node("fetch", node1)
        workflow.add_node("calc", node2)
        workflow.set_entry_point("fetch")
        workflow.add_edge("fetch", "calc")
        
        graph = workflow.compile()
        result = graph.invoke({"messages": [], "data": {}})
        
        print(f"\n✅ Graph Execution Complete:")
        print(f"   Data: {result.get('data', {})}")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

# ==============================
# TEST 4: EMBEDDING CONNECTION
# ==============================
def test_embedding_connection():
    """Test embedding model connection"""
    print("\n" + "="*60)
    print("TEST 4: Embedding Model Connection")
    print("="*60)
    
    if not embedding:
        print("\n⚠️  Embedding model not available")
        return False
    
    try:
        texts = ["database incident", "recovery procedure"]
        embeddings = embedding.embed_documents(texts)
        print(f"\n✅ Embedding Connection Successful:")
        print(f"   Generated {len(embeddings)} embeddings")
        print(f"   Dimension: {len(embeddings[0]) if embeddings else 0}")
        return True
    except Exception as e:
        if "SSL" in str(e):
            print(f"\n❌ SSL Error: {e}")
            return False
        else:
            print(f"\n⚠️  Error: {e}")
            return False

# ==============================
# TEST 5: DEEPAGENT (IF AVAILABLE)
# ==============================
def test_deepagent():
    """Test DeepAgent creation"""
    print("\n" + "="*60)
    print("TEST 5: DeepAgent Creation")
    print("="*60)
    
    if not DEEPAGENTS_AVAILABLE:
        print("\n⚠️  DeepAgents not installed - skipping")
        return False
    
    try:
        tools = [get_incident_summary, calculate_recovery_time, log_action]
        
        # Try different parameter combinations
        try:
            # Try with model parameter
            agent = create_deep_agent(model=llm, tools=tools)
            print(f"\n✅ DeepAgent Created Successfully (model parameter)")
        except TypeError:
            try:
                # Try with tools only
                agent = create_deep_agent(tools=tools)
                print(f"\n✅ DeepAgent Created Successfully (tools parameter)")
            except TypeError:
                try:
                    # Try positional arguments
                    agent = create_deep_agent(llm, tools)
                    print(f"\n✅ DeepAgent Created Successfully (positional args)")
                except TypeError as e:
                    print(f"\n📋 DeepAgent signature info:")
                    print(f"   Error: {e}")
                    print(f"   Trying alternative approach...")
                    
                    # Try to introspect the function
                    import inspect
                    sig = inspect.signature(create_deep_agent)
                    print(f"   Parameters: {list(sig.parameters.keys())}")
                    return False
        
        return True
    except Exception as e:
        print(f"\n⚠️  DeepAgent initialization: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==============================
# MAIN TEST RUNNER
# ==============================
def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🚀 DeepAgent Test Suite")
    print("="*60 + "\n")
    
    tests = [
        ("SSL Bypass Verification", test_ssl_verification),
        ("Simple Tool-Based Agent", test_simple_agent),
        ("Tool Calling", test_tool_calling),
        ("State Graph", test_state_graph),
        ("Embedding Connection", test_embedding_connection),
        ("DeepAgent Creation", test_deepagent),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\n✨ Total: {passed}/{total} tests passed\n")

if __name__ == "__main__":
    main()
