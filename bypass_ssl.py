import os
import sqlite3
import ssl
import httpx
import urllib3
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# ==============================
# AGGRESSIVE SSL BYPASS (Must be at the very top)
# ==============================
# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Disable SSL verification globally
ssl._create_default_https_context = ssl._create_unverified_context

# Monkey-patch requests to disable SSL verification
original_request = requests.Session.request
def patched_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return original_request(self, method, url, **kwargs)
requests.Session.request = patched_request

# Also patch the requests.get function directly
original_get = requests.get
def patched_get(url, **kwargs):
    kwargs['verify'] = False
    return original_get(url, **kwargs)
requests.get = patched_get

# ==============================
# 1️⃣ ENV + CONFIG
# ==============================
load_dotenv()

API_KEY = "sk-**************"
API_ENDPOINT = "https://genailab.tcs.in/"
LLM_MODEL = "azure/genailab-maas-gpt-4o"
EMBEDDING_MODEL_NAME = "azure/genailab-maas-text-embedding-3-large"
PERSIST_DIR = r"C:\Users\GENAIKOLGPUSR15\Desktop\Incident_management\incident_db\data\chroma_index"
DB_PATH = r"C:\Users\GENAIKOLGPUSR15\Desktop\Incident_management\incident_db\data\incident_iq.db"

# Set token cache directory
tiktoken_cache_dir = "./token"
os.makedirs(tiktoken_cache_dir, exist_ok=True)
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# ==============================
# 2️⃣ DATABASE ACCESS
# ==============================
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def fetch_knowledge_base(conn):
    cursor = conn.cursor()
    rows = cursor.execute("SELECT * FROM knowledge_base").fetchall()
    cols = [desc[0] for desc in cursor.description]
    return rows, cols

# ==============================
# 3️⃣ DOCUMENT PREPARATION
# ==============================
def prepare_documents(rows, cols, rbac_tags):
    docs, metadatas = [], []
    for row in rows:
        record = dict(zip(cols, row))
        text = f"""
Cause: {record['cause']}
Description: {record['description']}
Impact: {record['impact']}
Remediation: {record['remediation_steps']}
RCA: {record['rca']}
Business Impact: {record['business_impact']}
Estimated Recovery: {record['estimated_recovery_time']}
Environment: {record['environment']}
"""
        for tag in rbac_tags:
            docs.append(text)
            metadatas.append({
                "id": record["id"],
                "environment": record["environment"],
                "rbac_tag": tag,
                "namespace": "Acme_Corp.IT.OPS",
                "created_at": record["created_at"]
            })
    return docs, metadatas

# ==============================
# 4️⃣ MODEL INITIALIZATION & VECTOR STORE
# ==============================
def initialize_models():
    client = httpx.Client(verify=False)
    llm = ChatOpenAI(
        base_url=API_ENDPOINT,
        api_key=API_KEY,
        model=LLM_MODEL,
        temperature=0.3,
        http_client=client
    )
    
    # Fix for Azure model naming - explicitly set tiktoken model
    embedding_model = OpenAIEmbeddings(
        base_url=API_ENDPOINT,
        api_key=API_KEY,
        model=EMBEDDING_MODEL_NAME,
        http_client=client,
        tiktoken_enabled=True,
        tiktoken_model_name="text-embedding-3-large"  # Use standard OpenAI model name for tiktoken
    )
    return llm, embedding_model

def build_vector_store(docs, metadatas, embedding_model):
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model
    )
    
    # Process in batches to avoid memory issues with 15,000 documents
    batch_size = 1000
    total_docs = len(docs)
    
    print(f"Adding {total_docs} documents to ChromaDB in batches of {batch_size}...")
    
    for i in range(0, total_docs, batch_size):
        batch_docs = docs[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        vectordb.add_texts(texts=batch_docs, metadatas=batch_metas)
        print(f"  ✓ Processed {min(i+batch_size, total_docs)}/{total_docs} documents")
    
    vectordb.persist()
    print(f"✅ Successfully inserted {total_docs} documents with RBAC metadata into ChromaDB.")

# ==============================
# 5️⃣ MAIN EXECUTION
# ==============================
def main():
    rbac_tags = [
        "Acme_Corp-Engineering-admin",
        "Acme_Corp-Engineering-manager",
        "Acme_Corp-Engineering-employee"
    ]

    print("Connecting to database...")
    conn = get_db_connection()
    
    print("Fetching knowledge base...")
    rows, cols = fetch_knowledge_base(conn)
    
    print(f"Preparing documents for {len(rows)} records × {len(rbac_tags)} RBAC tags = {len(rows) * len(rbac_tags)} total documents...")
    docs, metadatas = prepare_documents(rows, cols, rbac_tags)
    
    print("Initializing models...")
    llm, embedding_model = initialize_models()
    
    print("Building vector store...")
    build_vector_store(docs, metadatas, embedding_model)
    
    conn.close()
    print("✅ Process completed successfully!")

if __name__ == "__main__":
    main()
