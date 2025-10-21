from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import psycopg2
from psycopg2.extras import execute_values
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from datetime import datetime
import json
from contextlib import asynccontextmanager

# === CONFIGURATION ===
EMBEDDING_MODEL = "google/embeddinggemma-300m"
LLM_API_URL = "http://localhost:8000/v1"  # Your LLM or other AI
LLM_MODEL = "llama-4-scout-17b"  # Your LLM model

POSTGRES_CONFIG = {
    "host": "localhost",
    "database": "vectordb",
    "user": "vectoruser",
    "password": "vectorpass"
}

# Global variables for model (will be initialized on startup)
tokenizer = None
model = None
device = None

# === LIFESPAN MANAGEMENT ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and cleanup on shutdown"""
    global tokenizer, model, device
    
    # Startup: Load embedding model
    print("Loading embedding model...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on: {device}")
    
    # Initialize database
    init_database()
    
    yield  # Application runs here
    
    # Shutdown: Cleanup resources
    print("Shutting down and cleaning up resources...")
    del tokenizer, model

# === DATABASE SETUP ===
def get_db():
    return psycopg2.connect(**POSTGRES_CONFIG)

def init_database():
    """Initialize database with pgvector"""
    conn = get_db()
    cur = conn.cursor()
    
    # Enable pgvector
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Check if table exists and has wrong dimensions
    cur.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'user_memories' AND column_name = 'embedding'
    """)
    
    embedding_column = cur.fetchone()
    
    if embedding_column:
        # Table exists, check if we need to migrate dimensions
        cur.execute("SELECT COUNT(*) FROM user_memories")
        count = cur.fetchone()[0]
        
        if count > 0:
            print("⚠️  Existing data found. You may need to recreate the database.")
            print("   Run: DROP TABLE user_memories; and restart the server")
        else:
            # Empty table, safe to recreate
            cur.execute("DROP TABLE IF EXISTS user_memories")
            print("🔄 Recreating table with correct dimensions...")
    
    # Table for user memories
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_memories (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            content TEXT NOT NULL,
            memory_type VARCHAR(50),  -- 'preference', 'fact', 'conversation'
            embedding vector(768),
            created_at TIMESTAMP DEFAULT NOW(),
            importance FLOAT DEFAULT 1.0
        )
    """)
    
    # Index for fast search
    cur.execute("""
        CREATE INDEX IF NOT EXISTS user_memories_embedding_idx 
        ON user_memories USING ivfflat (embedding vector_cosine_ops)
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Database initialized!")

# Initialize FastAPI app with lifespan
app = FastAPI(title="AI with Long-term Memory", lifespan=lifespan)

# === MODELS ===
class ChatRequest(BaseModel):
    user_id: str
    message: str
    save_to_memory: bool = True

class ChatResponse(BaseModel):
    response: str
    relevant_memories: List[dict]
    memory_saved: bool

class Memory(BaseModel):
    user_id: str
    content: str
    memory_type: str = "fact"
    importance: float = 1.0

# === HELPER FUNCTIONS ===
def get_embedding(text: str) -> np.ndarray:
    """Generate embedding for text"""
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**encoded)
    
    embeddings = outputs.last_hidden_state.mean(dim=1)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings.cpu().numpy()[0]

def save_memory(user_id: str, content: str, memory_type: str = "fact", importance: float = 1.0):
    """Save memory to database"""
    embedding = get_embedding(content)
    
    conn = get_db()
    cur = conn.cursor()
    
    cur.execute("""
        INSERT INTO user_memories (user_id, content, memory_type, embedding, importance)
        VALUES (%s, %s, %s, %s, %s)
    """, (user_id, content, memory_type, embedding.tolist(), importance))
    
    conn.commit()
    cur.close()
    conn.close()

def retrieve_memories(user_id: str, query: str, top_k: int = 3, min_similarity: float = 0.5):
    """Find relevant memories"""
    query_embedding = get_embedding(query)
    
    conn = get_db()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT 
            content, 
            memory_type,
            1 - (embedding <=> %s::vector) AS similarity,
            created_at,
            importance
        FROM user_memories
        WHERE user_id = %s
        ORDER BY similarity DESC
        LIMIT %s
    """, (query_embedding.tolist(), user_id, top_k * 2))  # Fetch more, filter later
    
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    # Filter by min_similarity
    filtered_results = [
        {
            "content": r[0],
            "type": r[1],
            "similarity": float(r[2]),
            "created_at": r[3].isoformat(),
            "importance": float(r[4])
        }
        for r in results if r[2] >= min_similarity
    ]
    
    return filtered_results[:top_k]

def generate_response_and_extract_facts(user_message: str, context: str) -> dict:
    """
    Single LLM call that generates response AND extracts facts.
    Returns dict with 'response' (str) and 'facts' (list).
    """
    import requests
    
    prompt = f"""{context}User: {user_message}

Task: 
1. Respond naturally to the user's message, using any relevant memories from context
2. Extract ONLY NEW facts worth remembering (don't duplicate existing memories!)

IMPORTANT: Look at the context above - those are existing memories. 
ONLY extract facts that are genuinely NEW and not already covered in the context.
Don't extract the same information in different words!

Respond with ONLY valid JSON in this exact format:
{{
  "response": "Your natural response to the user here",
  "facts": [
    {{"content": "Fact in clear sentence", "memory_type": "preference/personal/skill/goal/opinion/experience/other", "importance": 1.5}}
  ]
}}

Facts to extract (only if NEW):
- User preferences (likes, dislikes, favorites)
- Personal info (name, location, job, hobbies, family)
- Skills, knowledge, interests
- Goals, plans, projects
- Opinions, beliefs, values
- Experiences, stories

If no NEW facts worth remembering, set "facts" to empty array [].

JSON:"""

    try:
        response = requests.post(
            f"{LLM_API_URL}/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 600,  # Enough for response + facts
                "temperature": 0.5  # Balanced creativity and consistency
            },
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        response.raise_for_status()
        
        ai_response = response.json()["choices"][0]["message"]["content"].strip()
        
        # Extract JSON from response
        start_idx = ai_response.find('{')
        end_idx = ai_response.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = ai_response[start_idx:end_idx]
            result = json.loads(json_str)
            
            # Validate structure
            if 'response' not in result:
                print("⚠️  No 'response' field in AI output, using raw text")
                return {
                    'response': ai_response,
                    'facts': []
                }
            
            # Validate facts
            validated_facts = []
            for fact in result.get('facts', []):
                if isinstance(fact, dict) and 'content' in fact:
                    validated_facts.append({
                        'content': fact.get('content', ''),
                        'memory_type': fact.get('memory_type', 'other'),
                        'importance': float(fact.get('importance', 1.0))
                    })
            
            return {
                'response': result['response'],
                'facts': validated_facts
            }
        else:
            print(f"⚠️  AI response doesn't contain valid JSON: {ai_response[:100]}")
            return {
                'response': ai_response,
                'facts': []
            }
            
    except json.JSONDecodeError as e:
        print(f"⚠️  Failed to parse AI response as JSON: {e}")
        return {
            'response': ai_response if 'ai_response' in locals() else "[JSON Parse Error]",
            'facts': []
        }
    except Exception as e:
        print(f"⚠️  Error calling LLM: {e}")
        return {
            'response': f"[LLM Error: {str(e)}]",
            'facts': []
        }

# === MAIN LOGIC ===
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chatbot with long-term memory:
    1. Retrieve relevant memories from RAG
    2. Generate response AND extract facts in single LLM call
    3. Save new memories
    """
    
    # 1. Retrieve relevant memories
    memories = retrieve_memories(request.user_id, request.message, top_k=3, min_similarity=0.6)
    
    # 2. Build context with memories
    context = ""
    if memories:
        context = "What I remember about this user:\n"
        for mem in memories:
            context += f"- {mem['content']} (similarity: {mem['similarity']:.2f})\n"
        context += "\n"
    
    # 3. Single LLM call: generate response + extract facts
    result = generate_response_and_extract_facts(request.message, context)
    
    # 4. Save extracted facts to memory (with deduplication)
    memory_saved = False
    memories_saved_count = 0
    if request.save_to_memory and result['facts']:
        for fact in result['facts']:
            # Check if similar fact already exists
            existing = retrieve_memories(request.user_id, fact['content'], top_k=1, min_similarity=0.85)
            if existing:
                print(f"⚠️  Skipping duplicate: '{fact['content']}' (similar to: '{existing[0]['content']}')")
                continue
            
            save_memory(
                request.user_id, 
                fact['content'], 
                memory_type=fact['memory_type'], 
                importance=fact['importance']
            )
            memories_saved_count += 1
            memory_saved = True
        
        if memories_saved_count > 0:
            print(f"💾 Saved {memories_saved_count} new memories for user {request.user_id}")
        elif result['facts']:
            print(f"⚠️  All facts were duplicates - nothing saved")
    
    return ChatResponse(
        response=result['response'],
        relevant_memories=memories,
        memory_saved=memory_saved
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)