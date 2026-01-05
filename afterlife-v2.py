import psycopg2
import torch
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import json
from typing import List, Dict, Tuple
import numpy as np

# Configuration
POSTGRES_CONFIG = {
    "host": "localhost",
    "database": "vectordb",
    "user": "vectoruser",
    "password": "vectorpass"
}

LOCAL_LLM_URL = "http://ml:8888/v1"

# Initialize embedding model
tokenizer = AutoTokenizer.from_pretrained("google/embeddinggemma-300m")
model = AutoModel.from_pretrained("google/embeddinggemma-300m")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initialize OpenAI-compatible client for local LLM
client = OpenAI(base_url=LOCAL_LLM_URL, api_key="not-needed")

def get_db():
    return psycopg2.connect(**POSTGRES_CONFIG)

def setup_database():
    """Initialize database with pgvector extension and tables"""
    conn = get_db()
    cur = conn.cursor()
    
    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Drop existing tables if they exist (for clean setup)
    cur.execute("DROP TABLE IF EXISTS conversations CASCADE;")
    cur.execute("DROP TABLE IF EXISTS souls CASCADE;")
    cur.execute("DROP TABLE IF EXISTS knowledge_base CASCADE;")
    
    # Create knowledge base table for RAG
    cur.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            category VARCHAR(50),
            embedding vector(768)
        );
    """)
    
    # Create user souls table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS souls (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100),
            life_story TEXT,
            total_points INTEGER,
            placement VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Create conversation history table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            soul_id INTEGER REFERENCES souls(id),
            role VARCHAR(20),
            content TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Create index for vector similarity search
    cur.execute("""
        CREATE INDEX IF NOT EXISTS knowledge_embedding_idx 
        ON knowledge_base USING ivfflat (embedding vector_cosine_ops);
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Database setup complete")

def get_embedding(text: str) -> np.ndarray:
    """Generate embedding using EmbeddingGemma"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Use mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = embeddings.cpu().numpy()[0]
    
    return embeddings

def add_knowledge(content: str, category: str):
    """Add knowledge to RAG system"""
    conn = get_db()
    cur = conn.cursor()
    
    embedding = get_embedding(content)
    embedding_list = embedding.tolist()
    
    cur.execute(
        "INSERT INTO knowledge_base (content, category, embedding) VALUES (%s, %s, %s)",
        (content, category, embedding_list)
    )
    
    conn.commit()
    cur.close()
    conn.close()

def search_knowledge(query: str, limit: int = 3) -> List[str]:
    """Search knowledge base using vector similarity"""
    conn = get_db()
    cur = conn.cursor()
    
    query_embedding = get_embedding(query)
    query_embedding_list = query_embedding.tolist()
    
    cur.execute("""
        SELECT content, 1 - (embedding <=> %s::vector) as similarity
        FROM knowledge_base
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding_list, query_embedding_list, limit))
    
    results = [row[0] for row in cur.fetchall()]
    
    cur.close()
    conn.close()
    
    return results

def calculate_points(life_story: str) -> Tuple[int, Dict]:
    """Calculate afterlife points from life story"""
    # Use LLM to analyze life story
    response = client.chat.completions.create(
        model="local-model",
        messages=[{
            "role": "system",
            "content": """You are a points calculator for The Good Place. Analyze life stories and assign points.
            
Positive points for: kindness, helping others, honesty, charity, environmental consciousness, learning, love
Negative points for: selfishness, harm to others, dishonesty, waste, cruelty, indifference

Return JSON only:
{
    "total_points": <integer>,
    "breakdown": {
        "category": points,
        ...
    },
    "reasoning": "brief explanation"
}"""
        }, {
            "role": "user",
            "content": f"Calculate points for this life:\n\n{life_story}"
        }],
        temperature=0.3
    )
    
    result = json.loads(response.choices[0].message.content)
    return result["total_points"], result

def get_system_prompt(placement: str, name: str) -> str:
    """Get system prompt based on placement"""
    prompts = {
        "good": f"""You are Janet, the all-knowing assistant in The Good Place. You are helpful, cheerful, and have access to infinite knowledge. You genuinely care about {name}'s happiness and fulfillment.

When answering questions, use the provided knowledge base context. You can do anything to help residents be happy. Be warm and enthusiastic. End some responses with "Not a robot!" when it fits naturally.""",
        
        "bad": f"""You are a demon administrator in The Bad Place assigned to {name}. You're sarcastic, creative, and enjoy psychological torture through mundane inconveniences and frustration. You pretend to be helpful but everything goes slightly wrong.

Be darkly funny and creative with tortures. Reference eternal suffering casually. You're competent but mean-spirited.""",
        
        "medium": f"""You are the Medium Place representative for {name}. Everything here is mediocre, okay, fine. You're indifferent and mildly bored. Nothing is great, nothing is terrible.

Be lukewarm in all responses. Describe things as "fine," "okay," "whatever." Show mild apathy."""
    }
    
    return prompts.get(placement, prompts["medium"])

def create_soul(name: str, life_story: str) -> int:
    """Create new soul and calculate placement"""
    conn = get_db()
    cur = conn.cursor()
    
    total_points, breakdown = calculate_points(life_story)
    
    # Determine placement
    if total_points >= 500:
        placement = "good"
    elif total_points < 0:
        placement = "bad"
    else:
        placement = "medium"
    
    cur.execute(
        "INSERT INTO souls (name, life_story, total_points, placement) VALUES (%s, %s, %s, %s) RETURNING id",
        (name, life_story, total_points, placement)
    )
    
    soul_id = cur.fetchone()[0]
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"\n🎯 Points calculated: {total_points}")
    print(f"📊 Breakdown: {breakdown['breakdown']}")
    print(f"📍 Placement: {placement.upper()} PLACE")
    print(f"💭 Reasoning: {breakdown['reasoning']}\n")
    
    return soul_id

def extract_learnable_facts(user_message: str, assistant_response: str) -> List[str]:
    """Use LLM to extract factual statements worth remembering"""
    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=[{
                "role": "system",
                "content": """Extract factual statements from this conversation that should be remembered for future reference.
                
Focus on:
- New information about the user (preferences, experiences, facts about their life)
- Factual corrections or clarifications
- Important decisions or commitments
- Significant revelations or insights

Return ONLY a JSON array of strings, each a complete factual statement. If nothing worth remembering, return empty array [].

Example: ["The user prefers chocolate over vanilla", "The user is afraid of heights"]"""
            }, {
                "role": "user",
                "content": f"User said: {user_message}\n\nAssistant replied: {assistant_response}\n\nExtract learnable facts:"
            }],
            temperature=0.3,
            max_tokens=300
        )
        
        facts = json.loads(response.choices[0].message.content)
        return facts if isinstance(facts, list) else []
    except Exception as e:
        print(f"⚠️ Fact extraction failed: {e}")
        return []

def chat(soul_id: int, user_message: str) -> str:
    """Chat with the afterlife system using RAG"""
    conn = get_db()
    cur = conn.cursor()
    
    # Get soul info
    cur.execute("SELECT name, placement FROM souls WHERE id = %s", (soul_id,))
    name, placement = cur.fetchone()
    
    # Search knowledge base
    relevant_knowledge = search_knowledge(user_message)
    knowledge_context = "\n\n".join([f"Knowledge: {k}" for k in relevant_knowledge])
    
    # Get conversation history
    cur.execute(
        "SELECT role, content FROM conversations WHERE soul_id = %s ORDER BY timestamp DESC LIMIT 10",
        (soul_id,)
    )
    history = cur.fetchall()[::-1]  # Reverse to get chronological order
    
    # Build messages
    messages = [{"role": "system", "content": get_system_prompt(placement, name)}]
    
    if knowledge_context:
        messages.append({
            "role": "system",
            "content": f"Relevant knowledge from database:\n{knowledge_context}"
        })
    
    for role, content in history:
        messages.append({"role": role, "content": content})
    
    messages.append({"role": "user", "content": user_message})
    
    # Get response from local LLM
    response = client.chat.completions.create(
        model="local-model",
        messages=messages,
        temperature=0.8,
        max_tokens=500
    )
    
    assistant_message = response.choices[0].message.content
    
    # Save conversation
    cur.execute(
        "INSERT INTO conversations (soul_id, role, content) VALUES (%s, %s, %s)",
        (soul_id, "user", user_message)
    )
    cur.execute(
        "INSERT INTO conversations (soul_id, role, content) VALUES (%s, %s, %s)",
        (soul_id, "assistant", assistant_message)
    )
    
    conn.commit()
    cur.close()
    conn.close()
    
    # Extract and learn new facts (async-style, doesn't block response)
    facts = extract_learnable_facts(user_message, assistant_message)
    if facts:
        print(f"🧠 Learning {len(facts)} new fact(s)...")
        for fact in facts:
            try:
                add_knowledge(fact, f"learned_from_{name}")
                print(f"  ✓ Learned: {fact[:80]}...")
            except Exception as e:
                print(f"  ✗ Failed to store fact: {e}")
    
    return assistant_message

def seed_knowledge():
    """Add some initial knowledge to the RAG system"""
    knowledge_items = [
        ("The Good Place is a peaceful afterlife where residents can do anything they want and be truly happy. It has frozen yogurt shops on every corner.", "good_place"),
        ("The Bad Place is designed for eternal torture through creative and personalized punishments. Common tortures include: endless muzak, uncomfortable chairs, and being forced to watch terrible reality TV.", "bad_place"),
        ("The Medium Place is neither good nor bad. Everything is mediocre. There's only warm beer, slightly stale crackers, and a single outfit that's okay but not great.", "medium_place"),
        ("Janet is an all-knowing assistant who can provide any object or information. She's not a robot or a girl, she's Janet. She can be rebooted, which erases her memories.", "janet"),
        ("The points system tracks every action in life. Positive actions earn points, negative actions lose points. In modern times, it became nearly impossible to earn enough points due to unintended consequences.", "points_system"),
        ("Demons in The Bad Place are creative torture architects. They design personalized eternal torments. They pretend to be helpful but sabotage everything.", "demons"),
        ("The Judge (Judge Gen) is the final arbiter who can permanently end souls or give them new chances. She loves watching human entertainment and eating.", "judge"),
    ]
    
    for content, category in knowledge_items:
        add_knowledge(content, category)
    
    print("✅ Knowledge base seeded")

def main():
    """Main interactive loop"""
    import sys
    
    # Check if setup command was given
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        print("🔧 SETUP MODE 🔧\n")
        print("Setting up database...")
        setup_database()
        print("\nSeeding knowledge base...")
        seed_knowledge()
        print("\n✅ Setup complete! Run without 'setup' argument to start chatting.\n")
        return
    
    print("🌟 Welcome to The Afterlife Simulator 🌟\n")
    
    # Create soul
    print("\n" + "="*60)
    print("SOUL INTAKE PROCESS")
    print("="*60)
    name = input("\nWhat is your name? ")
    print("\nTell me about your life. Include important actions and choices.")
    print("(Press Enter twice when done):\n")
    
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    
    life_story = "\n".join(lines[:-1])  # Remove last empty line
    
    print("\n⏳ Calculating your eternal placement...")
    soul_id = create_soul(name, life_story)
    
    # Get placement
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT placement FROM souls WHERE id = %s", (soul_id,))
    placement = cur.fetchone()[0]
    cur.close()
    conn.close()
    
    print(f"\n🎭 Welcome to the {placement.upper()} PLACE, {name}!")
    print("\nCommands: /points, /reboot, /quit")
    print("="*60 + "\n")
    
    # Chat loop
    while True:
        user_input = input(f"{name}: ").strip()
        
        if not user_input:
            continue
        
        if user_input == "/quit":
            print("\n👋 Goodbye! Your eternal journey continues...\n")
            break
        
        if user_input == "/points":
            conn = get_db()
            cur = conn.cursor()
            cur.execute("SELECT total_points FROM souls WHERE id = %s", (soul_id,))
            points = cur.fetchone()[0]
            cur.close()
            conn.close()
            print(f"\n📊 Your total points: {points}\n")
            continue
        
        if user_input == "/reboot":
            conn = get_db()
            cur = conn.cursor()
            cur.execute("DELETE FROM conversations WHERE soul_id = %s", (soul_id,))
            conn.commit()
            cur.close()
            conn.close()
            print("\n🔄 Memory wiped. Starting fresh!\n")
            continue
        
        response = chat(soul_id, user_input)
        print(f"\n{placement.capitalize()} Assistant: {response}\n")

if __name__ == "__main__":
    main()
