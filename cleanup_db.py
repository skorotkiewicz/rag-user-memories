#!/usr/bin/env python3
"""
Database cleanup script - drops and recreates user_memories table
Use this if you have dimension mismatch errors
"""
import psycopg2

POSTGRES_CONFIG = {
    "host": "localhost",
    "database": "vectordb",
    "user": "vectoruser",
    "password": "vectorpass"
}

def cleanup_database():
    """Drop and recreate user_memories table"""
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    cur = conn.cursor()
    
    print("🗑️  Dropping existing user_memories table...")
    cur.execute("DROP TABLE IF EXISTS user_memories CASCADE")
    
    print("🔄 Recreating table with correct dimensions (768)...")
    cur.execute("""
        CREATE TABLE user_memories (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            content TEXT NOT NULL,
            memory_type VARCHAR(50),
            embedding vector(768),
            created_at TIMESTAMP DEFAULT NOW(),
            importance FLOAT DEFAULT 1.0
        )
    """)
    
    print("📊 Creating index...")
    cur.execute("""
        CREATE INDEX user_memories_embedding_idx 
        ON user_memories USING ivfflat (embedding vector_cosine_ops)
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    
    print("✅ Database cleaned up! You can now restart proxy.py")

if __name__ == "__main__":
    try:
        cleanup_database()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure PostgreSQL is running and accessible")
