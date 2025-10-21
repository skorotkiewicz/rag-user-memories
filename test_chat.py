#!/usr/bin/env python3
"""
Simple interactive chat tester for AI Memory Proxy
"""
import requests
import json

BASE_URL = "http://localhost:8001"

def chat_with_ai(user_id: str, message: str):
    """Send message to AI and get response"""
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json={
                "user_id": user_id,
                "message": message,
                "save_to_memory": True
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
            
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to server. Is proxy.py running on port 8001?"}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}

def main():
    print("=" * 60)
    print("🤖 AI Memory Proxy - Interactive Chat Tester")
    print("=" * 60)
    
    # Get user ID
    user_id = input("\nEnter your user ID: ").strip()
    if not user_id:
        user_id = "test_user"
        print(f"Using default user ID: {user_id}")
    
    print(f"\n✅ Connected as: {user_id}")
    print("💡 Type 'quit' or 'exit' to stop")
    print("💡 Type 'clear' to clear screen")
    print("💡 Type 'memories' to see relevant memories")
    print("-" * 60)
    
    while True:
        try:
            # Get user input
            user_input = input(f"\n[{user_id}] You: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                print("\n" * 50)  # Clear screen
                continue
            elif user_input.lower() == 'memories':
                print("\n📚 Recent memories will be shown in AI responses")
                continue
            elif not user_input:
                continue
            
            # Send to AI
            print("🤖 AI is thinking...")
            result = chat_with_ai(user_id, user_input)
            
            if "error" in result:
                print(f"❌ {result['error']}")
                continue
            
            # Display AI response
            print(f"\n🤖 AI: {result['response']}")
            
            # Show relevant memories if any
            if result.get('relevant_memories'):
                print(f"\n📚 Used {len(result['relevant_memories'])} memories:")
                for mem in result['relevant_memories']:
                    print(f"  • [{mem['type']}] {mem['content']} (similarity: {mem['similarity']:.2f})")
            
            # Show if memory was saved
            if result.get('memory_saved'):
                print("💾 New facts saved to memory")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
