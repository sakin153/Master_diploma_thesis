#!/usr/bin/env python3
"""Clear LLM cache to force fresh responses."""

import os
import sqlite3

# Default cache path
DEFAULT_CACHE_DB_PATH = "/var/tmp/ciare/.ollama_cache.sqlite3"

def clear_cache():
    """Clear the LLM cache database."""
    
    # Check CIARE_CACHE_DIR env var
    custom = os.environ.get("CIARE_CACHE_DIR")
    if custom:
        cache_path = os.path.join(custom, ".ollama_cache.sqlite3")
    else:
        cache_path = DEFAULT_CACHE_DB_PATH
    
    if not os.path.exists(cache_path):
        print(f"✓ Cache file does not exist: {cache_path}")
        return
    
    try:
        # Connect and clear the cache table
        with sqlite3.connect(cache_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM llm_cache")
            count = cursor.fetchone()[0]
            
            print(f"Found {count} cached LLM responses")
            
            if count > 0:
                conn.execute("DELETE FROM llm_cache")
                conn.commit()
                print(f"✓ Cleared {count} cached responses from {cache_path}")
            else:
                print(f"✓ Cache is already empty")
                
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        return
    
    print("\n✓ LLM cache cleared successfully!")
    print("Next run will generate fresh responses from the model.")

if __name__ == "__main__":
    clear_cache()
