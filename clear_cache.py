#!/usr/bin/env python3
"""
Clear Redis cache to resolve old citation issues
"""

import redis
import os

def clear_cache():
    """Clear all cached responses"""
    try:
        # Connect to Redis
        cache = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
            decode_responses=True
        )
        
        # Test connection
        cache.ping()
        print("✅ Connected to Redis")
        
        # Get all keys
        all_keys = list(cache.scan_iter(match="*"))
        print(f"Found {len(all_keys)} cache entries")
        
        if all_keys:
            # Show sample keys
            print("\nSample cache keys:")
            for key in all_keys[:5]:
                print(f"  {key}")
            
            # Clear all keys
            cleared_count = cache.delete(*all_keys)
            print(f"\n✅ Cleared {cleared_count} cache entries")
        else:
            print("No cache entries to clear")
        
        # Verify cleared
        remaining_keys = list(cache.scan_iter(match="*"))
        print(f"Remaining cache entries: {len(remaining_keys)}")
        
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")

if __name__ == "__main__":
    clear_cache()