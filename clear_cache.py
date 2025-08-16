#!/usr/bin/env python3
"""
Clear AAIRE cache and restart server with enhanced citation formatting
"""

import redis
import os
import subprocess
import time

def clear_cache():
    """Clear all cached responses to force fresh citations with page numbers"""
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
        
        # Clear all cached responses
        cache.flushdb()
        print("✅ All cached responses cleared")
        
        print("📄 Next responses will use new citation format with page numbers")
        print("💬 Follow-up questions limited to 3 for optimal UX")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Cache clear failed (OK if Redis not running): {e}")
        return False

def restart_server():
    """Restart the server to load citation improvements"""
    try:
        print("\n🔄 Restarting server with enhanced citation formatting...")
        
        # Kill existing process
        subprocess.run(['pkill', '-f', 'main.py'], check=False)
        print("   Stopped current server")
        
        # Wait for graceful shutdown
        time.sleep(3)
        
        # Start new server
        subprocess.Popen(['python3', 'main.py'])
        print("   Started server with enhanced citations")
        
        return True
        
    except Exception as e:
        print(f"❌ Server restart failed: {e}")
        return False

if __name__ == "__main__":
    print("🧹 **CLEARING CACHE FOR ENHANCED CITATIONS**\n")
    
    # Clear cache
    cache_cleared = clear_cache()
    
    # Ask about server restart
    if cache_cleared:
        print("\n🚀 **CACHE CLEARED SUCCESSFULLY**")
        print("   • Old [1], [2] citations removed")
        print("   • Next responses will show page numbers")
        print("   • Follow-up questions optimized to 3")
        
        # Auto-restart if running as script
        restart_server()
        
        print("\n🎉 **AAIRE READY WITH ENHANCED CITATIONS**")
        print("   Test by asking: 'what are the ratios used to assess capital health?'")
        print("   Should now show: 'LICAT.pdf, Page 2' instead of '[2]'")
    
    else:
        print("\n⚠️ **MANUAL RESTART RECOMMENDED**")
        print("   Run: pkill -f main.py && python3 main.py")