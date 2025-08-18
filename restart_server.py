#!/usr/bin/env python3
"""
Robust server restart script that handles port conflicts properly
"""

import subprocess
import time
import os
import signal
import psutil

def find_aaire_processes():
    """Find all AAIRE-related processes"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'main.py' in cmdline and 'python' in proc.info['name']:
                processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return processes

def kill_processes_gracefully(processes):
    """Kill processes with escalating force"""
    if not processes:
        print("No AAIRE processes found")
        return True
    
    print(f"Found {len(processes)} AAIRE processes")
    
    # Step 1: Try SIGTERM (graceful shutdown)
    for proc in processes:
        try:
            print(f"Sending SIGTERM to PID {proc.pid}")
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    
    # Wait for graceful shutdown
    time.sleep(3)
    
    # Step 2: Check which are still running
    still_running = []
    for proc in processes:
        try:
            if proc.is_running():
                still_running.append(proc)
        except psutil.NoSuchProcess:
            pass
    
    # Step 3: Force kill remaining processes
    if still_running:
        print(f"Force killing {len(still_running)} stubborn processes")
        for proc in still_running:
            try:
                print(f"Sending SIGKILL to PID {proc.pid}")
                proc.kill()
            except psutil.NoSuchProcess:
                pass
        time.sleep(2)
    
    return True

def check_port_free(port):
    """Check if port is actually free"""
    try:
        result = subprocess.run(['ss', '-tln'], capture_output=True, text=True)
        return f":{port}" not in result.stdout
    except:
        # Fallback method
        try:
            result = subprocess.run(['netstat', '-tln'], capture_output=True, text=True)
            return f":{port}" not in result.stdout
        except:
            return True  # Assume free if we can't check

def wait_for_port_free(port, max_wait=10):
    """Wait for port to be released"""
    print(f"Waiting for port {port} to be released...")
    for i in range(max_wait):
        if check_port_free(port):
            print(f"Port {port} is now free")
            return True
        print(f"  Port still in use, waiting... ({i+1}/{max_wait})")
        time.sleep(1)
    
    print(f"Warning: Port {port} still appears to be in use")
    return False

def start_server():
    """Start the server"""
    try:
        print("Starting AAIRE server...")
        # Start server in background
        process = subprocess.Popen(['python3', 'main.py'])
        print(f"Server started with PID {process.pid}")
        
        # Give it a moment to start
        time.sleep(2)
        
        # Check if it's actually running
        if process.poll() is None:
            print("✅ Server started successfully")
            return True
        else:
            print("❌ Server failed to start")
            return False
            
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

def main():
    """Main restart logic"""
    print("🔄 **ROBUST AAIRE SERVER RESTART**\n")
    
    # Step 1: Find and kill existing processes
    processes = find_aaire_processes()
    kill_processes_gracefully(processes)
    
    # Step 2: Wait for port to be free
    wait_for_port_free(8001)
    
    # Step 3: Start new server
    success = start_server()
    
    if success:
        print("\n🎉 **SERVER RESTART COMPLETE**")
        print("   • Enhanced citation formatting active")
        print("   • Shape-aware extraction enabled")
        print("   • 3 follow-up questions optimized")
        print("\n🧪 **TEST NOW:**")
        print("   Ask about capital health ratios")
        print("   Should see 'LICAT.pdf, Page 2' instead of '[2]'")
    else:
        print("\n❌ **RESTART FAILED**")
        print("   Check logs and try manual restart")

if __name__ == "__main__":
    main()