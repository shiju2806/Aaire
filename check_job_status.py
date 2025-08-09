#!/usr/bin/env python3
"""
Check processing job status for uploaded documents
"""

def check_job_status():
    """Check if we can find any trace of the job processing"""
    
    print("Checking for traces of job processing...")
    print("="*60)
    
    # The job would be stored in memory in the DocumentProcessor
    # But we can check for any logs or files that might indicate processing
    
    from pathlib import Path
    
    # Check for any temporary files
    temp_dirs = [
        Path("/tmp"),
        Path("data/uploads"),
        Path("/var/tmp")
    ]
    
    target_job_id = "6041a8f3-0a87-4297-b00f-8de6ea9a322a"
    
    for temp_dir in temp_dirs:
        if temp_dir.exists():
            print(f"\nChecking {temp_dir}:")
            try:
                for file_path in temp_dir.iterdir():
                    if target_job_id in str(file_path):
                        print(f"  ✅ Found: {file_path}")
                        return str(file_path)
                    elif "pwc" in str(file_path).lower() or "foreign" in str(file_path).lower():
                        print(f"  📄 Potential match: {file_path}")
                        
                print(f"  ❌ No job files found in {temp_dir}")
            except PermissionError:
                print(f"  ⚠️  Permission denied accessing {temp_dir}")
            except Exception as e:
                print(f"  ⚠️  Error accessing {temp_dir}: {e}")
    
    # Check if the application is even running
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'python' in result.stdout and ('main.py' in result.stdout or 'uvicorn' in result.stdout):
            print(f"\n✅ Python application appears to be running")
        else:
            print(f"\n❌ No Python application found running")
            print("This might explain why uploads aren't being processed")
    except Exception as e:
        print(f"\n⚠️  Could not check running processes: {e}")
    
    return None

if __name__ == "__main__":
    check_job_status()