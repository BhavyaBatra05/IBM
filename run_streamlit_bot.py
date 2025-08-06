#!/usr/bin/env python3
"""
Simple startup script for Streamlit Regional Language Study Bot
"""

import os
import subprocess
import sys
from pathlib import Path

def load_env():
    """Load environment variables from .env file"""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip('"\'')
                    os.environ[key] = value
        print("📄 Loaded .env file")
    else:
        print("⚠️  .env file not found")

def check_env():
    """Check environment setup"""
    print("🔍 Checking environment...")
    
    load_env()
    
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("❌ GROQ_API_KEY not found! Please set it in .env file")
        return False
    
    print(f"✅ GROQ_API_KEY: {groq_key[:20]}...")
    return True

def main():
    """Main function"""
    print("=" * 60)
    print("📚 Regional Language Study Bot - Streamlit Edition")
    print("=" * 60)
    
    if not check_env():
        print("❌ Environment check failed!")
        return
    
    print("🚀 Starting Streamlit app...")
    print("🌐 The app will open in your browser at: http://localhost:8501")
    print("⚠️  Press Ctrl+C to stop the application")
    print("-" * 60)
    
    try:
        # Start streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_study_bot.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    main()
