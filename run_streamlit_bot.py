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

    azure_key = os.getenv("AZURE_TRANSLATOR_KEY")
    azure_region = os.getenv("AZURE_TRANSLATOR_REGION")
    if azure_key and azure_region:
        print(f"✅ AZURE_TRANSLATOR_KEY: {azure_key[:8]}... (region: {azure_region})")
    else:
        print("⚠️  AZURE_TRANSLATOR_KEY / AZURE_TRANSLATOR_REGION not set - translation will use fallback text")

    return True

def main():
    """Main function"""
    print("=" * 60)
    print("📚 Regional Language Study Bot - Streamlit Edition")
    print("=" * 60)
    
    if not check_env():
        print("❌ Environment check failed!")
        return
    
    # On Elastic Beanstalk / Docker (and most cloud platforms) the app must
    # bind to 0.0.0.0 and to the port the platform assigns via $PORT, not to
    # localhost:8501 which is only reachable from inside the container.
    port = os.getenv("PORT", "8501")
    address = "0.0.0.0" if os.getenv("PORT") else "localhost"

    print("🚀 Starting Streamlit app...")
    print(f"🌐 Listening on {address}:{port}")
    print("⚠️  Press Ctrl+C to stop the application")
    print("-" * 60)
    
    try:
        # Start streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_study_bot.py",
            "--server.address", address,
            "--server.port", port,
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    main()
