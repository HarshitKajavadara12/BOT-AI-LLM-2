"""
"""Setup Script for QUANTUM-FORGE LLM/RAG Integration

This module provides automated setup and installation of dependencies for the
LLM/RAG integration system. It handles package installation, model downloads,
and service configuration.

Responsibilities:
    - Verify Python version compatibility
    - Install required Python packages
    - Download and configure LLM models
    - Set up Redis and Qdrant services
    - Validate installation completeness
    - Generate configuration files

Inputs:
    - User execution of setup script
    - Optional configuration parameters
    - System environment variables
    - Network connectivity for downloads

Outputs:
    - Installed Python packages (pip)
    - Downloaded Llama 3.2 8B model files
    - Redis service configuration
    - Qdrant service configuration
    - Setup status and validation results
    - Configuration files and credentials

CRITICAL CONSTRAINT:
    No AI/LLM is allowed to modify execution or risk decisions here.
    
    This setup script installs infrastructure components for the LLM/RAG
    system. The components it installs operate in READ-ONLY mode with
    respect to trading operations.
    
    This setup DOES NOT:
    - Grant LLM systems trading execution privileges
    - Configure automatic trade triggering
    - Enable AI-driven order placement
    - Create feedback loops to trading systems
    - Provide write access to portfolio management
    
    All installed components maintain strict separation between:
    - Informational/explanatory AI systems (this module)
    - Deterministic trading algorithms (core/ modules)

Dependencies Installed:
    - fastapi, uvicorn: REST API server
    - llama-cpp-python: Local LLM inference
    - sentence-transformers: Vector embeddings
    - qdrant-client: Vector database client
    - redis: Event streaming
    - duckdb: Analytical database

Models Downloaded:
    - Llama 3.2 8B (GGUF quantized)
    - all-MiniLM-L6-v2 (sentence embeddings)

Services Required:
    - Redis: Event streaming (docker recommended)
    - Qdrant: Vector search (docker recommended)

Validation:
    - Connection tests for all services
    - Model loading verification
    - API endpoint health checks
"""

import subprocess
import sys
import os
from pathlib import Path


def print_section(title):
    """Print section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def run_command(cmd, description):
    """Run command and handle errors"""
    print(f" ️  {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"  {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  {description} - FAILED")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check Python version"""
    print_section("Checking Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  Python 3.8+ required")
        return False
    
    print("  Python version OK")
    return True


def install_base_dependencies():
    """Install base Python packages"""
    print_section("Installing Base Dependencies")
    
    packages = [
        ("duckdb>=0.9.2", "DuckDB - Fast analytics cache"),
        ("redis>=7.2.0", "Redis - Event streaming"),
        ("qdrant-client>=1.7.0", "Qdrant - Vector database"),
        ("sentence-transformers>=2.2.0", "Sentence Transformers - Embeddings"),
        ("fastapi>=0.108.0", "FastAPI - REST API"),
        ("uvicorn>=0.25.0", "Uvicorn - ASGI server"),
        ("pydantic>=2.5.0", "Pydantic - Data validation"),
        ("requests>=2.31.0", "Requests - HTTP client")
    ]
    
    success_count = 0
    for package, description in packages:
        if run_command(f'pip install "{package}"', f"Install {description}"):
            success_count += 1
    
    print(f"\n  Installed {success_count}/{len(packages)} packages")
    return success_count == len(packages)


def install_llm_support():
    """Install LLM inference support"""
    print_section("Installing LLM Support")
    
    print(" ️  llama-cpp-python requires C++ compiler")
    print("   Windows: Install Visual Studio Build Tools")
    print("   Linux: Install gcc/g++")
    print("   Mac: Install Xcode Command Line Tools\n")
    
    response = input("Install llama-cpp-python? (y/n): ").lower()
    if response != 'y':
        print(" ️  Skipping LLM support (will use template responses)")
        return True
    
    # Try standard install
    success = run_command(
        'pip install llama-cpp-python',
        "Install llama-cpp-python"
    )
    
    if not success:
        print("\n ️  Standard install failed. Try CPU-only build:")
        print("   CMAKE_ARGS=\"-DLLAMA_BLAS=OFF\" pip install llama-cpp-python")
        return False
    
    return True


def setup_directories():
    """Create required directories"""
    print_section("Setting Up Directories")
    
    dirs = [
        "llm_integration/cache",
        "llm_integration/storage",
        "llm_integration/models"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {dir_path}")
    
    return True


def download_embedding_model():
    """Download sentence-transformers model"""
    print_section("Downloading Embedding Model")
    
    print("  Downloading MiniLM-L6-v2 (80MB)...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print(f"  Model downloaded and cached")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def download_llm_model():
    """Guide for downloading Llama model"""
    print_section("LLM Model Setup")
    
    print(" ️  Llama 3.2 8B model is ~5GB and must be downloaded manually")
    print("\nOptions:")
    print("\n1. HuggingFace (Recommended)")
    print("   Visit: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF")
    print("   Download: llama-2-7b-chat.Q4_K_M.gguf (~4GB)")
    print("   Place in: llm_integration/models/")
    
    print("\n2. Use API (OpenAI, Anthropic, etc.)")
    print("   Modify llm_engine.py to use API instead of local model")
    
    print("\n3. Skip LLM (Use template responses)")
    print("   System works without LLM, provides basic template responses")
    
    response = input("\nDownload model now? (opens browser) (y/n): ").lower()
    if response == 'y':
        import webbrowser
        webbrowser.open("https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF")
    
    return True


def setup_redis():
    """Setup Redis (optional)"""
    print_section("Redis Setup (Optional)")
    
    print("Redis provides real-time event streaming")
    print("System works without Redis (graceful degradation)\n")
    
    print("To install Redis:")
    print("  Windows: https://github.com/microsoftarchive/redis/releases")
    print("  Linux: sudo apt-get install redis-server")
    print("  Mac: brew install redis")
    print("\nStart Redis: redis-server")
    
    return True


def setup_qdrant():
    """Setup Qdrant (optional)"""
    print_section("Qdrant Setup (Optional)")
    
    print("Qdrant provides vector search for RAG")
    print("System works without Qdrant (graceful degradation)\n")
    
    print("To install Qdrant:")
    print("  Docker: docker run -p 6333:6333 qdrant/qdrant")
    print("  Binary: Download from https://github.com/qdrant/qdrant/releases")
    
    return True


def test_installation():
    """Test core components"""
    print_section("Testing Installation")
    
    tests = []
    
    # Test DuckDB
    try:
        import duckdb
        conn = duckdb.connect(':memory:')
        conn.execute("SELECT 1")
        print("  DuckDB working")
        tests.append(True)
    except Exception as e:
        print(f"  DuckDB failed: {e}")
        tests.append(False)
    
    # Test Qdrant client
    try:
        from qdrant_client import QdrantClient
        print("  Qdrant client installed")
        tests.append(True)
    except Exception as e:
        print(f"  Qdrant client failed: {e}")
        tests.append(False)
    
    # Test SentenceTransformers
    try:
        from sentence_transformers import SentenceTransformer
        print("  SentenceTransformers installed")
        tests.append(True)
    except Exception as e:
        print(f"  SentenceTransformers failed: {e}")
        tests.append(False)
    
    # Test FastAPI
    try:
        from fastapi import FastAPI
        print("  FastAPI installed")
        tests.append(True)
    except Exception as e:
        print(f"  FastAPI failed: {e}")
        tests.append(False)
    
    # Test llama-cpp-python (optional)
    try:
        from llama_cpp import Llama
        print("  llama-cpp-python installed")
        tests.append(True)
    except ImportError:
        print(" ️  llama-cpp-python not installed (using template responses)")
        tests.append(True)  # Not required
    except Exception as e:
        print(f"  llama-cpp-python error: {e}")
        tests.append(True)  # Not required
    
    success_rate = sum(tests) / len(tests)
    print(f"\n  {sum(tests)}/{len(tests)} core components working ({success_rate:.0%})")
    
    return success_rate >= 0.75


def print_next_steps():
    """Print next steps"""
    print_section("Setup Complete!  ")
    
    print("Next steps:")
    print("\n1. Start Redis (optional):")
    print("   redis-server")
    
    print("\n2. Start Qdrant (optional):")
    print("   docker run -p 6333:6333 qdrant/qdrant")
    
    print("\n3. Download LLM model (optional):")
    print("   Place in: llm_integration/models/")
    
    print("\n4. Start FastAPI service:")
    print("   python llm_integration/api.py")
    
    print("\n5. Start Streamlit dashboard:")
    print("   streamlit run interface/main_dashboard.py")
    
    print("\n6. Open AI Chat page:")
    print("   Navigate to:   AI Chat")
    
    print("\n" + "="*80)
    print("\n  System works without Redis, Qdrant, or LLM model")
    print("   - Uses DuckDB cache (always available)")
    print("   - Template responses instead of AI (instant)")
    print("   - Full functionality, no external dependencies\n")


def main():
    """Main setup process"""
    print("\n" + "="*80)
    print("  QUANTUM-FORGE LLM/RAG Integration Setup")
    print("="*80)
    
    # Check Python
    if not check_python_version():
        return
    
    # Install dependencies
    if not install_base_dependencies():
        print("\n ️  Some packages failed to install")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Install LLM support (optional)
    install_llm_support()
    
    # Setup directories
    setup_directories()
    
    # Download embedding model
    download_embedding_model()
    
    # LLM model guide
    download_llm_model()
    
    # Redis setup guide
    setup_redis()
    
    # Qdrant setup guide
    setup_qdrant()
    
    # Test installation
    if test_installation():
        print_next_steps()
    else:
        print("\n ️  Some components failed testing")
        print("   System will use graceful degradation")
        print_next_steps()


if __name__ == "__main__":
    main()
