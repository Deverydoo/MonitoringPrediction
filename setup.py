#!/usr/bin/env python3
# setup.py - Installation and setup script for Distilled Monitoring System

import os
import sys
import subprocess
import requests
import json
import platform
from pathlib import Path

def print_banner():
    """Print setup banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                  DISTILLED MONITORING SYSTEM                ║
║                     Setup & Installation                    ║
║                                                              ║
║  Enhanced with Ollama Llama 3.2 Fallback Support          ║
╚══════════════════════════════════════════════════════════════╝
""")

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_requirements():
    """Install Python requirements."""
    print("\nInstalling Python dependencies...")
    
    requirements = [
        "torch>=1.9.0,<2.0",
        "transformers[torch]>=4.20.0,<5.0",
        "datasets>=2.0.0",
        "numpy>=1.21.0,<1.25",
        "requests>=2.25.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "jupyter>=1.0.0",
        "ipywidgets>=7.6.0",
        "notebook>=6.4.0",
        "protobuf>=3.19.0,<3.21.0",
        "psutil>=5.8.0",
        "tqdm>=4.62.0"
    ]
    
    try:
        for requirement in requirements:
            print(f"Installing {requirement.split('>=')[0]}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", requirement
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("✅ All Python dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        print("Please install manually using: pip install -r requirements.txt")
        return False

def check_ollama_installation():
    """Check if Ollama is installed."""
    print("\nChecking Ollama installation...")
    
    try:
        # Try to run ollama command
        result = subprocess.run(
            ["ollama", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode == 0:
            print(f"✅ Ollama installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama not found in PATH")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Ollama not installed")
        return False

def install_ollama():
    """Provide instructions to install Ollama."""
    print("\n" + "="*50)
    print("OLLAMA INSTALLATION REQUIRED")
    print("="*50)
    
    system = platform.system().lower()
    
    if system == "darwin":  # macOS
        print("""
To install Ollama on macOS:

1. Visit https://ollama.ai
2. Download the macOS installer
3. Run the installer
4. Or use Homebrew: brew install ollama

After installation:
1. Start Ollama: ollama serve
2. Pull the model: ollama pull llama3.2:latest
""")
    
    elif system == "linux":
        print("""
To install Ollama on Linux:

1. Run the install script:
   curl -fsSL https://ollama.ai/install.sh | sh

2. Start Ollama:
   ollama serve

3. Pull the model:
   ollama pull llama3.2:latest
""")
    
    elif system == "windows":
        print("""
To install Ollama on Windows:

1. Visit https://ollama.ai
2. Download the Windows installer
3. Run the installer
4. Open Command Prompt or PowerShell

After installation:
1. Start Ollama: ollama serve
2. Pull the model: ollama pull llama3.2:latest
""")
    
    else:
        print("""
To install Ollama:

1. Visit https://ollama.ai
2. Follow the installation instructions for your system
3. Start Ollama: ollama serve
4. Pull the model: ollama pull llama3.2:latest
""")

def check_ollama_running():
    """Check if Ollama server is running."""
    print("\nChecking if Ollama server is running...")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama server is running")
            return True
        else:
            print("❌ Ollama server not responding")
            return False
            
    except requests.exceptions.RequestException:
        print("❌ Ollama server not running")
        return False

def check_ollama_model():
    """Check if llama3.2:latest model is available."""
    print("Checking for llama3.2:latest model...")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if 'llama3.2:latest' in model_names:
                print("✅ llama3.2:latest model available")
                return True
            else:
                print("❌ llama3.2:latest model not found")
                print(f"Available models: {model_names}")
                return False
        else:
            print("❌ Could not check models")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error checking models: {e}")
        return False

def pull_ollama_model():
    """Pull the llama3.2:latest model."""
    print("\nPulling llama3.2:latest model...")
    print("This may take several minutes depending on your internet connection...")
    
    try:
        response = requests.post(
            "http://localhost:11434/api/pull",
            json={"name": "llama3.2:latest"},
            stream=True,
            timeout=600  # 10 minutes timeout
        )
        
        if response.status_code == 200:
            print("Downloading model... (this may take a while)")
            
            for line in response.iter_lines():
                if line:
                    try:
                        progress = json.loads(line.decode('utf-8'))
                        if 'status' in progress:
                            # Show progress updates
                            status = progress['status']
                            if 'completed' in progress and 'total' in progress:
                                completed = progress['completed']
                                total = progress['total']
                                percent = (completed / total) * 100
                                print(f"\r{status}: {percent:.1f}%", end='', flush=True)
                            else:
                                print(f"\r{status}", end='', flush=True)
                    except json.JSONDecodeError:
                        pass
            
            print("\n✅ Model pulled successfully!")
            return True
        else:
            print(f"❌ Failed to pull model. Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error pulling model: {e}")
        return False

def setup_ollama():
    """Complete Ollama setup process."""
    print("\n" + "="*50)
    print("OLLAMA SETUP")
    print("="*50)
    
    # Check if Ollama is installed
    if not check_ollama_installation():
        install_ollama()
        print("\nPlease install Ollama and run this setup again.")
        return False
    
    # Check if Ollama server is running
    if not check_ollama_running():
        print("\nPlease start Ollama server:")
        print("Run: ollama serve")
        print("Then run this setup again.")
        return False
    
    # Check if model is available
    if not check_ollama_model():
        print("\nAttempting to pull llama3.2:latest model...")
        if not pull_ollama_model():
            print("\nPlease manually pull the model:")
            print("Run: ollama pull llama3.2:latest")
            return False
    
    print("\n✅ Ollama setup complete!")
    return True

def test_ollama_connection():
    """Test Ollama connection with a simple query."""
    print("\nTesting Ollama connection...")
    
    try:
        payload = {
            "model": "llama3.2:latest",
            "prompt": "What is system monitoring?",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 50
            }
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', '').strip()
            print("✅ Ollama test successful!")
            print(f"Sample response: {answer[:100]}...")
            return True
        else:
            print(f"❌ Ollama test failed. Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Ollama test error: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\nCreating project directories...")
    
    directories = [
        "training",
        "models",
        "checkpoints",
        "logs",
        "hf_cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}/")
    
    print("✅ All directories created")

def create_config_file():
    """Create a default configuration file."""
    print("\nCreating configuration file...")
    
    config_content = '''# config.json - User configuration
{
  "model_name": "distilbert-base-uncased",
  "max_length": 512,
  "batch_size": 16,
  "learning_rate": 2e-5,
  "epochs": 3,
  "language_samples": 1000,
  "metrics_samples": 5000,
  "anomaly_ratio": 0.2,
  
  "ollama_enabled": true,
  "ollama_url": "http://localhost:11434",
  "ollama_model": "llama3.2:latest",
  "ollama_timeout": 60,
  "ollama_max_tokens": 1000,
  "ollama_temperature": 0.7,
  
  "llm_url": "",
  "llm_key": "",
  "llm_timeout": 30,
  
  "use_cuda": true,
  "force_cpu": false,
  
  "training_dir": "./training/",
  "models_dir": "./models/",
  "checkpoints_dir": "./checkpoints/",
  "logs_dir": "./logs/",
  "hf_cache_dir": "./hf_cache/"
}'''
    
    with open("config.json", "w") as f:
        f.write(config_content)
    
    print("✅ Configuration file created: config.json")

def create_jupyter_startup():
    """Create a Jupyter startup notebook."""
    print("\nCreating Jupyter startup notebook...")
    
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Distilled Monitoring System\n",
                    "\n",
                    "## Quick Start Guide\n",
                    "\n",
                    "This notebook provides a complete environment for the Distilled Monitoring System with Ollama Llama 3.2 fallback support.\n",
                    "\n",
                    "### Getting Started\n",
                    "\n",
                    "Run the cells below in order to set up and use the system."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Import the main system\n",
                    "from main_notebook import *\n",
                    "\n",
                    "# Display quick start guide\n",
                    "quick_start_guide()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 1: Setup environment\n",
                    "setup()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 2: Check LLM status\n",
                    "check_llm()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 3: Test the foundational model\n",
                    "test_llm()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 4: Generate training datasets (this will take time)\n",
                    "# Warning: This step can take 30+ minutes depending on your system\n",
                    "generate_datasets()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 5: Train the model (this will take time)\n",
                    "# Warning: This step can take 1-4 hours depending on your hardware\n",
                    "train()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 6: Test model inference\n",
                    "test()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 7: Run monitoring demo\n",
                    "demo(minutes=5)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Check system status anytime\n",
                    "status()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Additional Commands\n",
                    "\n",
                    "- `setup_ollama()` - Quick Ollama setup\n",
                    "- `continue_train()` - Continue training from checkpoint\n",
                    "- `ollama_guide()` - Ollama installation guide\n",
                    "- `runner.show_status()` - Detailed system status\n",
                    "\n",
                    "## Configuration\n",
                    "\n",
                    "Modify settings in `config.json` or directly in the CONFIG dictionary:\n",
                    "\n",
                    "```python\n",
                    "from config import CONFIG\n",
                    "print(CONFIG)\n",
                    "```"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open("DistilleryQuickStart.ipynb", "w") as f:
        json.dump(notebook_content, f, indent=2)
    
    print("✅ Jupyter notebook created: DistilleryQuickStart.ipynb")

def create_readme():
    """Create a comprehensive README file."""
    print("\nCreating README file...")
    
    readme_content = '''# Distilled Monitoring System with Ollama Fallback

A specialized distilled language model for predictive monitoring of Linux systems and IBM Spectrum Conductor environments, enhanced with local Ollama Llama 3.2 fallback support.

## Features

- **Local LLM Fallback**: Uses Ollama Llama 3.2 when no remote LLM is configured
- **Distilled Language Model**: Lightweight model trained on system administration knowledge
- **Multi-source Data Integration**: Combines system metrics, logs, and domain expertise
- **Predictive Anomaly Detection**: ML-based anomaly detection with rule-based fallbacks
- **Real-time Monitoring**: Continuous monitoring with configurable alerts
- **Spectrum Conductor Integration**: Specialized support for IBM Spectrum Conductor

## Quick Start

### 1. Run Setup Script

```bash
python setup.py
```

This will:
- Check Python version compatibility
- Install required dependencies
- Check/setup Ollama installation
- Create necessary directories
- Create configuration files

### 2. Start Jupyter Notebook

```bash
jupyter notebook DistilleryQuickStart.ipynb
```

### 3. Follow the Notebook

The notebook will guide you through:
1. Environment setup
2. LLM connectivity testing
3. Dataset generation
4. Model training
5. Inference testing
6. Monitoring demo

## Ollama Setup

If you don't have Ollama installed:

1. **Install Ollama**: Visit https://ollama.ai
2. **Start server**: `ollama serve`
3. **Pull model**: `ollama pull llama3.2:latest`

The system will automatically fall back to Ollama when no remote LLM is configured.

## Manual Installation

If the setup script doesn't work:

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama (in another terminal)
ollama serve

# Pull the model
ollama pull llama3.2:latest

# Start Jupyter
jupyter notebook
```

## Configuration

Edit `config.json` to customize:

```json
{
  "ollama_enabled": true,
  "ollama_model": "llama3.2:latest",
  "ollama_url": "http://localhost:11434",
  "model_name": "distilbert-base-uncased",
  "epochs": 3,
  "batch_size": 16
}
```

## System Requirements

- Python 3.8+
- 8GB+ RAM (16GB+ recommended for training)
- 10GB+ free disk space
- CUDA-compatible GPU (optional but recommended)
- Ollama installation for LLM fallback

## Troubleshooting

### Ollama Issues
- **Connection refused**: Make sure `ollama serve` is running
- **Model not found**: Run `ollama pull llama3.2:latest`
- **Slow responses**: Increase `ollama_timeout` in config

### Training Issues
- **Out of memory**: Reduce `batch_size` in config
- **CUDA errors**: Set `force_cpu: true` in config
- **Slow training**: Enable GPU or reduce dataset size

### Dataset Generation Issues
- **Empty responses**: Check Ollama connectivity
- **Timeouts**: Increase `ollama_timeout`
- **Quality issues**: Try different temperature settings

## Usage Examples

```python
# Import the system
from main_notebook import *

# Setup environment
setup()

# Check LLM status
check_llm()

# Generate datasets
generate_datasets()

# Train model
train()

# Test inference
test()

# Run demo
demo(minutes=5)
```

## Architecture

```
Remote LLM (Optional) ──┐
                        ├─► Dataset Generation ──► Model Training ──► Inference
Ollama Fallback ────────┘
```

The system prioritizes remote LLMs but seamlessly falls back to local Ollama for self-contained operation.

## Support

1. Check the troubleshooting section above
2. Review logs in `./logs/` directory
3. Test with `check_llm()` and `test_llm()`
4. Verify Ollama with `ollama list`

For detailed documentation, see the generated notebooks and code comments.
'''
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    print("✅ README file created: README.md")

def check_gpu_support():
    """Check for GPU support."""
    print("\nChecking GPU support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA GPU available: {gpu_name} ({gpu_count} devices)")
            return True
        else:
            print("⚠️  No CUDA GPU detected - will use CPU training")
            return False
    except ImportError:
        print("⚠️  PyTorch not yet installed - GPU check will be done later")
        return False

def main():
    """Main setup function."""
    print_banner()
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Install Python requirements
    print("\n" + "="*50)
    print("INSTALLING PYTHON DEPENDENCIES")
    print("="*50)
    
    if not install_requirements():
        sys.exit(1)
    
    # Step 3: Check GPU support
    check_gpu_support()
    
    # Step 4: Setup Ollama
    ollama_success = setup_ollama()
    
    # Step 5: Create project structure
    print("\n" + "="*50)
    print("CREATING PROJECT STRUCTURE")
    print("="*50)
    
    create_directories()
    create_config_file()
    create_jupyter_startup()
    create_readme()
    
    # Step 6: Test Ollama if available
    if ollama_success:
        test_ollama_connection()
    
    # Final summary
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    
    print("✅ Python dependencies installed")
    print("✅ Project directories created")
    print("✅ Configuration files created")
    print("✅ Jupyter notebook created")
    
    if ollama_success:
        print("✅ Ollama setup and tested")
    else:
        print("⚠️  Ollama setup incomplete")
    
    print("\nNext steps:")
    print("1. Start Jupyter: jupyter notebook DistilleryQuickStart.ipynb")
    print("2. Follow the notebook instructions")
    print("3. If Ollama setup failed, run: ollama serve && ollama pull llama3.2:latest")
    
    print("\nFiles created:")
    print("  - DistilleryQuickStart.ipynb (Main notebook)")
    print("  - config.json (Configuration)")
    print("  - README.md (Documentation)")
    print("  - requirements.txt (Dependencies)")
    
    if not ollama_success:
        print("\n⚠️  Warning: Ollama not fully set up")
        print("The system will still work but may use fallback responses")
        print("for dataset generation until Ollama is properly configured.")

if __name__ == "__main__":
    main()