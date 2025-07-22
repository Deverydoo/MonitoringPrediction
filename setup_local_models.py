#!/usr/bin/env python3
"""
Setup script for local foundational models as fallback for dataset generation.
Downloads and caches models locally for offline use.
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model recommendations by size and purpose
RECOMMENDED_MODELS = {
    "small": {
        "model": "microsoft/DialoGPT-small",
        "size": "117M parameters",
        "memory": "~1GB",
        "description": "Small conversational model, good for basic responses"
    },
    "medium": {
        "model": "microsoft/DialoGPT-medium", 
        "size": "345M parameters",
        "memory": "~2GB",
        "description": "Medium conversational model, balanced performance"
    },
    "large": {
        "model": "microsoft/DialoGPT-large",
        "size": "762M parameters", 
        "memory": "~4GB",
        "description": "Large conversational model, better quality responses"
    },
    "technical": {
        "model": "microsoft/CodeBERT-base",
        "size": "125M parameters",
        "memory": "~1GB", 
        "description": "Technical/code-focused model for system administration"
    },
    "gpt2": {
        "model": "gpt2",
        "size": "124M parameters",
        "memory": "~1GB",
        "description": "General purpose GPT-2 model, widely compatible"
    },
    "gpt2-medium": {
        "model": "gpt2-medium",
        "size": "345M parameters",
        "memory": "~2GB",
        "description": "Medium GPT-2 model, better text generation"
    },
    "distilgpt2": {
        "model": "distilgpt2",
        "size": "82M parameters",
        "memory": "~500MB",
        "description": "Distilled GPT-2, very lightweight and fast"
    }
}

def check_system_requirements():
    """Check system requirements for local models."""
    print("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check available memory
    try:
        import psutil
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        
        print(f"📊 System Memory: {available_memory:.1f}GB available / {total_memory:.1f}GB total")
        
        if available_memory < 2:
            print("⚠️  Low memory detected (<2GB available)")
            print("   Consider using 'small' or 'distilgpt2' models")
        
    except ImportError:
        print("⚠️  psutil not installed, cannot check memory")
    
    # Check for GPU
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"🎮 GPU Available: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("💻 GPU not available, will use CPU")
    
    # Check Apple Silicon
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🍎 Apple Silicon (MPS) available")
    
    # Check disk space
    try:
        disk_usage = psutil.disk_usage('.')
        free_space = disk_usage.free / (1024**3)  # GB
        print(f"💾 Disk Space: {free_space:.1f}GB free")
        
        if free_space < 10:
            print("⚠️  Low disk space (<10GB free)")
            print("   Models can be 1-5GB each")
            
    except:
        print("⚠️  Cannot check disk space")
    
    print("✅ System requirements checked")
    return True

def install_dependencies():
    """Install required dependencies for local models."""
    print("\nInstalling dependencies...")
    
    import subprocess
    
    requirements = [
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "accelerate>=0.12.0",
        "psutil>=5.8.0",
        "bitsandbytes>=0.37.0",  # For quantization
        "sentencepiece>=0.1.96"  # For some tokenizers
    ]
    
    for requirement in requirements:
        try:
            print(f"Installing {requirement.split('>=')[0]}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", requirement
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to install {requirement}: {e}")
    
    print("✅ Dependencies installed")

def display_model_options():
    """Display available model options."""
    print("\n🤖 AVAILABLE LOCAL MODELS")
    print("=" * 50)
    
    for key, info in RECOMMENDED_MODELS.items():
        print(f"\n{key.upper()}:")
        print(f"  Model: {info['model']}")
        print(f"  Size: {info['size']}")
        print(f"  Memory: {info['memory']}")
        print(f"  Description: {info['description']}")
    
    print("\n💡 RECOMMENDATIONS:")
    print("  • For 4GB+ RAM: 'medium' or 'gpt2-medium'")
    print("  • For 2GB+ RAM: 'small' or 'gpt2'")
    print("  • For <2GB RAM: 'distilgpt2'")
    print("  • For technical content: 'technical'")
    print("  • For GPU users: Any model with quantization")

def download_model(model_name: str, local_path: str, quantization: str = "none"):
    """Download and cache a model locally."""
    print(f"\nDownloading {model_name}...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Create local directory
        safe_name = model_name.replace('/', '_')
        model_dir = Path(local_path) / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📦 Downloading to {model_dir}")
        
        # Download tokenizer
        print("  Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Download model with appropriate settings
        print("  Downloading model...")
        model_kwargs = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "trust_remote_code": False
        }
        
        if quantization == "8bit":
            model_kwargs["load_in_8bit"] = True
        elif quantization == "4bit":
            model_kwargs["load_in_4bit"] = True
        
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Save locally
        print("  Saving locally...")
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
        
        # Test the model
        print("  Testing model...")
        test_prompt = "System monitoring is"
        inputs = tokenizer.encode(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        test_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Test response: {test_response}")
        
        # Save model info
        model_info = {
            "model_name": model_name,
            "local_path": str(model_dir),
            "quantization": quantization,
            "downloaded_at": str(torch.utils.data.dataloader.default_collate([])),
            "test_successful": True
        }
        
        with open(model_dir / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Model {model_name} downloaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False

def list_downloaded_models(local_path: str):
    """List already downloaded models."""
    print(f"\n📚 DOWNLOADED MODELS IN {local_path}")
    print("=" * 50)
    
    model_path = Path(local_path)
    if not model_path.exists():
        print("No models downloaded yet")
        return []
    
    downloaded_models = []
    for model_dir in model_path.iterdir():
        if model_dir.is_dir():
            info_file = model_dir / "model_info.json"
            if info_file.exists():
                try:
                    with open(info_file, "r") as f:
                        model_info = json.load(f)
                    
                    print(f"\n✅ {model_info['model_name']}")
                    print(f"   Path: {model_info['local_path']}")
                    print(f"   Quantization: {model_info['quantization']}")
                    downloaded_models.append(model_info)
                    
                except Exception as e:
                    print(f"⚠️  {model_dir.name}: Error reading info - {e}")
            else:
                print(f"⚠️  {model_dir.name}: No model info found")
    
    return downloaded_models

def create_local_model_config(model_info: Dict, config_path: str = "./local_model_config.json"):
    """Create configuration file for local models."""
    config = {
        "local_model_enabled": True,
        "local_model_path": str(Path(model_info["local_path"]).parent),
        "local_model_name": model_info["model_name"],
        "local_model_quantization": model_info["quantization"],
        "local_model_device": "auto",
        "local_model_max_tokens": 150,
        "local_model_temperature": 0.7,
        "local_model_cache_responses": True,
        "created_at": str(torch.utils.data.dataloader.default_collate([])),
        "model_info": model_info
    }
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Local model config saved to {config_path}")

def test_local_model(model_path: str):
    """Test a local model."""
    print(f"\n🧪 Testing local model at {model_path}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Load model
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Test prompts
        test_prompts = [
            "Explain what high CPU usage means",
            "What causes OutOfMemoryError in Java",
            "How to troubleshoot network issues",
            "Linux system monitoring tools include"
        ]
        
        print("\nRunning test queries...")
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt}")
            
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Response: {response}")
        
        print("\n✅ Local model test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Local model test failed: {e}")
        return False

def setup_static_fallback(static_path: str = "./static_responses/"):
    """Setup comprehensive static fallback responses."""
    print(f"\n📝 Setting up static fallback responses in {static_path}")
    
    static_dir = Path(static_path)
    static_dir.mkdir(parents=True, exist_ok=True)
    
    # Comprehensive static responses
    static_responses = {
        "technical_explanations": {
            "cpu_usage": "CPU usage represents the percentage of processing power currently being used by the system. High CPU usage (>80%) may indicate heavy computational tasks, inefficient processes, or system stress. Monitor with tools like top, htop, or sar.",
            
            "memory_usage": "Memory usage shows how much RAM is being consumed by processes. High memory usage (>85%) can lead to swapping, performance degradation, and out-of-memory conditions. Use free, ps, or /proc/meminfo to monitor.",
            
            "disk_usage": "Disk usage indicates the percentage of storage space being used on filesystems. High disk usage (>90%) can cause application failures and system instability. Monitor with df, du, or lsblk commands.",
            
            "load_average": "Load average represents the average system load over 1, 5, and 15 minute periods. Values above the number of CPU cores indicate system stress. A load of 1.0 on a single-core system means 100% utilization.",
            
            "systemd": "systemd is a system and service manager for Linux operating systems. It manages system processes, services, and system initialization. Use systemctl to manage services and journalctl to view logs.",
            
            "garbage_collection": "Garbage collection (GC) is the process of automatically reclaiming memory occupied by objects that are no longer in use. In Java, excessive GC can cause performance issues. Monitor with jstat or enable GC logging.",
            
            "swap_usage": "Swap usage indicates when the system is using disk space as virtual memory. High swap usage suggests memory pressure and can severely impact performance. Monitor with free, swapon, or /proc/swaps.",
            
            "inode_usage": "Inodes are data structures that store file metadata. High inode usage can prevent file creation even when disk space is available. Check with df -i and find files with find / -type f | wc -l.",
            
            "network_io": "Network I/O measures data transfer rates over network interfaces. High network I/O may indicate heavy traffic, network congestion, or application issues. Monitor with netstat, ss, or iftop.",
            
            "disk_io": "Disk I/O measures read/write operations per second (IOPS) and throughput. High disk I/O can indicate bottlenecks. Monitor with iostat, iotop, or /proc/diskstats."
        },
        
        "error_interpretations": {
            "OutOfMemoryError": "OutOfMemoryError occurs when the JVM cannot allocate an object due to insufficient memory. Check heap settings (-Xmx), look for memory leaks, analyze heap dumps with jmap, and consider increasing available memory.",
            
            "connection_refused": "Connection refused typically means the target service is not running, not accepting connections on the specified port, or blocked by a firewall. Check service status, port availability with netstat, and firewall rules.",
            
            "disk_full": "Disk full error occurs when the filesystem has no available space. Clean up unnecessary files, check for large files with du, implement log rotation, or expand storage capacity.",
            
            "permission_denied": "Permission denied indicates insufficient privileges to access a resource. Check file permissions with ls -l, verify user/group ownership, and ensure proper sudo/root access where needed.",
            
            "segmentation_fault": "Segmentation fault occurs when a program tries to access restricted memory. This usually indicates a bug in the application code, corrupted memory, or incompatible libraries. Check core dumps and system logs.",
            
            "kernel_panic": "Kernel panic is a fatal system error causing immediate shutdown. Causes include hardware failures, driver issues, or kernel bugs. Check system logs, hardware diagnostics, and recent kernel updates.",
            
            "network_unreachable": "Network unreachable indicates routing issues or network connectivity problems. Check network configuration, routing tables with route or ip route, and network interfaces with ip addr.",
            
            "timeout": "Timeout errors occur when operations take longer than expected. Check network connectivity, system load, resource availability, and consider increasing timeout values where appropriate."
        },
        
        "troubleshooting_scenarios": {
            "high_cpu": "For high CPU usage: 1) Use top/htop to identify CPU-intensive processes, 2) Check for runaway processes or infinite loops, 3) Analyze process behavior with strace, 4) Consider process optimization or horizontal scaling, 5) Review recent deployments or changes.",
            
            "memory_leak": "For memory leaks: 1) Monitor memory usage over time with free/top, 2) Generate heap dumps with jmap for Java applications, 3) Analyze object references and retention, 4) Review application logs for memory-related errors, 5) Consider service restart as temporary fix.",
            
            "slow_performance": "For slow performance: 1) Check system resources (CPU, memory, disk, network), 2) Identify bottlenecks with profiling tools, 3) Review recent changes or deployments, 4) Monitor key performance metrics, 5) Analyze application and system logs.",
            
            "disk_space_issues": "For disk space problems: 1) Use df to check filesystem usage, 2) Find large files with find / -size +100M, 3) Check log files in /var/log, 4) Clean temporary files in /tmp, 5) Implement log rotation policies, 6) Consider storage expansion.",
            
            "network_connectivity": "For network issues: 1) Test connectivity with ping/traceroute, 2) Check interface status with ip addr, 3) Verify routing with ip route, 4) Check firewall rules with iptables -L, 5) Monitor network traffic with tcpdump or wireshark.",
            
            "service_startup_failure": "For service startup failures: 1) Check service status with systemctl status, 2) Review logs with journalctl -u service, 3) Verify configuration files, 4) Check dependencies and prerequisites, 5) Test manual startup and review error messages."
        },
        
        "best_practices": {
            "monitoring": "Effective monitoring includes: 1) Set up proactive alerts for key metrics, 2) Monitor trends not just current values, 3) Use appropriate thresholds to avoid alert fatigue, 4) Implement comprehensive logging, 5) Regular review and tuning of monitoring rules.",
            
            "performance_tuning": "Performance tuning best practices: 1) Establish baseline performance metrics, 2) Make incremental changes and measure impact, 3) Focus on bottlenecks first, 4) Test changes in non-production environments, 5) Document all changes and their effects.",
            
            "security_hardening": "Security hardening includes: 1) Keep systems updated with security patches, 2) Use principle of least privilege, 3) Implement proper firewall rules, 4) Regular security audits and vulnerability scans, 5) Monitor for suspicious activities.",
            
            "backup_recovery": "Backup and recovery: 1) Implement regular automated backups, 2) Test restore procedures regularly, 3) Store backups in multiple locations, 4) Document recovery procedures, 5) Monitor backup success and failures."
        }
    }
    
    # Save static responses
    with open(static_dir / "static_responses.json", "w") as f:
        json.dump(static_responses, f, indent=2)
    
    print(f"✅ Static fallback responses created ({len(static_responses)} categories)")
    
    # Create usage instructions
    usage_instructions = """
# Static Fallback Usage

This directory contains static fallback responses for when all AI models fail.

## Structure:
- static_responses.json: Main response database
- Keywords are matched against prompts
- Responses provide basic but accurate information

## Categories:
- technical_explanations: System concepts and metrics
- error_interpretations: Common error messages and solutions
- troubleshooting_scenarios: Step-by-step troubleshooting guides
- best_practices: General best practices and recommendations

## Usage:
The system automatically falls back to these responses when:
1. Remote LLM is unavailable
2. Ollama is not running
3. Local model fails to load
4. All other fallback methods fail

## Customization:
Edit static_responses.json to add more keywords and responses
specific to your environment and use cases.
"""
    
    with open(static_dir / "README.md", "w") as f:
        f.write(usage_instructions)
    
    return static_responses

def main():
    """Main setup function."""
    print("🚀 LOCAL MODEL SETUP FOR DISTILLED MONITORING SYSTEM")
    print("=" * 60)
    
    # Check system
    if not check_system_requirements():
        sys.exit(1)
    
    # Install dependencies
    install_dependencies()
    
    # Show options
    display_model_options()
    
    # Get user preferences
    print("\n" + "="*50)
    print("SETUP OPTIONS")
    print("="*50)
    
    local_path = input("Local model storage path (default: ./local_models/): ").strip()
    if not local_path:
        local_path = "./local_models/"
    
    # List existing models
    existing_models = list_downloaded_models(local_path)
    
    if existing_models:
        print(f"\nFound {len(existing_models)} existing models.")
        use_existing = input("Use existing model? (y/n): ").strip().lower()
        if use_existing == 'y':
            # Let user choose existing model
            if len(existing_models) == 1:
                selected_model = existing_models[0]
            else:
                print("\nSelect model:")
                for i, model in enumerate(existing_models, 1):
                    print(f"{i}. {model['model_name']}")
                
                choice = input("Choose model (1-{}): ".format(len(existing_models))).strip()
                try:
                    selected_model = existing_models[int(choice) - 1]
                except (ValueError, IndexError):
                    print("Invalid choice, using first model")
                    selected_model = existing_models[0]
            
            # Test existing model
            if test_local_model(selected_model['local_path']):
                create_local_model_config(selected_model)
                print("✅ Setup completed with existing model")
                return
    
    # Download new model
    print("\nWhich model would you like to download?")
    print("Recommendations based on your system:")
    
    # Simple recommendation logic
    try:
        import psutil
        available_memory = psutil.virtual_memory().available / (1024**3)
        if available_memory > 4:
            print("  Recommended: 'medium' or 'gpt2-medium' (you have plenty of RAM)")
        elif available_memory > 2:
            print("  Recommended: 'small' or 'gpt2' (moderate RAM)")
        else:
            print("  Recommended: 'distilgpt2' (limited RAM)")
    except:
        print("  Recommended: 'small' or 'gpt2' (safe choice)")
    
    model_choice = input("\nEnter model key (e.g., 'medium', 'gpt2'): ").strip().lower()
    
    if model_choice not in RECOMMENDED_MODELS:
        print(f"Unknown model '{model_choice}', using 'medium' as default")
        model_choice = "medium"
    
    model_name = RECOMMENDED_MODELS[model_choice]["model"]
    
    # Quantization option
    if torch.cuda.is_available():
        print("\nQuantization options (for GPU users):")
        print("  none: Full precision (best quality)")
        print("  8bit: 8-bit quantization (half memory)")
        print("  4bit: 4-bit quantization (quarter memory)")
        quantization = input("Choose quantization (none/8bit/4bit): ").strip().lower()
        if quantization not in ["none", "8bit", "4bit"]:
            quantization = "none"
    else:
        quantization = "none"
    
    # Download model
    print(f"\nDownloading {model_name} with {quantization} quantization...")
    if download_model(model_name, local_path, quantization):
        # Create config
        model_info = {
            "model_name": model_name,
            "local_path": str(Path(local_path) / model_name.replace('/', '_')),
            "quantization": quantization
        }
        create_local_model_config(model_info)
        
        # Test model
        test_local_model(model_info["local_path"])
        
        print("✅ Local model setup completed successfully!")
    else:
        print("❌ Model download failed")
    
    # Setup static fallback
    print("\n" + "="*50)
    print("STATIC FALLBACK SETUP")
    print("="*50)
    
    setup_static = input("Setup static fallback responses? (y/n): ").strip().lower()
    if setup_static == 'y':
        static_path = input("Static responses path (default: ./static_responses/): ").strip()
        if not static_path:
            static_path = "./static_responses/"
        
        setup_static_fallback(static_path)
        print("✅ Static fallback setup completed!")
    
    # Final summary
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nYour fallback system now includes:")
    print("  1. 🌐 Remote LLM (if configured)")
    print("  2. 🦙 Ollama (if available)")
    print("  3. 💻 Local Model (just set up)")
    print("  4. 📝 Static Responses (if enabled)")
    
    print("\nNext steps:")
    print("  1. Update your config.py with the new local model settings")
    print("  2. Test the complete fallback system")
    print("  3. Start generating your datasets!")
    
    print("\nConfiguration files created:")
    print("  - local_model_config.json")
    print("  - static_responses/ (if enabled)")

if __name__ == "__main__":
    main()