#!/usr/bin/env python3
"""
Ollama optimization script for faster dataset generation.
"""
import os
import subprocess
import requests
import json

def setup_ollama_environment():
    """Set environment variables for optimal Ollama performance."""
    
    # Performance environment variables
    env_vars = {
        "OLLAMA_NUM_PARALLEL": "4",          # Parallel requests per model
        "OLLAMA_MAX_LOADED_MODELS": "2",     # Keep 2 models loaded
        "OLLAMA_MAX_QUEUE": "512",           # Large queue for batch processing
        "OLLAMA_GPU_LAYERS": "-1",           # Use all GPU layers
        "OLLAMA_KEEP_ALIVE": "5m",           # Keep models loaded longer
        "OLLAMA_FLASH_ATTENTION": "1",       # Enable flash attention
        "OLLAMA_GPU_OVERHEAD": "0.1",        # Minimal GPU overhead
        "OLLAMA_HOST": "0.0.0.0:11434"       # Listen on all interfaces
    }
    
    print("🚀 Setting up Ollama environment for optimal performance...")
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  {key}={value}")
    
    return env_vars

def restart_ollama_with_optimization():
    """Restart Ollama with optimization settings - Windows compatible."""
    print("\n🔄 Restarting Ollama with optimizations...")
    
    try:
        import platform
        is_windows = platform.system().lower() == 'windows'
        
        if is_windows:
            # Windows approach - stop Ollama service
            print("  Stopping Ollama on Windows...")
            try:
                # Try to stop via taskkill
                subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], 
                             check=False, capture_output=True)
                subprocess.run(["taskkill", "/F", "/IM", "ollama_llama_server.exe"], 
                             check=False, capture_output=True)
                
                import time
                time.sleep(3)  # Wait for process to stop
                
                print("  ✅ Ollama processes stopped")
            except Exception as e:
                print(f"  ⚠️  Stop attempt: {e}")
            
            # Start Ollama with environment variables
            env_vars = setup_ollama_environment()
            
            # Start new process with optimized environment
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            
            subprocess.Popen(
                ["ollama", "serve"],
                env={**os.environ, **env_vars},
                startupinfo=startupinfo
            )
            
        else:
            # Linux/Mac approach
            subprocess.run(["pkill", "-f", "ollama"], check=False)
            env_vars = setup_ollama_environment()
            subprocess.Popen(["ollama", "serve"], env={**os.environ, **env_vars})
        
        print("✅ Ollama restarted with optimizations")
        return True
        
    except Exception as e:
        print(f"❌ Failed to restart Ollama: {e}")
        print("  💡 Manually restart Ollama and continue")
        return False

def preload_models():
    """Preload your available models into memory for faster access."""
    # Use your actual models from the ollama ls output
    priority_models = [
        "qwen2.5-coder:latest",    # Good for technical content
        "phi4:latest",
        "qwen2.5-coder:14b",       # Your current model
        "gemma3:12b",              # Alternative model
        "qwen3:14b",               # Another option
        "deepseek-r1:latest",       # Fallback model
        "llava:latest"
    ]
    
    print(f"\n📦 Preloading {len(priority_models)} priority models...")
    
    successful_loads = 0
    for model in priority_models:
        try:
            print(f"  Loading {model}...")
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": "test",
                    "stream": False,
                    "keep_alive": "10m",  # Keep loaded for 10 minutes
                    "options": {
                        "num_predict": 1,
                        "temperature": 0.1
                    }
                },
                timeout=45
            )
            
            if response.status_code == 200:
                print(f"    ✅ {model} preloaded and cached")
                successful_loads += 1
            else:
                print(f"    ⚠️  {model} failed to preload (status: {response.status_code})")
                
        except requests.exceptions.Timeout:
            print(f"    ⏰ {model} timeout - model may be loading")
        except Exception as e:
            print(f"    ❌ {model} error: {e}")
    
    print(f"\n✅ {successful_loads}/{len(priority_models)} models preloaded")
    
    # Show what's currently loaded
    try:
        import time
        time.sleep(2)
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
        if result.returncode == 0:
            print("\n📊 Currently loaded models:")
            print(result.stdout)
    except:
        pass

if __name__ == "__main__":
    setup_ollama_environment()
    restart_ollama_with_optimization()
    
    # Wait a moment for Ollama to start
    import time
    time.sleep(5)
    
    preload_models()
    print("\n✅ Ollama optimization complete!")