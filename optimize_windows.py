#!/usr/bin/env python3
"""
Simple Windows Ollama optimization without restart.
"""
import os
import requests
import time

def set_environment_variables():
    """Set performance environment variables."""
    env_vars = {
        "OLLAMA_NUM_PARALLEL": "6",
        "OLLAMA_MAX_LOADED_MODELS": "3", 
        "OLLAMA_MAX_QUEUE": "1024",
        "OLLAMA_KEEP_ALIVE": "10m",
        "OLLAMA_GPU_LAYERS": "-1",
        "OLLAMA_FLASH_ATTENTION": "1"
    }
    
    print("⚙️  Setting Ollama environment variables...")
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  {key}={value}")

def test_ollama_performance():
    """Test current Ollama performance."""
    models_to_test = ["qwen2.5-coder:14b", "gemma3:12b"]
    
    print("\n🧪 Testing Ollama performance...")
    
    for model in models_to_test:
        start_time = time.time()
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": "What is CPU monitoring?",
                    "stream": False,
                    "options": {"num_predict": 50}
                },
                timeout=30
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status_code == 200:
                print(f"  ✅ {model}: {duration:.2f}s")
            else:
                print(f"  ❌ {model}: Failed")
                
        except Exception as e:
            print(f"  ❌ {model}: {e}")

if __name__ == "__main__":
    set_environment_variables()
    time.sleep(2)
    test_ollama_performance()
    print("\n✅ Optimization variables set!")
    print("💡 Restart your dataset generation for best performance")