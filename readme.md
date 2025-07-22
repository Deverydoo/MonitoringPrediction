# Distilled LLM for Predictive Monitoring & Troubleshooting

A specialized distilled language model for predictive monitoring and intelligent troubleshooting of Linux systems, IBM Spectrum Conductor, and enterprise environments. Features continual learning, metric adjustment, and multi-model fallback architecture with complete local caching for portability.

## Overview

This system combines traditional system metrics with natural language understanding to provide:
- **Predictive anomaly detection** with continual learning
- **Intelligent troubleshooting assistance** via chat interface  
- **Technology-specific knowledge** distilled from foundational models
- **Self-contained operation** with local model caching
- **Multi-source data integration** from Splunk, Jira, Confluence, and system logs

## Architecture

```
Data Sources                 Model Chain                    Distilled Model
┌─────────────────┐         ┌─────────────────┐           ┌─────────────────┐
│ • Splunk Logs   │         │ 1. Remote API   │           │ • Classification│
│ • Jira Tickets │    ────▶│ 2. Ollama       │    ────▶  │ • Regression    │
│ • Confluence    │         │ 3. Local HF     │           │ • Anomaly Det.  │
│ • Spectrum REST │         │ 4. Static       │           │ • Chat Interface│
│ • VEMKD Logs    │         └─────────────────┘           └─────────────────┘
│ • RedHat Linux  │                  │                             │
└─────────────────┘                  ▼                             ▼
         │                  ┌─────────────────┐           ┌─────────────────┐
         └─────────────────▶│ Training Data   │           │ Continual       │
                            │ Generation      │           │ Learning &      │
                            └─────────────────┘           │ Metric Tuning   │
                                                          └─────────────────┘
```

## Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup Ollama (if using local fallback)
ollama serve
ollama pull deepseek-r1:latest
```

### 2. Jupyter Notebook Interface
```python
# Launch the main notebook
jupyter notebook Distillery.ipynb

# Follow the guided workflow:
from main_notebook import *

# 1. Setup environment and fallback system
setup()

# 2. Generate training datasets (resumable with progress tracking)
generate_datasets(language_count=2000, metrics_count=10000)

# 3. Train the distilled model
train()

# 4. Test inference capabilities
test()

# 5. Run monitoring demo
demo(minutes=5)
```

### 3. Production Usage
```python
from inference_and_monitoring import MonitoringInference, RealTimeMonitor

# Initialize inference engine
inference = MonitoringInference()

# Create real-time monitor
monitor = RealTimeMonitor(inference)

# Process metrics and get predictions
metrics = {
    'cpu_usage': 85.5, 'memory_usage': 92.3,
    'disk_usage': 45.2, 'load_average': 6.8
}
prediction = monitor.process_metrics(metrics)

# Start continuous monitoring
monitor.run_continuous_monitoring(interval_seconds=60)
```

## Key Features

### 🔄 Continual Learning
- **Metric adjustment over time** based on feedback and outcomes
- **Adaptive thresholds** that learn from environment patterns
- **Knowledge updates** from new incidents and resolutions
- **Performance tuning** based on prediction accuracy

### 🗣️ Chat Interface
- **Natural language troubleshooting** for technical issues
- **Context-aware responses** using system-specific knowledge
- **Technical term understanding** including slang and variations
- **Direct, on-topic answers** without unnecessary elaboration

### 📊 Multi-Source Integration
- **Splunk logs** via REST API with custom queries
- **Jira tickets** for incident correlation and learning
- **Confluence documentation** for knowledge base integration
- **IBM Spectrum Conductor** via REST endpoints
- **VEMKD logs** from RedHat Linux systems
- **System metrics** from standard monitoring tools

### 🔗 Multi-Model Fallback Chain
1. **Remote API** (Gemini/Claude) - Primary model access
2. **Ollama Local** (DeepSeek-R1) - Local inference fallback
3. **HuggingFace Cache** - Cached transformer models
4. **Static Responses** - Hardcoded fallback for reliability

### 💾 Local Caching & Portability
- **Complete self-containment** with local model storage
- **Configurable cache directories** for shared storage environments
- **HuggingFace model caching** in project-local directories
- **Portable deployment** across different environments

## Configuration

### Core Settings (`config.py`)
```python
CONFIG = {
    # Model fallback chain
    "llm_url": "https://api.anthropic.com/v1/messages",  # Remote API
    "llm_key": "your-api-key",
    "ollama_url": "http://localhost:11434",              # Local Ollama
    "ollama_model": "deepseek-r1:latest",
    "local_model_path": "./local_models/",               # HF cache fallback
    
    # Cache directories (configurable for shared storage)
    "hf_cache_dir": "./hf_cache/",
    "models_dir": "./models/",
    "training_dir": "./training/",
    
    # Data source endpoints
    "splunk_url": "https://your-splunk-instance",
    "splunk_token": "your-splunk-token",
    "spectrum_rest_url": "https://your-spectrum-cluster/platform/rest",
    "spectrum_auth": {"username": "user", "password": "pass"},
    
    # Training parameters
    "language_samples": 2000,
    "metrics_samples": 10000,
    "models_per_question": 2,  # Multiple models for variety
    "anomaly_ratio": 0.2,
    
    # Model settings
    "model_name": "bert-base-uncased",
    "max_length": 1024,
    "batch_size": 12,
    "epochs": 3
}
```

### Environment-Specific Cache Paths
```python
# For shared storage environments
CONFIG['hf_cache_dir'] = "/shared/ml_models/huggingface/"
CONFIG['models_dir'] = "/shared/ml_models/distilled/"

# For local development
CONFIG['hf_cache_dir'] = "./hf_cache/"
CONFIG['models_dir'] = "./models/"
```

## Training Data Generation

### Language Dataset
Generated by querying foundational models with:
- **Technical explanations** (60%) - System concepts, error meanings
- **Troubleshooting scenarios** (25%) - Step-by-step problem solving  
- **Conversational examples** (15%) - Natural chat interactions

### Metrics Dataset
Synthetic system metrics with:
- **Normal operation patterns** (80%)
- **Realistic anomaly scenarios** (20%)
- **Temporal correlations** and failure progressions
- **Environment-specific patterns** (Spectrum, Linux, Java)

### Progressive Training Process
```python
# Resumable dataset generation with progress tracking
from dataset_generator import DatasetGenerator

generator = DatasetGenerator()
generator.show_progress()  # Check current progress

# Continue from where you left off
language_data, metrics_data = generator.generate_complete_dataset()

# Retry any failed generations
generator.retry_failed()
```

## Data Source Integration

### Splunk Integration
```python
# Custom query examples for VEMKD logs
splunk_queries = {
    "error_logs": 'index=linux sourcetype=syslog "error" OR "exception"',
    "performance": 'index=system source="/var/log/vemkd*" | stats avg(cpu_usage)',
    "spectrum_jobs": 'index=spectrum sourcetype="conductor_logs"'
}

# Automatic data collection
metrics = monitor.collect_from_splunk(splunk_queries["performance"])
```

### IBM Spectrum Conductor
```python
# REST API endpoints for job and resource monitoring
spectrum_endpoints = [
    '/platform/rest/conductor/v1/clusters',
    '/platform/rest/conductor/v1/consumers', 
    '/platform/rest/conductor/v1/resourcegroups',
    '/platform/rest/conductor/v1/workloads'
]

# Automatic integration
spectrum_metrics = monitor.collect_from_spectrum()
```

### Jira & Confluence Learning
```python
# Correlation with incident data for continual learning
incident_patterns = analyze_jira_incidents()
knowledge_updates = extract_confluence_solutions()

# Feed back into model for continuous improvement
model.update_knowledge(incident_patterns, knowledge_updates)
```

## Monitoring & Inference

### Real-Time Anomaly Detection
- **Multi-head neural network** for classification, regression, anomaly scoring
- **Rule-based fallbacks** for reliability when ML models are uncertain
- **Confidence scoring** to determine prediction reliability
- **Actionable recommendations** generated based on detected patterns

### Chat Interface Capabilities
- **Technical troubleshooting** with context from system state
- **Error interpretation** with specific remediation steps
- **Performance guidance** based on current system metrics
- **Knowledge queries** about system administration concepts

### Continual Learning Features
- **Feedback incorporation** from incident outcomes
- **Threshold adaptation** based on environment patterns
- **Model fine-tuning** with new data and scenarios
- **Performance tracking** and automatic improvement

## Directory Structure

```
distilled-monitoring-system/
├── config.py                    # Configuration with cache paths
├── main_notebook.py             # Jupyter interface functions  
├── dataset_generator.py         # Resumable data generation
├── distilled_model_trainer.py   # Multi-task model training
├── inference_and_monitoring.py  # Real-time monitoring engine
├── data_config_manager.py       # YAML configuration management
├── Distillery.ipynb            # Main guided notebook
├── requirements.txt             # Dependencies
├── setup.py                     # Automated setup script
├── hf_cache/                    # Local HuggingFace cache
├── models/                      # Trained distilled models
├── training/                    # Generated training datasets
├── data_config/                 # YAML configuration files
├── logs/                        # Training and inference logs
├── static_responses/            # Fallback response database
└── local_models/               # Local model storage
```

## Advanced Usage

### Custom Model Integration
```python
# Add your own foundational model to the chain
from config import model_chain

def custom_model_query(prompt, max_tokens=300):
    # Your model implementation
    return response

# Register in the model chain
model_chain.add_model("custom", custom_model_query)
```

### Environment-Specific Tuning
```python
# Tune for your specific environment
CONFIG['spectrum_metrics_weight'] = 1.5  # Emphasize Spectrum data
CONFIG['linux_log_patterns'] = ['vemkd', 'systemd', 'kernel']
CONFIG['alert_thresholds'] = {
    'cpu_usage': 85.0,      # Environment-specific thresholds
    'memory_usage': 90.0,
    'spectrum_queue_depth': 100
}
```

### Production Deployment
```python
# Production monitoring setup
from inference_and_monitoring import MonitoringInference, RealTimeMonitor

# Initialize with production configuration
inference = MonitoringInference(model_path="./models/production_model/")
monitor = RealTimeMonitor(inference)

# Setup data source connections
monitor.configure_splunk(CONFIG['splunk_url'], CONFIG['splunk_token'])
monitor.configure_spectrum(CONFIG['spectrum_rest_url'], CONFIG['spectrum_auth'])

# Start continuous monitoring with alerting
monitor.run_continuous_monitoring(
    interval_seconds=30,
    alert_callback=send_alert_to_ops_team,
    learn_from_incidents=True
)
```

## System Requirements

- **Python 3.8+** with PyTorch 2.0+
- **8GB+ RAM** (16GB+ recommended for training)
- **10GB+ disk space** for model cache and datasets
- **GPU support** (optional but recommended for training)
- **Network access** for initial model downloads and data source APIs

## Troubleshooting

### Model Chain Issues
```python
# Test the fallback chain
from config import test_fallback_system
test_fallback_system()

# Check individual components
check_remote_api()  # Test remote model access
check_ollama()      # Test local Ollama
check_hf_cache()    # Test cached models
```

### Training Issues
- **Out of memory**: Reduce `batch_size` in CONFIG
- **Slow training**: Enable GPU support or reduce dataset size
- **Poor performance**: Increase `language_samples` and `epochs`

### Data Source Issues
- **Splunk timeout**: Increase `splunk_timeout` in CONFIG
- **Spectrum auth**: Verify credentials and endpoint URLs
- **Log parsing**: Check log format patterns in data configs

## Contributing

This system is designed for continual improvement:

1. **Add new data sources** by extending the monitoring collectors
2. **Improve training data** by adding domain-specific YAML configs
3. **Enhance the chat interface** with new conversation patterns
4. **Optimize performance** through model architecture improvements

## License

Internal use only. Ensure compliance with Wells Fargo AI and data policies.

---

**Last Updated**: July 2025 | **Version**: 1.0 | **Architecture**: Multi-Model Fallback with Continual Learning