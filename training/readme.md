# training Directory

## Overview

The `training/` directory contains the generated JSON datasets used to train the distilled monitoring model. These files are created by processing the YAML configurations in `data_config/` through multiple foundational models (Ollama, remote APIs) to create comprehensive training data.

## Generated Files

### Core Dataset Files
- **`language_dataset.json`** - Natural language training samples for chat interface
- **`metrics_dataset.json`** - System metrics and anomaly detection training data
- **`generation_progress.pkl`** - Binary progress tracking for resumable generation

### Backup/Archive Files (if present)
- **`language_dataset_backup_YYYYMMDD.json`** - Automatic backups before major updates
- **`metrics_dataset_backup_YYYYMMDD.json`** - Metrics data backups
- **`generation_logs/`** - Detailed generation logs and statistics

## File Structure Details

### language_dataset.json
Contains natural language samples for the chat interface training:

```json
{
  "metadata": {
    "generated_at": "2025-07-20T10:30:00",
    "generator": "optimized_v2_smart_resume",
    "session_id": "session_20250720_103000",
    "total_samples": 5640,
    "new_samples_added": 3241,
    "config_hash": "abc123...",
    "sample_distribution": {
      "technical_explanation": 2542,
      "error_troubleshooting": 66,
      "conversation": 280,
      "personality_response": 16,
      "question_style_variation": 20,
      "english_pattern": 192,
      "project_management": 1250,
      "banking_scenario": 1250,
      "metrics_explanation": 24
    },
    "model_distribution": {
      "qwen2.5-coder:14b": 1128,
      "gemma3:12b": 987,
      "deepseek-r1:latest": 856,
      "phi4:latest": 743
    },
    "rotation_info": {
      "total_models": 19,
      "model_pool": ["qwen2.5-coder:14b", "gemma3:12b", ...],
      "swap_interval": 25,
      "questions_processed": 3241
    }
  },
  "samples": [
    {
      "type": "technical_explanation",
      "response": "CPU usage represents the percentage of processing power...",
      "model": "qwen2.5-coder:14b",
      "model_type": "ollama_direct",
      "session_id": "session_20250720_103000",
      "timestamp": "2025-07-20T10:30:15.123456",
      "generation_batch": 0,
      "term": "cpu_usage",
      "category": "system_metrics"
    }
  ]
}
```

### metrics_dataset.json
Contains synthetic system metrics for anomaly detection training:

```json
{
  "metadata": {
    "generated_at": "2025-07-20T11:45:00",
    "generator": "optimized_v2",
    "session_id": "session_20250720_103000",
    "total_samples": 20000,
    "anomaly_ratio": 0.2,
    "config_hash": "abc123...",
    "generation_stats": {
      "normal_samples": 16000,
      "anomaly_samples": 4000,
      "anomaly_types": {
        "cpu_spike": 1200,
        "memory_leak": 1100,
        "disk_full": 900,
        "network_congestion": 800
      }
    }
  },
  "training_samples": [
    {
      "id": "normal_20250720_114500",
      "timestamp": "2025-07-20T11:45:00.000000",
      "metrics": {
        "cpu_usage": 25.3,
        "memory_usage": 45.7,
        "disk_usage": 67.2,
        "load_average": 1.8,
        "java_heap_usage": 55.0,
        "java_gc_time": 3.2,
        "spectrum_active_jobs": 12,
        "vemkd_process_count": 156
      },
      "status": "normal",
      "anomaly_type": null,
      "explanation": "System operating within normal parameters.",
      "session_id": "session_20250720_103000",
      "batch_id": 0
    }
  ]
}
```

## Data Sources Integration

### Technology Coverage
- **Red Hat Linux**: System metrics, vemkd logs, service states
- **IBM Spectrum Conductor**: Job queues, resource allocation, cluster status
- **Splunk**: Log patterns, query results, alert configurations
- **Jira/Confluence**: Ticket patterns, documentation snippets
- **Java Applications**: Heap usage, GC metrics, thread states

### Sample Type Distribution
| Type | Count | Source | Purpose |
|------|-------|--------|---------|
| technical_explanation | 2,542 | technical_terms.yaml | Core concept understanding |
| error_troubleshooting | 66 | error_patterns.yaml | Problem resolution |
| conversation | 280 | conversation_prompts.yaml | Natural dialogue |
| project_management | 1,250 | PM/banking terms | Enterprise context |
| banking_scenario | 1,250 | Regulatory/compliance | Executive briefings |
| metrics_explanation | 24 | metrics_patterns.yaml | Monitoring concepts |
| personality_response | 16 | personality_types.yaml | Response variety |
| question_style_variation | 20 | question_styles.yaml | Query handling |
| english_pattern | 192 | english_patterns.yaml | Language fluency |

## Generation Process

### Multi-Model Distillation
1. **YAML Content Processing**: Extract terms, patterns, scenarios
2. **Prompt Generation**: Create contextual prompts using templates
3. **Model Rotation**: Query 2+ models per prompt for variety
4. **Response Collection**: Gather and validate model responses
5. **Quality Filtering**: Remove short/low-quality responses
6. **Dataset Assembly**: Combine into structured JSON format

### Model Sources
- **Ollama Models**: qwen2.5-coder, gemma3, deepseek-r1, phi4, etc.
- **Remote APIs**: Gemini, Claude (if configured)
- **Local Models**: Cached HuggingFace transformers
- **Static Fallback**: Hardcoded responses for reliability

### Progress Tracking
The system maintains detailed progress in `generation_progress.pkl`:
- Session management and resumability
- Per-sample-type completion tracking
- Failed item retry mechanisms
- Model rotation state persistence
- Generation performance metrics

## Data Quality Assurance

### Content Validation
- **Response Length**: Minimum 20 words per response
- **Technical Accuracy**: Sourced from domain-specific YAML
- **Language Quality**: Natural, conversational tone
- **Variety**: Multiple models prevent response monotony

### Metadata Tracking
Each sample includes:
- **Source Model**: Which model generated the response
- **Generation Context**: Term, category, error type, etc.
- **Session Information**: When and how it was generated
- **Batch Tracking**: Generation sequence and grouping

### Error Handling
- Failed generations logged and retryable
- Malformed responses automatically filtered
- Model timeouts gracefully handled
- Progress preserved across interruptions

## Usage in Training

### Language Model Training
```python
# Load language dataset
with open('training/language_dataset.json', 'r') as f:
    lang_data = json.load(f)

# Extract samples by type
tech_samples = [s for s in lang_data['samples'] if s['type'] == 'technical_explanation']
conv_samples = [s for s in lang_data['samples'] if s['type'] == 'conversation']
```

### Metrics Model Training
```python
# Load metrics dataset
with open('training/metrics_dataset.json', 'r') as f:
    metrics_data = json.load(f)

# Separate normal vs anomaly samples
normal_samples = [s for s in metrics_data['training_samples'] if s['status'] == 'normal']
anomaly_samples = [s for s in metrics_data['training_samples'] if s['status'] == 'anomaly']
```

## Performance Metrics

### Generation Statistics
- **Sample Rate**: ~2-5 samples/minute depending on models
- **Model Efficiency**: Parallel processing with rotation
- **Memory Usage**: Batch processing to manage memory
- **Disk Usage**: JSON compression for large datasets

### Quality Metrics
- **Response Diversity**: Multiple models per prompt
- **Content Coverage**: All YAML domains represented
- **Technical Depth**: Domain-specific terminology
- **Conversational Quality**: Natural language patterns

## Maintenance Commands

### Dataset Analysis
```python
# Analyze sample distribution
from dataset_generator import OptimizedDatasetGenerator
generator = OptimizedDatasetGenerator()
generator.show_completion_summary()

# Check model usage
from pathlib import Path
import json

with open('training/language_dataset.json', 'r') as f:
    data = json.load(f)
    
print("Model Distribution:")
for model, count in data['metadata']['model_distribution'].items():
    print(f"  {model}: {count}")
```

### Progress Management
```python
# Check current progress
show_progress()

# Resume interrupted generation
generate_datasets()

# Retry failed items
retry_failed()

# Start fresh (WARNING: deletes existing data)
reset_progress()
```

## File Management

### Backup Strategy
- Automatic backups before major regeneration
- Timestamped backup files for recovery
- Progress files enable safe interruption/resume
- Configuration hash tracking prevents conflicts

### Disk Space Management
- **language_dataset.json**: ~50-200MB depending on sample count
- **metrics_dataset.json**: ~100-500MB depending on metrics samples
- **Backup files**: Additional 2x space for safety
- **Logs**: ~10-50MB for detailed generation logs

### Cleanup Commands
```python
# Remove old backups (keep last 3)
import glob
from pathlib import Path

backup_files = sorted(glob.glob('training/*_backup_*.json'))
for old_backup in backup_files[:-3]:
    Path(old_backup).unlink()
    print(f"Removed old backup: {old_backup}")
```

## Integration Points

### Model Training Pipeline
1. **Data Loading**: Read JSON datasets
2. **Preprocessing**: Tokenization and encoding
3. **Training**: Multi-task learning (classification, regression, anomaly detection)
4. **Validation**: Hold-out testing and metrics evaluation
5. **Deployment**: Save trained model for inference

### Monitoring System Integration
1. **Real-time Inference**: Load trained model
2. **Metric Processing**: Apply learned patterns
3. **Anomaly Detection**: Use trained thresholds
4. **Response Generation**: Natural language explanations
5. **Continual Learning**: Update with new incident data

## Troubleshooting

### Common Issues
- **Large file sizes**: Reduce sample counts in config
- **Generation timeouts**: Increase Ollama timeout settings
- **Memory errors**: Reduce batch size or use CPU training
- **Corrupted files**: Use backup files for recovery

### Recovery Procedures
```python
# Check file integrity
import json
try:
    with open('training/language_dataset.json', 'r') as f:
        data = json.load(f)
    print("✅ File is valid JSON")
except json.JSONDecodeError as e:
    print(f"❌ JSON error: {e}")
    # Restore from backup
```

## Performance Optimization

### Generation Speed
- **Model Preloading**: Keep Ollama models warm
- **Parallel Processing**: Multiple concurrent requests
- **Batch Optimization**: Process similar items together
- **Smart Resume**: Skip completed items on restart

### Storage Efficiency
- **JSON Compression**: Efficient encoding
- **Incremental Updates**: Add new samples without full regeneration
- **Metadata Optimization**: Track only essential information
- **Archive Strategy**: Compress old datasets

This training directory serves as the central repository for all generated training data, providing the foundation for creating a specialized monitoring LLM that understands your specific technology stack and operational patterns.