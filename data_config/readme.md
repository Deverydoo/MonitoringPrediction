# data_config Directory

## Overview

The `data_config/` directory contains YAML configuration files that drive dynamic training data generation for the distilled monitoring system. The system automatically discovers and processes all YAML files in this directory, making it easy to add new knowledge domains without code changes.

## Dynamic Discovery System

The system scans this directory for `.yaml` files and automatically:
- Loads all YAML configurations at startup
- Calculates training targets based on content volume
- Distributes samples across discovered knowledge domains
- Adapts to new files without configuration changes

**Current Files (9 discovered):**
- `technical_terms.yaml` - System administration terminology
- `error_patterns.yaml` - Common error messages and solutions
- `conversation_prompts.yaml` - Workplace dialogue patterns
- `personality_types.yaml` - Response style variations
- `question_styles.yaml` - Question formatting patterns
- `english_patterns.yaml` - Natural language templates
- `project_management_terms.yaml` - PM/banking terminology
- `metrics_patterns.yaml` - System metric definitions
- `response_templates.yaml` - Response structure templates

## Content-Based Target Calculation

Training targets are dynamically calculated using the formula:
**YAML Content × models_per_question = Training Samples**

Example calculation:
- technical_terms.yaml: 1,271 terms × 2 models = 2,542 samples
- error_patterns.yaml: 33 patterns × 2 models = 66 samples
- Total: 5,640 language training samples

## File Structure Standards

### Technical Terms (`technical_terms.yaml`)
```yaml
category_name:
  - term1
  - term2
  - term3
```

### Error Patterns (`error_patterns.yaml`)
```yaml
error_category:
  error_group:
    - "error message 1"
    - "error message 2"
```

### Conversation Prompts (`conversation_prompts.yaml`)
```yaml
conversation_styles:
  - style1
  - style2
explanation_prompts:
  category:
    - "prompt template {term}"
```

### Personality Types (`personality_types.yaml`)
```yaml
personalities:
  - name: "helpful_colleague"
    description: "A knowledgeable team member"
    style: "friendly, practical"
```

## Adding New Knowledge Domains

To add a new knowledge domain:

1. **Create new YAML file**: `new_domain.yaml`
2. **Add content**: Follow existing patterns
3. **Restart system**: Will auto-discover new file
4. **Check targets**: Use `show_progress()` to see new calculations

Example new file (`security_patterns.yaml`):
```yaml
security_concepts:
  - firewall_rules
  - access_control
  - vulnerability_scanning
security_incidents:
  network:
    - "unauthorized access detected"
    - "suspicious network traffic"
```

## Technology-Specific Content

### IBM Spectrum Conductor
- Workload management terminology
- Resource allocation concepts
- Job scheduling patterns

### Red Hat Linux
- System administration commands
- Log file patterns (vemkd)
- Service management

### Splunk Integration
- Query patterns
- Log analysis terminology
- Alert configurations

### Jira/Confluence
- Ticket lifecycle terminology
- Documentation patterns
- Workflow concepts

## Dynamic Sample Distribution

The system automatically distributes samples based on content volume:

| Sample Type | Ratio | Source |
|-------------|-------|--------|
| technical_explanation | 45% | technical_terms.yaml |
| error_troubleshooting | 12% | error_patterns.yaml |
| conversation | 25% | conversation_prompts.yaml |
| personality_response | 3% | personality_types.yaml |
| question_style_variation | 4% | question_styles.yaml |
| english_pattern | 5% | english_patterns.yaml |
| project_management | 4% | project_management_terms.yaml |
| metrics_explanation | 2% | metrics_patterns.yaml |

## Model Rotation Integration

Each YAML file feeds into the model rotation system where:
- Multiple Ollama models generate responses
- Content gets cross-pollinated across models
- Response variety increases naturally
- Training data quality improves

## Configuration Validation

The system validates YAML files for:
- **Structure consistency**: Required keys present
- **Content quality**: Minimum content thresholds
- **Format compliance**: Valid YAML syntax
- **Encoding compatibility**: UTF-8 support

## Performance Optimization

### Content Caching
- YAML files loaded once at startup
- Content hash tracking for change detection
- Smart reload only when files modified

### Batch Processing
- Content processed in optimized batches
- Memory-efficient loading patterns
- Progress tracking per file

### Parallel Generation
- Multiple models query same content
- Concurrent processing where possible
- Rate limiting to prevent overload

## Maintenance Commands

```python
# Check current YAML content
from dataset_generator import OptimizedDatasetGenerator
generator = OptimizedDatasetGenerator()
generator.show_dynamic_sizing_info()

# Validate all YAML files
from data_config_manager import DataConfigManager
manager = DataConfigManager()
manager.validate_configs()

# Show content statistics
manager.show_stats()
```

## Environment Portability

The data_config directory is fully portable:
- **No absolute paths**: All references relative
- **Self-contained**: No external dependencies
- **Cross-platform**: Works on Windows/Linux/Mac
- **Version controlled**: Track changes over time

## Best Practices

### Adding Content
1. **Start small**: Add 10-20 items initially
2. **Test quality**: Generate samples and review
3. **Scale gradually**: Increase content over time
4. **Monitor metrics**: Use progress tracking

### File Organization
1. **Logical grouping**: Group related concepts
2. **Clear naming**: Descriptive file names
3. **Consistent structure**: Follow existing patterns
4. **Documentation**: Comment complex sections

### Performance Tuning
1. **Balance content**: Avoid files with 1000+ items
2. **Split large files**: Break into logical chunks
3. **Monitor generation**: Track time per file
4. **Optimize models**: Adjust models_per_question

## Troubleshooting

### Common Issues
- **Empty targets**: Check YAML structure
- **Slow generation**: Reduce content or models
- **Memory issues**: Split large files
- **Encoding errors**: Ensure UTF-8 encoding

### Debug Commands
```python
# Check file discovery
generator._load_all_configs()

# Validate specific file
manager.validate_configs()

# Show generation plan
generator._get_smart_generation_plan()
```

## Future Expansion

The system is designed to scale:
- **New domains**: Simply add YAML files
- **More models**: Increase Ollama model pool
- **Larger datasets**: Automatic target scaling
- **Complex patterns**: Advanced YAML structures

Ready for 15+ YAML files covering comprehensive monitoring knowledge.