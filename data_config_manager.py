#!/usr/bin/env python3
"""
data_config_manager.py
Utility script to help manage YAML configuration files for dataset generation
"""

import yaml
import json
from pathlib import Path
import shutil
from datetime import datetime
from typing import Dict, List, Any

class DataConfigManager:
    """Manages YAML configuration files for dataset generation."""
    
    def __init__(self, config_dir: str = "./data_config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Template configurations
        self.templates = {
            "technical_terms.yaml": self._get_technical_terms_template(),
            "conversation_prompts.yaml": self._get_conversation_prompts_template(),
            "error_patterns.yaml": self._get_error_patterns_template(),
            "metrics_patterns.yaml": self._get_metrics_patterns_template(),
            "personality_types.yaml": self._get_personality_types_template(),
            "question_styles.yaml": self._get_question_styles_template(),
            "english_patterns.yaml": self._get_english_patterns_template()
        }
    
    def _get_technical_terms_template(self) -> Dict:
        """Get technical terms template."""
        return {
            "linux": [
                "systemd", "cron", "iptables", "journalctl", "systemctl", "ps", "top", "htop",
                "df", "du", "lsof", "netstat", "ss", "sar", "iostat", "vmstat", "dmesg"
            ],
            "java": [
                "OutOfMemoryError", "StackOverflowError", "NullPointerException", "GC",
                "garbage collection", "JVM", "heap", "stack", "metaspace", "thread dump"
            ],
            "monitoring": [
                "threshold", "alert", "metric", "KPI", "SLA", "anomaly", "baseline",
                "performance", "latency", "throughput", "availability", "reliability"
            ],
            "spark": [
                "executor", "driver", "RDD", "DataFrame", "partition", "shuffle", "spill",
                "broadcast", "accumulator", "catalyst", "lineage", "checkpoint"
            ],
            "spectrum": [
                "Spectrum Conductor", "workload", "resource group", "consumer", "allocation",
                "priority", "slot", "demand", "supply", "elastic", "auto scaling"
            ]
        }
    
    def _get_conversation_prompts_template(self) -> Dict:
        """Get conversation prompts template."""
        return {
            "conversation_styles": [
                "casual_explanation", "expert_breakdown", "troubleshooting_help", 
                "newbie_friendly", "practical_advice", "step_by_step"
            ],
            "conversation_types": [
                "mentor_moment", "debugging_buddy", "knowledge_share", 
                "onboarding_help", "crisis_debrief", "tool_recommendation"
            ],
            "explanation_prompts": {
                "casual_explanation": [
                    "Hey, can you explain {term} in simple terms?",
                    "What exactly is {term} and why should I care about it?",
                    "Someone mentioned {term} in a meeting. Can you break it down for me?"
                ],
                "expert_breakdown": [
                    "Can you give me a technical deep-dive on {term}?",
                    "From an architecture perspective, how does {term} work?",
                    "What are the performance implications of {term}?"
                ]
            },
            "error_scenarios": {
                "frustrated_user": [
                    "I keep getting this error: '{error}' and it's driving me crazy!",
                    "This stupid error '{error}' keeps popping up. I have no idea what to do."
                ],
                "urgent_production": [
                    "URGENT: Production is down with error '{error}'. Need immediate help!",
                    "Red alert! We're seeing '{error}' in production. What's the quickest fix?"
                ]
            }
        }
    
    def _get_error_patterns_template(self) -> Dict:
        """Get error patterns template."""
        return {
            "java_errors": {
                "heap_errors": [
                    "java.lang.OutOfMemoryError: Java heap space",
                    "java.lang.OutOfMemoryError: GC overhead limit exceeded",
                    "java.lang.OutOfMemoryError: Metaspace"
                ],
                "runtime_errors": [
                    "java.lang.StackOverflowError",
                    "java.lang.ClassNotFoundException",
                    "java.lang.NoSuchMethodError"
                ]
            },
            "linux_errors": {
                "system_errors": [
                    "segmentation fault", "kernel panic", "out of memory",
                    "disk full", "permission denied", "no such file or directory"
                ],
                "network_errors": [
                    "connection refused", "network unreachable", "no route to host"
                ]
            },
            "troubleshooting_scenarios": {
                "performance": [
                    "High CPU usage on Linux server",
                    "Memory leak in Java application", 
                    "Spark job running slowly",
                    "Database queries taking too long"
                ],
                "availability": [
                    "Service becoming unresponsive",
                    "Intermittent connection failures",
                    "Load balancer health check failures"
                ]
            }
        }
    
    def _get_metrics_patterns_template(self) -> Dict:
        """Get metrics patterns template."""
        return {
            "normal_ranges": {
                "system_metrics": {
                    "cpu_usage": [5, 30],
                    "memory_usage": [20, 60],
                    "disk_usage": [10, 70],
                    "load_average": [0.5, 2.0],
                    "network_io": [1000, 50000],
                    "disk_io": [100, 10000]
                },
                "java_metrics": {
                    "heap_usage": [30, 70],
                    "gc_time": [1, 5],
                    "thread_count": [20, 100],
                    "class_count": [5000, 20000]
                }
            },
            "anomaly_patterns": {
                "cpu_spike": {
                    "description": "High CPU utilization with increased load",
                    "metrics": {
                        "cpu_usage": [80, 100],
                        "load_average": [8, 16]
                    },
                    "correlated_effects": {
                        "java_gc_time": [10, 25],
                        "response_time": [2000, 10000]
                    }
                },
                "memory_leak": {
                    "description": "Gradual memory consumption leading to exhaustion",
                    "metrics": {
                        "memory_usage": [85, 99],
                        "swap_usage": [50, 100]
                    },
                    "correlated_effects": {
                        "java_heap_usage": [90, 99],
                        "java_gc_frequency": [20, 50]
                    }
                }
            }
        }
    
    def _get_personality_types_template(self) -> Dict:
        """Get personality types template."""
        return {
            "personalities": [
                {
                    "name": "helpful_colleague",
                    "description": "A knowledgeable team member who enjoys helping others learn",
                    "style": "friendly, practical, uses real examples"
                },
                {
                    "name": "patient_mentor",
                    "description": "An experienced engineer who takes time to explain concepts thoroughly",
                    "style": "detailed explanations, step-by-step approach"
                },
                {
                    "name": "practical_engineer",
                    "description": "A hands-on technical person focused on getting things done",
                    "style": "direct, solution-oriented, tool-focused"
                }
            ]
        }
    
    def _get_question_styles_template(self) -> Dict:
        """Get question styles template."""
        return {
            "styles": [
                {
                    "name": "direct",
                    "pattern": "What is {topic}?",
                    "tone": "straightforward",
                    "context": "quick_reference"
                },
                {
                    "name": "exploratory", 
                    "pattern": "Can you explain how {topic} works?",
                    "tone": "curious",
                    "context": "learning"
                },
                {
                    "name": "practical",
                    "pattern": "How do I use {topic} in practice?",
                    "tone": "application_focused",
                    "context": "implementation"
                }
            ]
        }
    
    def _get_english_patterns_template(self) -> Dict:
        """Get English patterns template."""
        return {
            "templates": {
                "general_knowledge": [
                    "Explain the concept of {topic} in simple terms",
                    "What are the key benefits of {topic}?",
                    "How does {topic} work in practice?"
                ],
                "conversational": [
                    "Tell me about your experience with {topic}",
                    "What would you recommend for someone new to {topic}?",
                    "Share some practical tips about {topic}"
                ],
                "technical_discussion": [
                    "From a technical perspective, how does {topic} function?",
                    "What are the implementation details of {topic}?",
                    "How do you optimize {topic} for better performance?"
                ]
            },
            "natural_language_topics": {
                "soft_skills": [
                    "communication", "teamwork", "leadership", "mentoring",
                    "collaboration", "problem_solving", "critical_thinking"
                ],
                "workplace_concepts": [
                    "project_management", "agile_methodology", "code_review",
                    "documentation", "knowledge_sharing", "continuous_learning"
                ]
            }
        }
    
    def create_default_configs(self, overwrite: bool = False):
        """Create default configuration files."""
        print("🔧 Creating default configuration files...")
        
        created_files = []
        skipped_files = []
        
        for filename, template in self.templates.items():
            file_path = self.config_dir / filename
            
            if file_path.exists() and not overwrite:
                skipped_files.append(filename)
                continue
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(template, f, default_flow_style=False, allow_unicode=True, indent=2)
                created_files.append(filename)
                print(f"  ✅ Created {filename}")
            except Exception as e:
                print(f"  ❌ Failed to create {filename}: {e}")
        
        print(f"\n📊 Summary:")
        print(f"  Created: {len(created_files)} files")
        print(f"  Skipped: {len(skipped_files)} files")
        
        if skipped_files:
            print(f"\n  Skipped files (already exist):")
            for filename in skipped_files:
                print(f"    - {filename}")
            print(f"  Use overwrite=True to replace existing files")
        
        return created_files, skipped_files
    
    def validate_configs(self) -> Dict[str, Any]:
        """Validate all configuration files."""
        print("🔍 Validating configuration files...")
        
        validation_results = {
            "valid_files": [],
            "invalid_files": [],
            "missing_files": [],
            "warnings": []
        }
        
        for filename in self.templates.keys():
            file_path = self.config_dir / filename
            
            if not file_path.exists():
                validation_results["missing_files"].append(filename)
                print(f"  ❌ Missing: {filename}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = yaml.safe_load(f)
                
                if content is None:
                    validation_results["invalid_files"].append(filename)
                    print(f"  ❌ Invalid (empty): {filename}")
                    continue
                
                # Basic structure validation
                if filename == "technical_terms.yaml":
                    if not isinstance(content, dict):
                        validation_results["warnings"].append(f"{filename}: Expected dict structure")
                elif filename == "conversation_prompts.yaml":
                    required_keys = ["conversation_styles", "conversation_types"]
                    missing_keys = [key for key in required_keys if key not in content]
                    if missing_keys:
                        validation_results["warnings"].append(f"{filename}: Missing keys: {missing_keys}")
                
                validation_results["valid_files"].append(filename)
                print(f"  ✅ Valid: {filename}")
                
            except yaml.YAMLError as e:
                validation_results["invalid_files"].append(filename)
                print(f"  ❌ YAML Error in {filename}: {e}")
            except Exception as e:
                validation_results["invalid_files"].append(filename)
                print(f"  ❌ Error validating {filename}: {e}")
        
        # Summary
        print(f"\n📊 Validation Summary:")
        print(f"  Valid: {len(validation_results['valid_files'])}")
        print(f"  Invalid: {len(validation_results['invalid_files'])}")
        print(f"  Missing: {len(validation_results['missing_files'])}")
        print(f"  Warnings: {len(validation_results['warnings'])}")
        
        if validation_results['warnings']:
            print(f"\n⚠️  Warnings:")
            for warning in validation_results['warnings']:
                print(f"    - {warning}")
        
        return validation_results
    
    def backup_configs(self, backup_dir: str = None):
        """Create backup of all configuration files."""
        if backup_dir is None:
            backup_dir = f"./data_config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)
        
        print(f"💾 Creating backup in {backup_path}...")
        
        backed_up = []
        for filename in self.templates.keys():
            file_path = self.config_dir / filename
            if file_path.exists():
                backup_file = backup_path / filename
                shutil.copy2(file_path, backup_file)
                backed_up.append(filename)
                print(f"  ✅ Backed up {filename}")
        
        print(f"\n📊 Backup Summary:")
        print(f"  Files backed up: {len(backed_up)}")
        print(f"  Backup location: {backup_path}")
        
        return backup_path
    
    def merge_configs(self, source_dir: str):
        """Merge configurations from another directory."""
        source_path = Path(source_dir)
        
        if not source_path.exists():
            print(f"❌ Source directory {source_path} does not exist")
            return
        
        print(f"🔀 Merging configurations from {source_path}...")
        
        merged_files = []
        for filename in self.templates.keys():
            source_file = source_path / filename
            target_file = self.config_dir / filename
            
            if not source_file.exists():
                continue
            
            try:
                # Load both files
                with open(source_file, 'r', encoding='utf-8') as f:
                    source_content = yaml.safe_load(f) or {}
                
                if target_file.exists():
                    with open(target_file, 'r', encoding='utf-8') as f:
                        target_content = yaml.safe_load(f) or {}
                else:
                    target_content = {}
                
                # Merge (source takes precedence)
                if isinstance(source_content, dict) and isinstance(target_content, dict):
                    merged_content = {**target_content, **source_content}
                else:
                    merged_content = source_content
                
                # Save merged content
                with open(target_file, 'w', encoding='utf-8') as f:
                    yaml.dump(merged_content, f, default_flow_style=False, allow_unicode=True, indent=2)
                
                merged_files.append(filename)
                print(f"  ✅ Merged {filename}")
                
            except Exception as e:
                print(f"  ❌ Failed to merge {filename}: {e}")
        
        print(f"\n📊 Merge Summary:")
        print(f"  Files merged: {len(merged_files)}")
        
        return merged_files
    
    def show_stats(self):
        """Show statistics about configuration files."""
        print("📊 Configuration File Statistics")
        print("=" * 50)
        
        total_terms = 0
        total_prompts = 0
        total_errors = 0
        
        for filename in self.templates.keys():
            file_path = self.config_dir / filename
            
            if not file_path.exists():
                print(f"{filename}: ❌ Missing")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = yaml.safe_load(f) or {}
                
                print(f"\n{filename}:")
                
                if filename == "technical_terms.yaml":
                    if isinstance(content, dict):
                        for category, terms in content.items():
                            if isinstance(terms, list):
                                count = len(terms)
                                total_terms += count
                                print(f"  {category}: {count} terms")
                
                elif filename == "conversation_prompts.yaml":
                    styles = len(content.get('conversation_styles', []))
                    types = len(content.get('conversation_types', []))
                    explanations = sum(len(prompts) for prompts in content.get('explanation_prompts', {}).values())
                    total_prompts += styles + types + explanations
                    print(f"  Conversation styles: {styles}")
                    print(f"  Conversation types: {types}")
                    print(f"  Explanation prompts: {explanations}")
                
                elif filename == "error_patterns.yaml":
                    error_count = 0
                    for category, error_groups in content.items():
                        if isinstance(error_groups, dict):
                            for group, errors in error_groups.items():
                                if isinstance(errors, list):
                                    error_count += len(errors)
                    total_errors += error_count
                    print(f"  Total error patterns: {error_count}")
                
                elif filename == "personality_types.yaml":
                    personalities = len(content.get('personalities', []))
                    print(f"  Personality types: {personalities}")
                
                elif filename == "question_styles.yaml":
                    styles = len(content.get('styles', []))
                    print(f"  Question styles: {styles}")
                
                elif filename == "english_patterns.yaml":
                    templates = content.get('templates', {})
                    template_count = sum(len(prompts) for prompts in templates.values())
                    print(f"  English templates: {template_count}")
                
                elif filename == "metrics_patterns.yaml":
                    normal_ranges = content.get('normal_ranges', {})
                    anomaly_patterns = len(content.get('anomaly_patterns', {}))
                    normal_count = sum(len(metrics) for metrics in normal_ranges.values() if isinstance(metrics, dict))
                    print(f"  Normal metric ranges: {normal_count}")
                    print(f"  Anomaly patterns: {anomaly_patterns}")
                
            except Exception as e:
                print(f"{filename}: ❌ Error reading ({e})")
        
        print(f"\n" + "=" * 50)
        print(f"TOTALS:")
        print(f"  Technical terms: {total_terms}")
        print(f"  Conversation elements: {total_prompts}")
        print(f"  Error patterns: {total_errors}")
        print("=" * 50)
    
    def interactive_setup(self):
        """Interactive setup wizard for configuration files."""
        print("🧙 Interactive Configuration Setup Wizard")
        print("=" * 50)
        
        # Check existing files
        existing_files = []
        missing_files = []
        
        for filename in self.templates.keys():
            file_path = self.config_dir / filename
            if file_path.exists():
                existing_files.append(filename)
            else:
                missing_files.append(filename)
        
        if existing_files:
            print(f"\n✅ Found {len(existing_files)} existing configuration files:")
            for filename in existing_files:
                print(f"   - {filename}")
        
        if missing_files:
            print(f"\n❌ Missing {len(missing_files)} configuration files:")
            for filename in missing_files:
                print(f"   - {filename}")
            
            create_missing = input(f"\nCreate missing files with default templates? (y/n): ").strip().lower()
            if create_missing == 'y':
                self.create_default_configs(overwrite=False)
        
        # Validate all files
        print(f"\n🔍 Validating configurations...")
        validation_results = self.validate_configs()
        
        if validation_results['invalid_files']:
            fix_invalid = input(f"\nRecreate invalid files? (y/n): ").strip().lower()
            if fix_invalid == 'y':
                for filename in validation_results['invalid_files']:
                    file_path = self.config_dir / filename
                    with open(file_path, 'w', encoding='utf-8') as f:
                        yaml.dump(self.templates[filename], f, default_flow_style=False, allow_unicode=True, indent=2)
                    print(f"  ✅ Recreated {filename}")
        
        # Show statistics
        print(f"\n📊 Current configuration statistics:")
        self.show_stats()
        
        # Backup option
        backup_choice = input(f"\nCreate backup of current configurations? (y/n): ").strip().lower()
        if backup_choice == 'y':
            self.backup_configs()
        
        print(f"\n✅ Configuration setup complete!")
        print(f"   Configuration directory: {self.config_dir}")
        print(f"   Ready for dataset generation!")


def main():
    """Main function for command-line usage."""
    import sys
    
    manager = DataConfigManager()
    
    if len(sys.argv) < 2:
        print("Data Configuration Manager")
        print("Usage: python data_config_manager.py <command>")
        print()
        print("Commands:")
        print("  create     - Create default configuration files")
        print("  validate   - Validate existing configuration files")
        print("  backup     - Create backup of configuration files")
        print("  stats      - Show configuration statistics")
        print("  setup      - Interactive setup wizard")
        return
    
    command = sys.argv[1].lower()
    
    if command == "create":
        overwrite = "--overwrite" in sys.argv
        manager.create_default_configs(overwrite=overwrite)
    elif command == "validate":
        manager.validate_configs()
    elif command == "backup":
        manager.backup_configs()
    elif command == "stats":
        manager.show_stats()
    elif command == "setup":
        manager.interactive_setup()
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()