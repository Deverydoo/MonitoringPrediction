import json
import yaml
import random
import pickle
import requests
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
from dataclasses import dataclass, asdict

from config import CONFIG, model_chain
from common_utils import (
    load_dataset_file, save_dataset_file, analyze_existing_datasets,
    get_dataset_paths, save_generation_progress, load_generation_progress,
    DATASET_FORMATS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationProgress:
    """Enhanced progress tracking for resumable generation."""
    session_id: str
    started_at: str
    last_updated: str
    language_completed: int = 0
    metrics_completed: int = 0
    failed_items: List[str] = None
    current_batch: int = 0
    samples_since_save: int = 0
    yaml_content_hash: str = ""
    
    def __post_init__(self):
        if self.failed_items is None:
            self.failed_items = []

class EnhancedDatasetGenerator:
    """Enhanced dataset generator with dynamic YAML discovery and robust progress tracking."""
    
    def __init__(self, config_dir: str = None):
        self.config_dir = Path(config_dir or CONFIG.get('data_config_dir', './data_config'))
        self.training_dir = Path(CONFIG.get('training_dir', './training/'))
        self.progress_file = Path("./generation_progress.pkl")
        self.save_frequency = CONFIG.get('rotation_round_size', 25)
        
        # Use ONLY common_utils functions
        from common_utils import get_dataset_paths, load_generation_progress
        self.dataset_paths = get_dataset_paths(self.training_dir)
        
        # Discover and load YAML configs FIRST
        self.yaml_configs = self._discover_and_load_yaml_files()
        
        # Load progress using common_utils - NO duplicate function
        self.progress = self._load_or_create_progress()
        
        # Rich content extraction
        self.conversation_styles = self._extract_conversation_styles()
        self.personality_types = self._extract_personality_types()
        self.question_styles = self._extract_question_styles()
        self.error_scenarios = self._extract_error_scenarios()
        self.technical_terms = self._extract_technical_terms()
        self.english_patterns = self._extract_english_patterns()
        
        # Generation stats
        self.generation_stats = {
            "samples_generated": 0,
            "api_calls": 0,
            "errors": 0,
            "saves_completed": 0,
            "models_per_question": CONFIG.get('models_per_question', 2)
        }
        
        logger.info(f"✅ Enhanced generator initialized")
        logger.info(f"   YAML configs: {len(self.yaml_configs)}")
        logger.info(f"   Technical terms: {len(self.technical_terms)}")
        logger.info(f"   Conversation styles: {len(self.conversation_styles)}")
    
    def _discover_and_load_yaml_files(self) -> Dict[str, Dict]:
        """Dynamically discover and load all YAML configuration files."""
        yaml_configs = {}
        
        if not self.config_dir.exists():
            logger.warning(f"Config directory not found: {self.config_dir}")
            return yaml_configs
        
        # Discover all .yaml and .yml files
        yaml_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.yml"))
        
        logger.info(f"🔍 Discovering YAML files in {self.config_dir}")
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                if data:
                    # Calculate content metrics for dynamic sizing
                    item_count = self._count_yaml_items(data)
                    category = self._classify_yaml_content(yaml_file.name, data)
                    
                    yaml_configs[yaml_file.stem] = {
                        'filename': yaml_file.name,
                        'path': str(yaml_file),
                        'data': data,
                        'item_count': item_count,
                        'category': category,
                        'loaded_at': datetime.now().isoformat()
                    }
                    
                    logger.info(f"  ✅ {yaml_file.name}: {item_count} items ({category})")
                else:
                    logger.warning(f"  ⚠️  {yaml_file.name}: Empty or invalid")
                    
            except Exception as e:
                logger.error(f"  ❌ {yaml_file.name}: Failed to load - {e}")
        
        logger.info(f"📊 Total YAML configs loaded: {len(yaml_configs)}")
        return yaml_configs
    
    def _count_yaml_items(self, data: Dict) -> int:
        """Count items in YAML data for dynamic target calculation."""
        if not isinstance(data, dict):
            return 0
        
        total_items = 0
        
        # Count different types of content
        for key, value in data.items():
            if isinstance(value, list):
                total_items += len(value)
            elif isinstance(value, dict):
                # Recursively count nested dictionaries
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list):
                        total_items += len(sub_value)
                    elif isinstance(sub_value, dict):
                        total_items += len(sub_value)
        
        return total_items
    
    def _classify_yaml_content(self, filename: str, data: Dict) -> str:
        """Classify YAML content type for targeted generation."""
        filename_lower = filename.lower()
        
        if 'technical' in filename_lower or 'terms' in filename_lower:
            return 'technical_terms'
        elif 'conversation' in filename_lower or 'prompt' in filename_lower:
            return 'conversation_patterns'
        elif 'error' in filename_lower or 'pattern' in filename_lower:
            return 'error_scenarios'
        elif 'metric' in filename_lower:
            return 'metrics_patterns'
        elif 'personality' in filename_lower:
            return 'personality_types'
        elif 'question' in filename_lower or 'style' in filename_lower:
            return 'question_styles'
        elif 'english' in filename_lower:
            return 'english_patterns'
        elif 'project' in filename_lower or 'management' in filename_lower:
            return 'project_management'
        else:
            return 'general_knowledge'
    
    def _extract_conversation_styles(self) -> Dict:
        """Extract conversation styles from YAML configs."""
        styles = {'basic_styles': []}
        
        for config_name, config_data in self.yaml_configs.items():
            data = config_data.get('data', {})
            
            # Extract conversation styles
            if 'conversation_styles' in data:
                styles['basic_styles'].extend(data['conversation_styles'])
            
            # Extract conversation types
            if 'conversation_types' in data:
                styles.setdefault('conversation_types', []).extend(data['conversation_types'])
            
            # Extract explanation prompts
            if 'explanation_prompts' in data:
                styles.setdefault('explanation_prompts', {}).update(data['explanation_prompts'])
        
        return styles
    
    def _extract_personality_types(self) -> List[Dict]:
        """Extract personality types from YAML configs."""
        personalities = []
        
        for config_name, config_data in self.yaml_configs.items():
            data = config_data.get('data', {})
            
            if 'personalities' in data:
                personalities.extend(data['personalities'])
        
        return personalities
    
    def _extract_question_styles(self) -> List[Dict]:
        """Extract question styles from YAML configs."""
        styles = []
        
        for config_name, config_data in self.yaml_configs.items():
            data = config_data.get('data', {})
            
            if 'styles' in data:
                styles.extend(data['styles'])
        
        return styles
    
    def _extract_error_scenarios(self) -> Dict:
        """Extract error scenarios from YAML configs."""
        scenarios = {}
        
        for config_name, config_data in self.yaml_configs.items():
            data = config_data.get('data', {})
            
            # Extract error scenarios
            if 'error_scenarios' in data:
                scenarios.update(data['error_scenarios'])
            
            # Extract specific error patterns
            for error_type in ['java_errors', 'linux_errors', 'spark_errors']:
                if error_type in data:
                    scenarios.setdefault(error_type, {}).update(data[error_type])
        
        return scenarios
    
    def _extract_technical_terms(self) -> Dict:
        """Extract technical terms from YAML configs."""
        terms = {}
        
        for config_name, config_data in self.yaml_configs.items():
            data = config_data.get('data', {})
            
            # Look for technical term categories
            for key, value in data.items():
                if isinstance(value, list) and key not in ['personalities', 'styles', 'conversation_styles']:
                    # This looks like a list of technical terms
                    terms[key] = value
        
        total_terms = sum(len(term_list) for term_list in terms.values())
        logger.info(f"🔧 Extracted technical terms: {total_terms} terms across {len(terms)} categories")
        return terms
    
    def _extract_english_patterns(self) -> Dict:
        """Extract English language patterns from YAML configs."""
        patterns = {}
        
        for config_name, config_data in self.yaml_configs.items():
            data = config_data.get('data', {})
            
            # Extract English templates and patterns
            if 'templates' in data:
                patterns.setdefault('templates', {}).update(data['templates'])
            
            if 'communication_patterns' in data:
                patterns.setdefault('communication_patterns', {}).update(data['communication_patterns'])
            
            if 'natural_language_topics' in data:
                patterns.setdefault('natural_language_topics', {}).update(data['natural_language_topics'])
        
        return patterns
    
    def calculate_enhanced_targets(self) -> Tuple[Dict[str, int], int]:
        """Calculate dynamic generation targets based on YAML content."""
        models_per_question = CONFIG.get('models_per_question', 2)
        base_multiplier = CONFIG.get('variety_multiplier', 1.5)
        
        # Calculate base targets from YAML content
        targets = {}
        
        # Technical explanations (largest category)
        technical_count = sum(len(terms) for terms in self.technical_terms.values())
        if technical_count > 0:
            base_technical = int(technical_count * base_multiplier)
            targets['technical_explanation'] = base_technical * models_per_question
        
        # Conversation patterns
        conversation_count = len(self.conversation_styles.get('basic_styles', []))
        if conversation_count > 0:
            base_conversation = int(conversation_count * base_multiplier * 5)  # More variety
            targets['conversation_example'] = base_conversation * models_per_question
        
        # Error scenarios
        error_count = sum(len(scenarios) for scenarios in self.error_scenarios.values())
        if error_count > 0:
            base_errors = int(error_count * base_multiplier)
            targets['error_troubleshooting'] = base_errors * models_per_question
        
        # English patterns
        english_count = sum(len(patterns) for patterns in self.english_patterns.values())
        if english_count > 0:
            base_english = int(english_count * base_multiplier)
            targets['english_pattern'] = base_english * models_per_question
        
        # Personality responses
        personality_count = len(self.personality_types)
        if personality_count > 0:
            base_personality = int(personality_count * base_multiplier * 3)
            targets['personality_response'] = base_personality * models_per_question
        
        # Question styles
        question_count = len(self.question_styles)
        if question_count > 0:
            base_questions = int(question_count * base_multiplier * 2)
            targets['question_style_variation'] = base_questions * models_per_question
        
        # Calculate total
        total_target = sum(targets.values())
        
        logger.info(f"🎯 Dynamic targets calculated:")
        logger.info(f"   Base multiplier: {base_multiplier}")
        logger.info(f"   Models per question: {models_per_question}")
        logger.info(f"   Total target: {total_target}")
        
        return targets, total_target
    
    def _load_or_create_progress(self) -> GenerationProgress:
        """Load progress using ONLY common_utils - no duplication."""
        from common_utils import load_generation_progress
        
        progress_data = load_generation_progress(self.progress_file)
        
        if progress_data:
            try:
                if isinstance(progress_data, dict):
                    progress = GenerationProgress(**progress_data)
                else:
                    progress = progress_data
                
                logger.info(f"📊 Resuming session {progress.session_id}")
                return progress
            except Exception as e:
                logger.warning(f"Invalid progress structure: {e}, starting new session")
        
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"🆕 Starting new session: {session_id}")
        
        return GenerationProgress(
            session_id=session_id,
            started_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )
    
    def _save_progress_pickle(self):
        """Save progress using ONLY common_utils - no duplication."""
        from common_utils import save_generation_progress
        
        self.progress.last_updated = datetime.now().isoformat()
        save_generation_progress(asdict(self.progress), self.progress_file)
    
    def _analyze_existing_dataset(self) -> Dict[str, int]:
        """Use ONLY common_utils analysis - no duplicate function."""
        from common_utils import analyze_existing_datasets
        
        analysis = analyze_existing_datasets(self.training_dir)
        
        # Extract sample counts by type
        existing_counts = {}
        if analysis['language_dataset']['exists']:
            distribution = analysis['language_dataset'].get('distribution', {})
            for sample_type, count in distribution.items():
                existing_counts[sample_type] = count
        
        return existing_counts
    
    def show_progress_with_multiplier(self):
        """Enhanced progress display showing base/actual with multiplier."""
        targets, total_target = self.calculate_enhanced_targets()
        existing = self._analyze_existing_dataset()
        models_per_question = CONFIG.get('models_per_question', 2)
        
        print(f"\n{'='*70}")
        print("ENHANCED DATASET GENERATION PROGRESS")
        print(f"{'='*70}")
        print(f"Session: {self.progress.session_id}")
        print(f"Models per question: {models_per_question}")
        print(f"Save frequency: every {self.save_frequency} samples")
        print(f"")
        
        total_existing = 0
        total_needed = 0
        
        for sample_type, actual_target in targets.items():
            base_target = actual_target // models_per_question
            current = existing.get(sample_type, 0)
            needed = max(0, actual_target - current)
            
            total_existing += current
            total_needed += needed
            
            percent = (current / actual_target * 100) if actual_target > 0 else 100
            status = "✅" if needed == 0 else "🔄"
            
            print(f"{status} {sample_type}:")
            print(f"    Progress: {current}/{actual_target} ({percent:.1f}%)")
            print(f"    Base: {base_target} × {models_per_question} models = {actual_target} total")
            print(f"    Remaining: {needed}")
            print("")
        
        overall_percent = (total_existing / total_target * 100) if total_target > 0 else 100
        print(f"📈 OVERALL: {total_existing}/{total_target} ({overall_percent:.1f}%)")
        print(f"🎯 Remaining: {total_needed} samples")
        print(f"💾 Saves completed: {self.generation_stats['saves_completed']}")
        print(f"{'='*70}")
    
    def generate_complete_dataset(self, language_count: int = None, metrics_count: int = None):
        """Generate complete dataset - ONLY write to files, return success status."""
        logger.info("🗣️ Starting complete dataset generation")
        
        try:
            # Generate language samples
            language_success = self.generate_rich_language_dataset(language_count)
            
            # Generate metrics samples  
            metrics_success = self.generate_enhanced_metrics_dataset(metrics_count)
            
            # Return success status instead of data
            if language_success is not None and metrics_success is not None:
                print("✅ Dataset generation completed successfully!")
                print(f"   Language dataset: {len(language_success) if isinstance(language_success, list) else 'Updated'}")
                print(f"   Metrics dataset: {len(metrics_success) if isinstance(metrics_success, list) else 'Updated'}")
                return True, True  # Success for both
            else:
                print("❌ Dataset generation failed")
                return False, False
                
        except Exception as e:
            logger.error(f"Dataset generation error: {e}")
            print(f"❌ Dataset generation failed: {e}")
            return False, False

    def _save_final_dataset(self):
        """Save final dataset using ONLY common_utils."""
        from common_utils import load_dataset_file, save_dataset_file
        
        # Load existing data
        existing_data = load_dataset_file(self.dataset_paths['language_dataset'])
        
        if existing_data and existing_data.get("samples"):
            # Update metadata
            existing_data["metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "total_samples": len(existing_data["samples"]),
                "session_id": self.progress.session_id,
                "models_per_question": CONFIG.get('models_per_question', 2),
                "generation_stats": self.generation_stats,
                "sample_distribution": self._get_sample_distribution(existing_data["samples"])
            }
            
            # Final save using common_utils
            save_dataset_file(existing_data, self.dataset_paths['language_dataset'], 'language_dataset')
            logger.info(f"✅ Final dataset saved: {len(existing_data['samples'])} samples")
            
            # CLEANUP using common_utils periodic_cleanup
            if self.progress_file.exists():
                try:
                    self.progress_file.unlink()
                    logger.info("🗑️ Cleaned up generation progress pickle")
                except Exception as e:
                    logger.warning(f"Failed to cleanup pickle: {e}")    
    
    def generate_rich_language_dataset(self, target_count: int = None) -> List[Dict]:
        """Generate rich language dataset using enhanced logic."""
        if target_count is None:
            targets, total_target = self.calculate_enhanced_targets()
            target_count = total_target
        
        logger.info(f"🗣️ Starting rich language generation: {target_count} target")
        
        # Check existing samples
        existing = self._analyze_existing_dataset()
        existing_total = sum(existing.values())
        
        if existing_total >= target_count:
            logger.info(f"✅ Target already met: {existing_total}/{target_count}")
            return []
        
        needed = target_count - existing_total
        logger.info(f"📊 Need to generate: {needed} samples")
        
        # Generate samples in batches
        all_samples = []
        batch_size = self.save_frequency
        
        try:
            for i in range(needed):
                # Generate a sample using various methods
                sample = self._generate_diverse_sample(i, needed)
                
                if sample:
                    all_samples.append(sample)
                    self.progress.samples_since_save += 1
                    self.generation_stats["samples_generated"] += 1
                
                # Incremental save
                if len(all_samples) >= batch_size:
                    self._save_incremental_progress(all_samples)
                    self.generation_stats["saves_completed"] += 1
                    all_samples = []  # Reset batch
                    self.progress.samples_since_save = 0
                
                # Progress logging
                if (i + 1) % 50 == 0:
                    progress_pct = ((i + 1) / needed) * 100
                    logger.info(f"📊 Progress: {i + 1}/{needed} ({progress_pct:.1f}%)")
            
            # Save any remaining samples
            if all_samples:
                self._save_incremental_progress(all_samples)
                self.generation_stats["saves_completed"] += 1
            
            # Final save with complete dataset
            logger.info("💾 Creating final dataset...")
            self._save_final_dataset()
            
            logger.info(f"✅ Rich language generation complete!")
            return []  # Data is saved to file
            
        except KeyboardInterrupt:
            logger.info("⏸️  Generation interrupted by user")
            if all_samples:
                self._save_incremental_progress(all_samples)
            return []
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            raise
    
    def _generate_diverse_sample(self, index: int, total: int) -> Optional[Dict]:
        """Generate a diverse sample using various techniques."""
        try:
            # Rotate through different sample types
            sample_types = ['technical_explanation', 'conversation_example', 'error_troubleshooting', 'english_pattern']
            sample_type = sample_types[index % len(sample_types)]
            
            # Generate based on type
            if sample_type == 'technical_explanation':
                return self._generate_technical_sample()
            elif sample_type == 'conversation_example':
                return self._generate_conversation_sample()
            elif sample_type == 'error_troubleshooting':
                return self._generate_error_sample()
            elif sample_type == 'english_pattern':
                return self._generate_english_sample()
            
        except Exception as e:
            logger.error(f"Error generating sample {index}: {e}")
            self.progress.failed_items.append(f"sample_{index}_{sample_type}")
            
        return None
    
    def _generate_technical_sample(self) -> Optional[Dict]:
        """Generate a technical explanation sample."""
        if not self.technical_terms:
            return None
        
        # Pick random category and term
        category = random.choice(list(self.technical_terms.keys()))
        terms = self.technical_terms[category]
        if not terms:
            return None
        
        term = random.choice(terms)
        
        # Create prompt
        prompt_templates = [
            f"Explain what {term} means in system monitoring",
            f"What is {term} and how does it work?",
            f"Tell me about {term} in simple terms",
            f"How do you troubleshoot issues with {term}?",
            f"What are the best practices for {term}?"
        ]
        
        prompt = random.choice(prompt_templates)
        
        # Get response from model chain
        responses = model_chain.generate_responses(prompt, max_tokens=300)
        
        if responses:
            response = responses[0]
            return {
                "type": "technical_explanation",
                "category": category,
                "term": term,
                "prompt": prompt,
                "explanation": response.get("response", ""),
                "model": response.get("model", "unknown"),
                "session_id": self.progress.session_id,
                "timestamp": datetime.now().isoformat()
            }
        
        return None
    
    def _generate_conversation_sample(self) -> Optional[Dict]:
        """Generate a conversation example sample."""
        if not self.conversation_styles.get('basic_styles'):
            return None
        
        style = random.choice(self.conversation_styles['basic_styles'])
        
        # Create conversational prompt
        prompts = [
            f"Have a {style} conversation about system monitoring",
            f"Explain CPU usage in a {style} way",
            f"Troubleshoot a memory issue using {style} communication",
            f"Discuss network problems in a {style} manner"
        ]
        
        prompt = random.choice(prompts)
        
        # Get response
        responses = model_chain.generate_responses(prompt, max_tokens=250)
        
        if responses:
            response = responses[0]
            return {
                "type": "conversation_example",
                "style": style,
                "prompt": prompt,
                "explanation": response.get("response", ""),
                "model": response.get("model", "unknown"),
                "session_id": self.progress.session_id,
                "timestamp": datetime.now().isoformat()
            }
        
        return None
    
    def _generate_error_sample(self) -> Optional[Dict]:
        """Generate an error troubleshooting sample."""
        if not self.error_scenarios:
            return None
        
        # Pick random error type
        error_type = random.choice(list(self.error_scenarios.keys()))
        error_data = self.error_scenarios[error_type]
        
        if isinstance(error_data, dict) and error_data:
            error_subtype = random.choice(list(error_data.keys()))
            errors = error_data[error_subtype]
            if isinstance(errors, list) and errors:
                error = random.choice(errors)
            else:
                error = error_subtype
        elif isinstance(error_data, list) and error_data:
            error = random.choice(error_data)
        else:
            return None
        
        # Create troubleshooting prompt
        prompt = f"How do you troubleshoot '{error}'? Provide step-by-step guidance."
        
        # Get response
        responses = model_chain.generate_responses(prompt, max_tokens=400)
        
        if responses:
            response = responses[0]
            return {
                "type": "error_troubleshooting",
                "error_type": error_type,
                "error": error,
                "prompt": prompt,
                "explanation": response.get("response", ""),
                "model": response.get("model", "unknown"),
                "session_id": self.progress.session_id,
                "timestamp": datetime.now().isoformat()
            }
        
        return None
    
    def _generate_english_sample(self) -> Optional[Dict]:
        """Generate an English pattern sample."""
        if not self.english_patterns.get('templates'):
            return None
        
        # Pick random template category
        template_categories = list(self.english_patterns['templates'].keys())
        category = random.choice(template_categories)
        templates = self.english_patterns['templates'][category]
        
        if not templates:
            return None
        
        template = random.choice(templates)
        
        # Fill template with technical topic
        if '{topic}' in template:
            topics = ['CPU monitoring', 'memory management', 'disk usage', 'network troubleshooting', 'log analysis']
            topic = random.choice(topics)
            prompt = template.replace('{topic}', topic)
        else:
            prompt = template
        
        # Get response
        responses = model_chain.generate_responses(prompt, max_tokens=200)
        
        if responses:
            response = responses[0]
            return {
                "type": "english_pattern",
                "category": category,
                "template": template,
                "prompt": prompt,
                "explanation": response.get("response", ""),
                "model": response.get("model", "unknown"),
                "session_id": self.progress.session_id,
                "timestamp": datetime.now().isoformat()
            }
        
        return None
    
    def _save_incremental_progress(self, new_samples: List[Dict]):
        """Save incremental progress using common utilities."""
        if not new_samples:
            return
        
        # Load existing data
        existing_data = load_dataset_file(self.dataset_paths['language_dataset'])
        if not existing_data:
            existing_data = {"samples": [], "metadata": {}}
        
        # Add new samples
        existing_data["samples"].extend(new_samples)
        
        # Save updated data
        save_dataset_file(existing_data, self.dataset_paths['language_dataset'], 'language_dataset')
        
        # Save progress
        self._save_progress_pickle()
        
        logger.info(f"💾 Saved {len(new_samples)} samples ({len(existing_data['samples'])} total)")
    
    def _save_incremental_progress(self, new_samples: List[Dict]):
        """Save incremental progress using ONLY common_utils."""
        if not new_samples:
            return
        
        from common_utils import load_dataset_file, save_dataset_file
        
        # Load existing data
        existing_data = load_dataset_file(self.dataset_paths['language_dataset'])
        if not existing_data:
            existing_data = {"samples": [], "metadata": {}}
        
        # Add new samples
        existing_data["samples"].extend(new_samples)
        
        # Save updated data
        save_dataset_file(existing_data, self.dataset_paths['language_dataset'], 'language_dataset')
        
        # Save progress
        self._save_progress_pickle()
        
        logger.info(f"💾 Saved {len(new_samples)} samples ({len(existing_data['samples'])} total)")
    
    def _get_sample_distribution(self, samples: List[Dict]) -> Dict[str, int]:
        """Get distribution of sample types."""
        distribution = {}
        for sample in samples:
            sample_type = sample.get('type', 'unknown')
            distribution[sample_type] = distribution.get(sample_type, 0) + 1
        return distribution
    
    def generate_enhanced_metrics_dataset(self, target_count: int = None) -> Dict:
        """Generate enhanced metrics dataset."""
        if target_count is None:
            target_count = CONFIG.get('metrics_samples', 10000)
        
        logger.info(f"📊 Starting metrics generation: {target_count} target")
        
        # Check existing
        existing_data = load_dataset_file(self.dataset_paths['metrics_dataset'])
        existing_count = len(existing_data.get('training_samples', [])) if existing_data else 0
        
        if existing_count >= target_count:
            logger.info(f"✅ Metrics target already met: {existing_count}/{target_count}")
            return existing_data
        
        needed = target_count - existing_count
        logger.info(f"📊 Need to generate: {needed} metrics samples")
        
        # Generate metrics samples
        metrics_data = self._generate_metrics_samples(needed, existing_data)
        
        # Save final metrics dataset
        save_dataset_file(metrics_data, self.dataset_paths['metrics_dataset'], 'metrics_dataset')
        
        logger.info(f"✅ Metrics generation complete: {len(metrics_data.get('training_samples', []))} total")
        return metrics_data
    
    def _generate_metrics_samples(self, count: int, existing_data: Dict = None) -> Dict:
        """Generate metrics samples with anomaly patterns."""
        import random
        from datetime import datetime, timedelta
        
        if existing_data:
            samples = existing_data.get('training_samples', [])
        else:
            samples = []
        
        # Load metrics patterns from YAML
        metrics_config = self._get_metrics_config()
        normal_ranges = metrics_config.get('normal_ranges', {})
        anomaly_patterns = metrics_config.get('anomaly_patterns', {})
        server_profiles = metrics_config.get('server_profiles', {})
        
        anomaly_ratio = CONFIG.get('anomaly_ratio', 0.2)
        anomaly_count = int(count * anomaly_ratio)
        normal_count = count - anomaly_count
        
        # Generate normal samples
        for i in range(normal_count):
            timestamp = datetime.now() - timedelta(days=random.randint(1, 30))
            sample = self._generate_normal_metrics_sample(timestamp, normal_ranges, server_profiles)
            samples.append(sample)
        
        # Generate anomaly samples
        for i in range(anomaly_count):
            timestamp = datetime.now() - timedelta(days=random.randint(1, 30))
            sample = self._generate_anomaly_metrics_sample(timestamp, normal_ranges, anomaly_patterns, server_profiles)
            samples.append(sample)
        
        return {
            "training_samples": samples,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_samples": len(samples),
                "anomaly_samples": anomaly_count,
                "normal_samples": normal_count,
                "anomaly_ratio": anomaly_ratio,
                "session_id": self.progress.session_id
            }
        }
   
    def _get_metrics_config(self) -> Dict:
       """Get metrics configuration from YAML files."""
       metrics_config = {
           'normal_ranges': {},
           'anomaly_patterns': {},
           'server_profiles': {}
       }
       
       for config_name, config_data in self.yaml_configs.items():
           data = config_data.get('data', {})
           
           if 'normal_ranges' in data:
               metrics_config['normal_ranges'].update(data['normal_ranges'])
           
           if 'anomaly_patterns' in data:
               metrics_config['anomaly_patterns'].update(data['anomaly_patterns'])
           
           if 'server_profiles' in data:
               metrics_config['server_profiles'].update(data['server_profiles'])
       
       return metrics_config
    
    def _generate_normal_metrics_sample(self, timestamp, normal_ranges, server_profiles):
       """Generate a normal metrics sample."""
       import random
       
       # Pick server profile
       profile_name = random.choice(list(server_profiles.keys())) if server_profiles else "standard_performance"
       profile = server_profiles.get(profile_name, {})
       multipliers = profile.get('base_multipliers', {})
       
       # Generate server name
       server_patterns = [
           f"pprva00a{random.randint(18, 99):04d}",
           f"psrva00a{random.randint(18, 99):04d}",
           f"cppr{random.randint(10, 99):02d}a{random.randint(1000, 9999):04d}",
           f"csrva{random.randint(10, 99):02d}a{random.randint(1000, 9999):04d}",
           f"crva{random.randint(10, 99):02d}a{random.randint(1000, 9999):04d}"
       ]
       server_name = random.choice(server_patterns)
       
       # Generate metrics within normal ranges
       system_metrics = normal_ranges.get('system_metrics', {})
       java_metrics = normal_ranges.get('java_metrics', {})
       
       metrics = {}
       
       # System metrics
       for metric, range_vals in system_metrics.items():
           if len(range_vals) >= 2:
               base_value = random.uniform(range_vals[0], range_vals[1])
               multiplier = multipliers.get(metric, 1.0)
               metrics[metric] = round(base_value * multiplier, 2)
       
       # Java metrics
       for metric, range_vals in java_metrics.items():
           if len(range_vals) >= 2:
               base_value = random.uniform(range_vals[0], range_vals[1])
               multiplier = multipliers.get(metric, 1.0)
               metrics[metric] = round(base_value * multiplier, 2)
       
       # Generate explanation
       explanation = f"System operating normally on {server_name}. All metrics within expected ranges for {profile_name} profile."
       
       return {
           "id": f"sample_{timestamp.strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
           "timestamp": timestamp.isoformat(),
           "server_name": server_name,
           "metrics": metrics,
           "status": "normal",
           "explanation": explanation,
           "server_profile": profile_name
       }
    
    def _generate_anomaly_metrics_sample(self, timestamp, normal_ranges, anomaly_patterns, server_profiles):
       """Generate an anomaly metrics sample."""
       import random
       
       # Pick anomaly pattern
       pattern_name = random.choice(list(anomaly_patterns.keys())) if anomaly_patterns else "cpu_spike"
       pattern = anomaly_patterns.get(pattern_name, {})
       
       # Pick server profile
       profile_name = random.choice(list(server_profiles.keys())) if server_profiles else "standard_performance"
       profile = server_profiles.get(profile_name, {})
       
       # Generate server name
       server_patterns = [
           f"pprva00a{random.randint(18, 99):04d}",
           f"psrva00a{random.randint(18, 99):04d}",
           f"cppr{random.randint(10, 99):02d}a{random.randint(1000, 9999):04d}"
       ]
       server_name = random.choice(server_patterns)
       
       # Start with normal metrics
       system_metrics = normal_ranges.get('system_metrics', {})
       java_metrics = normal_ranges.get('java_metrics', {})
       
       metrics = {}
       
       # Generate base normal metrics
       for metric, range_vals in system_metrics.items():
           if len(range_vals) >= 2:
               metrics[metric] = round(random.uniform(range_vals[0], range_vals[1]), 2)
       
       for metric, range_vals in java_metrics.items():
           if len(range_vals) >= 2:
               metrics[metric] = round(random.uniform(range_vals[0], range_vals[1]), 2)
       
       # Apply anomaly pattern
       pattern_metrics = pattern.get('metrics', {})
       for metric, range_vals in pattern_metrics.items():
           if len(range_vals) >= 2:
               metrics[metric] = round(random.uniform(range_vals[0], range_vals[1]), 2)
       
       # Apply correlated effects
       correlated_effects = pattern.get('correlated_effects', {})
       for metric, range_vals in correlated_effects.items():
           if len(range_vals) >= 2:
               metrics[metric] = round(random.uniform(range_vals[0], range_vals[1]), 2)
       
       # Generate explanation
       description = pattern.get('description', f'{pattern_name} detected')
       explanation = f"ANOMALY: {description} on {server_name}. Metrics indicate {pattern_name.replace('_', ' ')} condition requiring investigation."
       
       return {
           "id": f"anomaly_{timestamp.strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
           "timestamp": timestamp.isoformat(),
           "server_name": server_name,
           "metrics": metrics,
           "status": "anomaly",
           "explanation": explanation,
           "anomaly_pattern": pattern_name,
           "server_profile": profile_name
       }
    
    def reset_progress(self):
       """Reset generation progress."""
       if self.progress_file.exists():
           self.progress_file.unlink()
       
       # Reinitialize progress
       self.progress = self._load_or_create_progress()
       self.generation_stats = {
           "samples_generated": 0,
           "api_calls": 0,
           "errors": 0,
           "saves_completed": 0,
           "models_per_question": CONFIG.get('models_per_question', 2)
       }
       
       logger.info("✅ Progress reset")
    
    def retry_failed(self):
       """Retry failed generation items."""
       if hasattr(self.progress, 'failed_items'):
           failed_count = len(self.progress.failed_items)
           self.progress.failed_items.clear()
           self._save_progress_pickle()
           logger.info(f"✅ Cleared {failed_count} failed items for retry")
       else:
           logger.info("✅ No failed items to retry")

# Backwards compatibility
DatasetGenerator = EnhancedDatasetGenerator

# Export functions for notebook interface
def generate_datasets(self, language_count: int = None, metrics_count: int = None):
    """Generate training datasets using enhanced generator."""
    if not self.setup_complete:
        print("❌ Run setup() first")
        return False, False
    
    print("\n📊 DATASET GENERATION")
    print("="*50)
    
    try:
        # Use the corrected method that returns success status
        language_success, metrics_success = self.generator.generate_complete_dataset(
            language_count, metrics_count
        )
        
        # Update status using common utilities
        self.datasets_exist = self._check_datasets_using_common_utils()
        
        if language_success and metrics_success:
            print("✅ Dataset generation completed!")
            return True, True
        else:
            print("❌ Dataset generation failed")
            return False, False
            
    except KeyboardInterrupt:
        print("\n⏸️  Generation interrupted - progress saved")
        return False, False
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        logger.error(f"Dataset generation error: {e}")
        return False, False

def show_progress():
   """Show enhanced progress with multiplier display."""
   generator = EnhancedDatasetGenerator()
   generator.show_progress_with_multiplier()

def reset_progress():
   """Reset progress."""
   generator = EnhancedDatasetGenerator()
   generator.reset_progress()

def retry_failed():
   """Retry failed items."""
   generator = EnhancedDatasetGenerator()
   generator.retry_failed()