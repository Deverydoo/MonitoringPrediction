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
from dataclasses import dataclass, asdict
from config import CONFIG, model_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationProgress:
    """Progress tracking for resumable generation."""
    session_id: str
    started_at: str
    last_updated: str
    language_completed: int = 0
    metrics_completed: int = 0
    failed_items: List[str] = None
    
    def __post_init__(self):
        if self.failed_items is None:
            self.failed_items = []

class DatasetGenerator:
    """Enhanced generator with frequent saves and dynamic YAML discovery."""
    
    def __init__(self, config_dir: str = None):
        self.config_dir = Path(config_dir or CONFIG.get('yaml_config_dir', './data_config'))
        self.progress_file = Path("./generation_progress.pkl")
        
        # Enhanced discovery and caching
        self.yaml_discovery_cache = {}
        self.last_yaml_scan = 0
        self.yaml_scan_interval = CONFIG.get('yaml_discovery_interval', 300)  # 5 minutes
        
        # Dynamic YAML discovery
        self.config_data = self._discover_and_load_yaml_files()
        self.progress = self._load_or_create_progress()
        
        # Optimization settings
        self.rotation_round_size = CONFIG.get('rotation_round_size', 50)
        self.samples_since_save = 0
        
        # Error tracking
        self.failed_terms = set()
        self.failed_errors = []
        self.generation_stats = {
            "samples_generated": 0,
            "api_calls": 0,
            "errors": 0,
            "skipped_terms": 0,
            "rotation_rounds": 0,
            "json_saves": 0
        }
        
        logger.info(f"✅ Generator initialized with {len(self.config_data)} YAML files")
    
    def _discover_and_load_yaml_files(self) -> Dict[str, Any]:
        """Dynamic YAML discovery with caching and change detection."""
        current_time = time.time()
        
        # Use cache if recent and no changes
        if (current_time - self.last_yaml_scan) < self.yaml_scan_interval:
            if self.yaml_discovery_cache:
                return self.yaml_discovery_cache
        
        self.config_dir.mkdir(exist_ok=True)
        config_data = {}
        
        # Efficient glob-based discovery
        yaml_patterns = ["*.yaml", "*.yml"]
        yaml_files = []
        for pattern in yaml_patterns:
            yaml_files.extend(self.config_dir.glob(pattern))
        
        logger.info(f"📂 Dynamically discovering {len(yaml_files)} YAML files in {self.config_dir}")
        
        # Parallel-style processing (if needed, can add threading here)
        for file_path in yaml_files:
            try:
                # Check modification time for efficiency
                file_mtime = file_path.stat().st_mtime
                cache_key = f"{file_path.name}_{file_mtime}"
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                if data:
                    cleaned_data = self._clean_and_validate_yaml(data, file_path.stem)
                    if cleaned_data:
                        item_count = self._count_items(cleaned_data)
                        config_data[file_path.stem] = {
                            'data': cleaned_data,
                            'category': self._categorize_yaml(file_path.stem),
                            'item_count': item_count,
                            'file_path': str(file_path),
                            'modified_time': file_mtime
                        }
                        logger.info(f"  ✅ {file_path.name}: {item_count} items")
                else:
                    logger.warning(f"  ⚠️  {file_path.name}: Empty file")
                    
            except Exception as e:
                logger.error(f"  ❌ {file_path.name}: {e}")
        
        # Update cache
        self.yaml_discovery_cache = config_data
        self.last_yaml_scan = current_time
        
        logger.info(f"📊 Dynamically loaded {len(config_data)} YAML configurations")
        return config_data
    
    def save_after_rotation_round(self, samples: List[Dict]):
        """Save JSON after each complete rotation round."""
        if len(samples) % self.rotation_round_size == 0 and samples:
            try:
                self._save_language_dataset_incremental(samples[-self.rotation_round_size:])
                self.generation_stats["json_saves"] += 1
                self.generation_stats["rotation_rounds"] += 1
                
                logger.info(f"💾 Rotation round {self.generation_stats['rotation_rounds']} saved")
                logger.info(f"   Total saves: {self.generation_stats['json_saves']}")
                
                # Also save progress
                self._save_progress()
                
            except Exception as e:
                logger.error(f"Failed to save rotation round: {e}")
    
    def _save_language_dataset_incremental(self, new_samples: List[Dict]):
        """Incremental save with atomic operation."""
        output_file = Path(CONFIG['training_dir']) / 'language_dataset.json'
        temp_file = output_file.with_suffix('.json.tmp')
        
        # Load existing
        existing_samples = []
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    existing_samples = data.get('samples', []) if isinstance(data, dict) else data
            except Exception as e:
                logger.warning(f"Could not load existing samples: {e}")
        
        # Combine with new
        all_samples = existing_samples + new_samples
        
        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_samples": len(all_samples),
                "new_samples_this_save": len(new_samples),
                "session_id": self.progress.session_id,
                "generation_stats": self.generation_stats,
                "yaml_files_count": len(self.config_data)
            },
            "samples": all_samples
        }
        
        # Atomic save operation
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            
            # Atomic move
            temp_file.replace(output_file)
            
            logger.debug(f"💾 Incremental save: +{len(new_samples)} samples, {len(all_samples)} total")
            
        except Exception as e:
            logger.error(f"Atomic save failed: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise
    
    def generate_language_dataset_optimized(self, target_count: int = None) -> List[Dict]:
        """Optimized generation with frequent saves and dynamic discovery."""
        # Refresh YAML files periodically
        if time.time() - self.last_yaml_scan > self.yaml_scan_interval:
            logger.info("🔄 Refreshing YAML configuration files...")
            self.config_data = self._discover_and_load_yaml_files()
        
        if target_count is None:
            targets, total_target = self._calculate_dynamic_targets()
            existing = self._analyze_existing_dataset()
            
            needed = {}
            for sample_type, target in targets.items():
                current = existing.get(sample_type, 0)
                needed[sample_type] = max(0, target - current)
            
            target_count = sum(needed.values())
            
            if target_count == 0:
                logger.info("✅ All language samples already complete!")
                return []
        else:
            targets, _ = self._calculate_dynamic_targets()
            existing = self._analyze_existing_dataset()
            needed = {}
            for sample_type, target in targets.items():
                current = existing.get(sample_type, 0)
                needed[sample_type] = max(0, target - current)
        
        logger.info(f"🗣️ Optimized generation: {target_count} samples with rotation saves")
        
        # Enhanced rotation with frequent saves
        active_types = [sample_type for sample_type, count in needed.items() if count > 0]
        if not active_types:
            logger.info("✅ All sample types complete!")
            return []
        
        logger.info(f"🔄 Active types: {active_types}")
        logger.info(f"💾 Will save JSON every {self.rotation_round_size} samples")
        
        completed_per_type = {sample_type: 0 for sample_type in active_types}
        rotation_counter = 0
        current_type_index = 0
        samples = []
        
        all_terms = self._extract_all_technical_terms()
        if not all_terms:
            logger.error("No technical terms found in YAML files")
            return []
        
        # Main generation loop with optimizations
        while sum(completed_per_type.values()) < target_count and active_types:
            current_type = active_types[current_type_index]
            
            # Check completion
            if completed_per_type[current_type] >= needed[current_type]:
                active_types.remove(current_type)
                if not active_types:
                    break
                current_type_index = current_type_index % len(active_types)
                continue
            
            try:
                # Generate with performance tracking
                start_time = time.time()
                sample = self._generate_sample_for_type(current_type, all_terms)
                generation_time = time.time() - start_time
                
                if sample:
                    samples.append(sample)
                    completed_per_type[current_type] += 1
                    self.generation_stats["samples_generated"] += 1
                    self.progress.language_completed += 1
                    self.samples_since_save += 1
                    
                    # Save after rotation round completion
                    if self.samples_since_save >= self.rotation_round_size:
                        self.save_after_rotation_round(samples)
                        self.samples_since_save = 0
                    
                    # Progress logging (less frequent)
                    total_completed = sum(completed_per_type.values())
                    if total_completed % 100 == 0:  # Every 100 instead of 25
                        self._log_distributed_progress(completed_per_type, needed, total_completed, target_count)
                        logger.info(f"⚡ Avg generation time: {generation_time:.1f}s")
            
            except Exception as e:
                logger.error(f"Error generating {current_type} sample: {e}")
                self.generation_stats["errors"] += 1
            
            # Efficient rotation (every 20 instead of 10)
            rotation_counter += 1
            if rotation_counter >= 20:
                rotation_counter = 0
                current_type_index = (current_type_index + 1) % len(active_types)
        
        # Final save
        if self.samples_since_save > 0:
            self.save_after_rotation_round(samples)
        
        # Final summary
        self._log_final_distribution_summary(completed_per_type, needed)
        
        return samples
    
    
    def _categorize_yaml(self, filename: str) -> str:
        """Categorize YAML file for better organization."""
        category_mapping = {
            'technical_terms': 'terminology',
            'error_patterns': 'errors',
            'conversation_prompts': 'conversations',
            'metrics_patterns': 'metrics',
            'personality_types': 'personalities',
            'question_styles': 'questions',
            'english_patterns': 'language',
            'response_templates': 'templates',
            'project_management_terms': 'business',
            'splunk_queries': 'data_sources',
            'jira_patterns': 'data_sources',
            'spectrum_config': 'data_sources'
        }
        return category_mapping.get(filename, 'general')
    
    def _clean_and_validate_yaml(self, data: Any, filename: str) -> Any:
        """Clean and validate YAML data based on expected structure."""
        cleaned = self._clean_data(data)
        
        # Apply specific validation based on filename patterns
        if 'terms' in filename or filename == 'technical_terms':
            return self._validate_terms_structure(cleaned)
        elif 'error' in filename:
            return self._validate_error_structure(cleaned)
        elif 'patterns' in filename:
            return self._validate_pattern_structure(cleaned)
        else:
            return cleaned
    
    def _clean_data(self, data: Any) -> Any:
        """Recursively clean data of None values and empty structures."""
        if data is None:
            return None
        elif isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                if key is not None and value is not None:
                    cleaned_value = self._clean_data(value)
                    if cleaned_value not in [None, {}, []]:
                        cleaned[key] = cleaned_value
            return cleaned
        elif isinstance(data, list):
            cleaned = []
            for item in data:
                if item is not None:
                    cleaned_item = self._clean_data(item)
                    if cleaned_item not in [None, {}, []]:
                        cleaned.append(cleaned_item)
            return cleaned
        else:
            return data
    
    def _validate_terms_structure(self, data: Dict) -> Dict:
        """Validate technical terms structure."""
        validated = {}
        for category, terms in data.items():
            if isinstance(terms, list):
                valid_terms = [
                    term.strip() for term in terms 
                    if isinstance(term, str) and len(term.strip()) > 1
                ]
                if valid_terms:
                    validated[category] = valid_terms
        return validated
    
    def _validate_error_structure(self, data: Dict) -> Dict:
        """Validate error patterns structure."""
        validated = {}
        
        def process_errors(obj: Any, path: str = ""):
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    processed = process_errors(value, f"{path}.{key}")
                    if processed:
                        result[key] = processed
                return result if result else None
            elif isinstance(obj, list):
                valid_items = [
                    item.strip() if isinstance(item, str) else item
                    for item in obj
                    if item and (not isinstance(item, str) or len(item.strip()) > 1)
                ]
                return valid_items if valid_items else None
            else:
                return obj
        
        return process_errors(data) or {}
    
    def _validate_pattern_structure(self, data: Dict) -> Dict:
        """Validate generic pattern structure."""
        validated = {}
        for key, value in data.items():
            if value is None:
                continue
            elif isinstance(value, dict):
                cleaned_dict = self._validate_pattern_structure(value)
                if cleaned_dict:
                    validated[key] = cleaned_dict
            elif isinstance(value, list):
                cleaned_list = [
                    item for item in value 
                    if item is not None and (not isinstance(item, str) or len(str(item).strip()) > 0)
                ]
                if cleaned_list:
                    validated[key] = cleaned_list
            else:
                validated[key] = value
        return validated
    
    def _count_items(self, data: Any) -> int:
        """Count total items in data structure."""
        if isinstance(data, dict):
            return sum(self._count_items(v) for v in data.values())
        elif isinstance(data, list):
            return len(data)
        else:
            return 1 if data is not None else 0
    
    def _load_or_create_progress(self) -> GenerationProgress:
        """Load existing progress or create new session."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'rb') as f:
                    progress = pickle.load(f)
                logger.info(f"📊 Resuming session {progress.session_id}")
                return progress
            except Exception as e:
                logger.error(f"Error loading progress: {e}")
        
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"🆕 Starting new session: {session_id}")
        
        return GenerationProgress(
            session_id=session_id,
            started_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )
    
    def _save_progress(self):
        """Save current progress."""
        self.progress.last_updated = datetime.now().isoformat()
        try:
            with open(self.progress_file, 'wb') as f:
                pickle.dump(self.progress, f)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def _extract_all_technical_terms(self) -> List[Tuple[str, str]]:
        """Extract technical terms from all YAML files."""
        all_terms = []
        
        for file_name, file_data in self.config_data.items():
            data = file_data.get('data', {})
            category = file_data.get('category', 'general')
            
            # Extract based on file type
            if file_name == 'technical_terms':
                for cat, terms in data.items():
                    if isinstance(terms, list):
                        all_terms.extend([(term.strip(), cat) for term in terms if term])
            else:
                # Extract from other structures
                terms_found = self._extract_terms_from_structure(data, category)
                all_terms.extend(terms_found)
        
        # Remove duplicates and failed terms
        unique_terms = []
        seen = set()
        for term, category in all_terms:
            if term not in seen and term not in self.failed_terms:
                unique_terms.append((term, category))
                seen.add(term)
        
        logger.info(f"📋 Extracted {len(unique_terms)} unique technical terms")
        return unique_terms
    
    def _extract_terms_from_structure(self, data: Any, category: str) -> List[Tuple[str, str]]:
        """Extract potential technical terms from any data structure."""
        terms = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list) and value and isinstance(value[0], str):
                    # Likely a list of terms
                    terms.extend([(item.strip(), category) for item in value 
                                 if isinstance(item, str) and len(item.strip()) > 2])
                elif isinstance(value, dict):
                    terms.extend(self._extract_terms_from_structure(value, category))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str) and len(item.strip()) > 2:
                    terms.append((item.strip(), category))
        
        return terms
    
    def _calculate_dynamic_targets(self) -> Tuple[Dict[str, int], int]:
        """Calculate generation targets based on discovered YAML content."""
        targets = {}
        
        # Base calculations on actual content
        for file_name, file_data in self.config_data.items():
            item_count = file_data.get('item_count', 0)
            category = file_data.get('category', 'general')
            
            # Map categories to sample types
            if category == 'terminology':
                targets['technical_explanation'] = targets.get('technical_explanation', 0) + min(item_count, 500)
            elif category == 'errors':
                targets['error_interpretation'] = targets.get('error_interpretation', 0) + min(item_count * 2, 300)
            elif category == 'conversations':
                targets['conversational_samples'] = targets.get('conversational_samples', 0) + min(item_count // 2, 200)
            elif category == 'language':
                targets['english_language_samples'] = targets.get('english_language_samples', 0) + min(item_count, 100)
            elif category == 'questions':
                targets['question_style_samples'] = targets.get('question_style_samples', 0) + min(item_count * 3, 150)
            elif category == 'business':
                targets['business_terminology'] = targets.get('business_terminology', 0) + min(item_count // 10, 100)
        
        # Apply multiplier for variety
        variety_multiplier = CONFIG.get('variety_multiplier', 1.5)
        for key in targets:
            targets[key] = int(targets[key] * variety_multiplier)
        
        total_target = sum(targets.values())
        
        logger.info(f"📊 Dynamic targets calculated: {total_target} total samples")
        for category, target in targets.items():
            logger.info(f"  {category}: {target}")
        
        return targets, total_target
    
    def _analyze_existing_dataset(self) -> Dict[str, int]:
        """Analyze existing dataset to avoid duplicates."""
        existing = {}
        
        lang_file = Path(CONFIG['training_dir']) / 'language_dataset.json'
        if lang_file.exists():
            try:
                with open(lang_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                samples = data.get('samples', []) if isinstance(data, dict) else data
                
                for sample in samples:
                    sample_type = sample.get('type', 'unknown')
                    existing[sample_type] = existing.get(sample_type, 0) + 1
                
                logger.info(f"📋 Found {sum(existing.values())} existing samples")
                
            except Exception as e:
                logger.warning(f"Could not analyze existing dataset: {e}")
        
        return existing
    
    def generate_language_dataset(self, target_count: int = None) -> List[Dict]:
        """Generate language training samples with distributed rotation across types."""
        if target_count is None:
            targets, total_target = self._calculate_dynamic_targets()
            existing = self._analyze_existing_dataset()
            
            # Calculate needed samples
            needed = {}
            for sample_type, target in targets.items():
                current = existing.get(sample_type, 0)
                needed[sample_type] = max(0, target - current)
            
            target_count = sum(needed.values())
            
            if target_count == 0:
                logger.info("✅ All language samples already complete!")
                return []
        else:
            # If target_count is specified, calculate distribution
            targets, _ = self._calculate_dynamic_targets()
            existing = self._analyze_existing_dataset()
            needed = {}
            for sample_type, target in targets.items():
                current = existing.get(sample_type, 0)
                needed[sample_type] = max(0, target - current)
        
        logger.info(f"🗣️ Generating {target_count} language samples with distributed rotation")
        
        # Create rotation queue for active sample types
        active_types = [sample_type for sample_type, count in needed.items() if count > 0]
        if not active_types:
            logger.info("✅ All sample types complete!")
            return []
        
        logger.info(f"🔄 Active types in rotation: {active_types}")
        
        # Track completion per type
        completed_per_type = {sample_type: 0 for sample_type in active_types}
        rotation_counter = 0
        current_type_index = 0
        samples = []
        
        all_terms = self._extract_all_technical_terms()
        if not all_terms:
            logger.error("No technical terms found in YAML files")
            return []
        
        # Generation loop with rotation every 10 samples
        while sum(completed_per_type.values()) < target_count and active_types:
            current_type = active_types[current_type_index]
            
            # Check if current type is complete
            if completed_per_type[current_type] >= needed[current_type]:
                # Remove completed type from rotation
                active_types.remove(current_type)
                if not active_types:
                    break
                current_type_index = current_type_index % len(active_types)
                continue
            
            try:
                # Generate sample for current type
                sample = self._generate_sample_for_type(current_type, all_terms)
                
                if sample:
                    samples.append(sample)
                    completed_per_type[current_type] += 1
                    self.generation_stats["samples_generated"] += 1
                    self.progress.language_completed += 1
                    
                    # Progress logging
                    total_completed = sum(completed_per_type.values())
                    if total_completed % 25 == 0:
                        self._log_distributed_progress(completed_per_type, needed, total_completed, target_count)
                        self._save_progress()
            
            except Exception as e:
                logger.error(f"Error generating {current_type} sample: {e}")
                self.generation_stats["errors"] += 1
            
            # Rotate to next type every 10 generations
            rotation_counter += 1
            if rotation_counter >= 10:
                rotation_counter = 0
                current_type_index = (current_type_index + 1) % len(active_types)
                logger.debug(f"🔄 Rotating to {active_types[current_type_index] if active_types else 'none'}")
        
        # Final progress save and summary
        self._save_language_dataset(samples)
        self._log_final_distribution_summary(completed_per_type, needed)
        
        return samples

    def _generate_sample_for_type(self, sample_type: str, all_terms: List[Tuple[str, str]]) -> Optional[Dict]:
        """Generate a sample for a specific type."""
        try:
            if sample_type == 'technical_explanation':
                return self._generate_technical_explanation_sample(all_terms)
            elif sample_type == 'error_interpretation':
                return self._generate_error_interpretation_sample(all_terms)
            elif sample_type == 'conversational_samples':
                return self._generate_conversational_sample(all_terms)
            elif sample_type == 'english_language_samples':
                return self._generate_english_language_sample(all_terms)
            elif sample_type == 'question_style_samples':
                return self._generate_question_style_sample(all_terms)
            elif sample_type == 'business_terminology':
                return self._generate_business_terminology_sample(all_terms)
            else:
                # Default to technical explanation
                return self._generate_technical_explanation_sample(all_terms)
        
        except Exception as e:
            logger.error(f"Error generating {sample_type}: {e}")
            return None

    def _generate_english_language_sample(self, all_terms: List[Tuple[str, str]]) -> Optional[Dict]:
        """Generate English language pattern sample."""
        english_data = self.config_data.get('english_patterns', {}).get('data', {})
        templates = english_data.get('templates', {})
        
        if not templates:
            return self._generate_technical_explanation_sample(all_terms)
        
        template_category = random.choice(list(templates.keys()))
        template_list = templates[template_category]
        template = random.choice(template_list)
        
        term, category = random.choice(all_terms)
        prompt = template.format(topic=term)
        
        responses = model_chain.generate_responses(prompt, max_tokens=200)
        
        if responses:
            best_response = max(responses, key=lambda r: r.get('quality_score', 0))
            
            if len(best_response['response']) > CONFIG.get('response_quality_threshold', 15):
                return {
                    "type": "english_language_samples",
                    "term": term,
                    "category": category,
                    "template_category": template_category,
                    "prompt": prompt,
                    "response": best_response['response'],
                    "model": best_response.get('model', 'unknown'),
                    "session_id": self.progress.session_id,
                    "timestamp": datetime.now().isoformat()
                }
        
        return None
    
    def _generate_question_style_sample(self, all_terms: List[Tuple[str, str]]) -> Optional[Dict]:
        """Generate question style sample."""
        question_data = self.config_data.get('question_styles', {}).get('data', {})
        styles = question_data.get('styles', [])
        
        if not styles:
            return self._generate_technical_explanation_sample(all_terms)
        
        style = random.choice(styles)
        term, category = random.choice(all_terms)
        
        pattern = style.get('pattern', 'What is {topic}?')
        prompt = pattern.format(topic=term)
        
        responses = model_chain.generate_responses(prompt, max_tokens=200)
        
        if responses:
            best_response = max(responses, key=lambda r: r.get('quality_score', 0))
            
            if len(best_response['response']) > CONFIG.get('response_quality_threshold', 15):
                return {
                    "type": "question_style_samples",
                    "term": term,
                    "category": category,
                    "question_style": style.get('name', 'unknown'),
                    "prompt": prompt,
                    "response": best_response['response'],
                    "model": best_response.get('model', 'unknown'),
                    "session_id": self.progress.session_id,
                    "timestamp": datetime.now().isoformat()
                }
        
        return None
    
    def _generate_business_terminology_sample(self, all_terms: List[Tuple[str, str]]) -> Optional[Dict]:
        """Generate business terminology sample."""
        business_data = self.config_data.get('project_management_terms', {}).get('data', {})
        
        if not business_data:
            return self._generate_technical_explanation_sample(all_terms)
        
        # Extract business terms
        business_terms = []
        for category, terms in business_data.items():
            if isinstance(terms, list):
                business_terms.extend([(term, category) for term in terms])
        
        if business_terms:
            term, category = random.choice(business_terms)
        else:
            term, category = random.choice(all_terms)
        
        prompt = f"From a business perspective, explain '{term}' in {category}. Include business impact, ROI considerations, and stakeholder concerns."
        
        responses = model_chain.generate_responses(prompt, max_tokens=250)
        
        if responses:
            best_response = max(responses, key=lambda r: r.get('quality_score', 0))
            
            if len(best_response['response']) > CONFIG.get('response_quality_threshold', 15):
                return {
                    "type": "business_terminology",
                    "term": term,
                    "category": category,
                    "prompt": prompt,
                    "response": best_response['response'],
                    "model": best_response.get('model', 'unknown'),
                    "session_id": self.progress.session_id,
                    "timestamp": datetime.now().isoformat()
                }
        
        return None

    def _log_distributed_progress(self, completed: Dict[str, int], needed: Dict[str, int], total_completed: int, target_count: int):
        """Log progress with distribution details."""
        
        logger.info(f"📊 Progress: {total_completed}/{target_count} ({total_completed/target_count*100:.1f}%)")
        
        for sample_type in completed.keys():
            current = completed[sample_type]
            target = needed[sample_type]
            percent = (current / target * 100) if target > 0 else 100
            status = "✅" if current >= target else "🔄"
            logger.info(f"  {status} {sample_type}: {current}/{target} ({percent:.1f}%)")
    
    def _log_final_distribution_summary(self, completed: Dict[str, int], needed: Dict[str, int]):
        """Log final distribution summary."""
        logger.info("📊 Final Distribution Summary:")
        
        for sample_type, generated in completed.items():
            target = needed[sample_type]
            percent = (generated / target * 100) if target > 0 else 100
            status = "✅" if generated >= target else "⚠️"
            logger.info(f"  {status} {sample_type}: {generated}/{target} ({percent:.1f}%)")
            
    def _generate_conversational_sample(self, all_terms: List[Tuple[str, str]]) -> Optional[Dict]:
        """Generate conversational sample."""
        conv_data = self.config_data.get('conversation_prompts', {}).get('data', {})
        styles = conv_data.get('conversation_styles', ['casual_explanation'])
        types = conv_data.get('conversation_types', ['technical_help'])
        
        term, category = random.choice(all_terms)
        style = random.choice(styles)
        conv_type = random.choice(types)
        
        prompt = f"In a {style} {conv_type} conversation, explain {term} to a colleague in {category}."
        
        responses = model_chain.generate_responses(prompt, max_tokens=250)
        
        if responses:
            best_response = max(responses, key=lambda r: r.get('quality_score', 0))
            
            if len(best_response['response']) > CONFIG.get('response_quality_threshold', 15):
                return {
                    "type": "conversational_samples",
                    "term": term,
                    "category": category,
                    "style": style,
                    "conversation_type": conv_type,
                    "prompt": prompt,
                    "response": best_response['response'],
                    "model": best_response.get('model', 'unknown'),
                    "session_id": self.progress.session_id,
                    "timestamp": datetime.now().isoformat()
                }
        
        return None
        
    def _generate_error_interpretation_sample(self, all_terms: List[Tuple[str, str]]) -> Optional[Dict]:
        """Generate error interpretation sample."""
        # Get error patterns from config
        error_patterns = []
        for file_name, file_data in self.config_data.items():
            if 'error' in file_name or file_data.get('category') == 'errors':
                data = file_data.get('data', {})
                errors = self._extract_errors_from_structure(data)
                error_patterns.extend(errors)
        
        if not error_patterns:
            # Fallback to technical term
            return self._generate_technical_explanation_sample(all_terms)
        
        error, category = random.choice(error_patterns)
        
        prompt = f"Explain the error '{error}' in {category} systems. Include what causes it, how to diagnose it, and step-by-step resolution."
        
        responses = model_chain.generate_responses(prompt, max_tokens=350)
        
        if responses:
            best_response = max(responses, key=lambda r: r.get('quality_score', 0))
            
            if len(best_response['response']) > CONFIG.get('response_quality_threshold', 15):
                return {
                    "type": "error_interpretation",
                    "error": error,
                    "category": category,
                    "prompt": prompt,
                    "response": best_response['response'],
                    "model": best_response.get('model', 'unknown'),
                    "session_id": self.progress.session_id,
                    "timestamp": datetime.now().isoformat()
                }
        
        return None

    def _generate_technical_explanation_sample(self, all_terms: List[Tuple[str, str]]) -> Optional[Dict]:
        """Generate technical explanation sample."""
        term, category = random.choice(all_terms)
        
        if term in self.failed_terms:
            self.generation_stats["skipped_terms"] += 1
            return None
        
        prompt = f"Provide a comprehensive technical explanation of '{term}' in {category}. Include purpose, common issues, monitoring importance, and troubleshooting steps."
        
        responses = model_chain.generate_responses(prompt, max_tokens=300)
        
        if responses:
            best_response = max(responses, key=lambda r: r.get('quality_score', 0))
            
            if len(best_response['response']) > CONFIG.get('response_quality_threshold', 15):
                return {
                    "type": "technical_explanation",
                    "term": term,
                    "category": category,
                    "prompt": prompt,
                    "response": best_response['response'],
                    "model": best_response.get('model', 'unknown'),
                    "session_id": self.progress.session_id,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                self.failed_terms.add(term)
        else:
            self.failed_terms.add(term)
        
        return None

    def generate_conversational_samples(self, target_count: int = 100) -> List[Dict]:
        """Generate conversational interaction samples."""
        samples = []
        
        # Get conversation prompts
        conv_data = self.config_data.get('conversation_prompts', {}).get('data', {})
        if not conv_data:
            logger.warning("No conversation prompts found")
            return samples
        
        # Extract all conversation elements
        styles = conv_data.get('conversation_styles', [])
        types = conv_data.get('conversation_types', [])
        scenarios = conv_data.get('error_scenarios', {})
        
        for i in range(target_count):
            try:
                # Mix different conversation elements
                style = random.choice(styles) if styles else 'casual'
                conv_type = random.choice(types) if types else 'technical_help'
                
                # Get a technical term for context
                terms = self._extract_all_technical_terms()
                if terms:
                    term, category = random.choice(terms)
                    
                    # Create conversational prompts
                    prompts = [
                        f"In a {style} {conv_type} conversation, explain {term} to a colleague.",
                        f"During a {conv_type}, someone asks about {term}. Respond in a {style} manner.",
                        f"You're having a {style} chat about {category}. Explain {term} naturally.",
                    ]
                    
                    prompt = random.choice(prompts)
                    responses = model_chain.generate_responses(prompt, max_tokens=250)
                    
                    if responses:
                        best_response = max(responses, key=lambda r: r.get('quality_score', 0))
                        
                        sample = {
                            "type": "conversational_samples",
                            "style": style,
                            "conversation_type": conv_type,
                            "term": term,
                            "prompt": prompt,
                            "response": best_response['response'],
                            "model": best_response.get('model', 'unknown'),
                            "timestamp": datetime.now().isoformat()
                        }
                        samples.append(sample)
                        
            except Exception as e:
                logger.error(f"Error generating conversational sample {i}: {e}")
                continue
        
        return samples

    def generate_specialized_datasets(self) -> Dict[str, List[Dict]]:
        """Generate specialized datasets based on discovered YAML content."""
        specialized_samples = {}
        
        # Generate samples for each discovered category
        for file_name, file_data in self.config_data.items():
            category = file_data.get('category', 'general')
            item_count = file_data.get('item_count', 0)
            
            if item_count == 0:
                continue
            
            # Calculate samples to generate
            sample_count = min(item_count * 2, 100)  # Cap at 100 per file
            
            logger.info(f"Generating {sample_count} samples for {file_name}")
            
            if category == 'errors':
                samples = self.generate_error_interpretation_samples(sample_count)
            elif category == 'conversations':
                samples = self.generate_conversational_samples(sample_count)
            elif category == 'data_sources':
                samples = self._generate_data_source_samples(file_name, sample_count)
            else:
                # Default technical explanation
                samples = self._generate_category_samples(file_name, sample_count)
            
            if samples:
                specialized_samples[file_name] = samples
        
        return specialized_samples

    def _generate_data_source_samples(self, source_name: str, count: int) -> List[Dict]:
        """Generate samples for data source specific content (Splunk, Jira, etc)."""
        samples = []
        data = self.config_data.get(source_name, {}).get('data', {})
        
        if 'splunk' in source_name.lower():
            # Generate Splunk query explanations
            queries = self._extract_splunk_queries(data)
            for i in range(min(count, len(queries) * 2)):
                query = random.choice(queries)
                prompt = f"Explain this Splunk query and what it monitors: {query}"
                
                responses = model_chain.generate_responses(prompt, max_tokens=200)
                if responses:
                    best = max(responses, key=lambda r: r.get('quality_score', 0))
                    samples.append({
                        "type": "splunk_query_explanation",
                        "query": query,
                        "response": best['response'],
                        "timestamp": datetime.now().isoformat()
                    })
        
        elif 'jira' in source_name.lower():
            # Generate Jira workflow explanations
            patterns = self._extract_jira_patterns(data)
            for i in range(min(count, len(patterns) * 2)):
                pattern = random.choice(patterns)
                prompt = f"Explain this Jira workflow pattern and best practices: {pattern}"
                
                responses = model_chain.generate_responses(prompt, max_tokens=200)
                if responses:
                    best = max(responses, key=lambda r: r.get('quality_score', 0))
                    samples.append({
                        "type": "jira_pattern_explanation",
                        "pattern": pattern,
                        "response": best['response'],
                        "timestamp": datetime.now().isoformat()
                    })
        
        return samples

    def _extract_splunk_queries(self, data: Any) -> List[str]:
        """Extract Splunk queries from data."""
        queries = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and ('index=' in value or 'search' in value):
                    queries.append(value)
                elif isinstance(value, list):
                    queries.extend([q for q in value if isinstance(q, str) and 'index=' in q])
        
        return queries
    
    def _extract_jira_patterns(self, data: Any) -> List[str]:
        """Extract Jira patterns from data."""
        patterns = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    patterns.extend([str(p) for p in value if p])
                elif isinstance(value, str):
                    patterns.append(value)
        
        return patterns

    def validate_dataset_quality(self) -> Dict[str, Any]:
        """Validate the quality of generated datasets."""
        report = {
            "language_dataset": {},
            "metrics_dataset": {},
            "quality_issues": [],
            "recommendations": []
        }
        
        # Check language dataset
        lang_file = Path(CONFIG['training_dir']) / 'language_dataset.json'
        if lang_file.exists():
            with open(lang_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                samples = data.get('samples', [])
                
                # Analyze quality metrics
                response_lengths = [len(s.get('response', '')) for s in samples]
                avg_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
                
                # Check for diversity
                unique_terms = len(set(s.get('term', '') for s in samples))
                unique_models = len(set(s.get('model', '') for s in samples))
                
                report['language_dataset'] = {
                    'total_samples': len(samples),
                    'average_response_length': avg_length,
                    'unique_terms': unique_terms,
                    'unique_models': unique_models,
                    'sample_types': {}
                }
                
                # Count by type
                for sample in samples:
                    sample_type = sample.get('type', 'unknown')
                    report['language_dataset']['sample_types'][sample_type] = \
                        report['language_dataset']['sample_types'].get(sample_type, 0) + 1
                
                # Quality checks
                if avg_length < 100:
                    report['quality_issues'].append("Average response length is too short")
                    report['recommendations'].append("Increase max_tokens in generation")
                
                if unique_models < 2:
                    report['quality_issues'].append("Low model diversity")
                    report['recommendations'].append("Ensure multiple models are available")
        
        # Check metrics dataset
        metrics_file = Path(CONFIG['training_dir']) / 'metrics_dataset.json'
        if metrics_file.exists():
            with open(metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                samples = data.get('training_samples', [])
                
                # Analyze distribution
                normal_count = sum(1 for s in samples if s.get('status') == 'normal')
                anomaly_count = sum(1 for s in samples if s.get('status') == 'anomaly')
                
                report['metrics_dataset'] = {
                    'total_samples': len(samples),
                    'normal_samples': normal_count,
                    'anomaly_samples': anomaly_count,
                    'anomaly_ratio': anomaly_count / len(samples) if samples else 0
                }
                
                # Check balance
                if report['metrics_dataset']['anomaly_ratio'] < 0.15:
                    report['quality_issues'].append("Low anomaly ratio")
                    report['recommendations'].append("Increase anomaly_ratio in CONFIG")
        
        return report
    
    def export_for_training(self, output_format: str = 'pytorch') -> str:
        """Export datasets in format ready for training."""
        output_dir = Path(CONFIG['training_dir']) / 'exported'
        output_dir.mkdir(exist_ok=True)
        
        if output_format == 'pytorch':
            # Export as PyTorch tensors
            import torch
            from transformers import AutoTokenizer
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
            
            # Process language dataset
            lang_file = Path(CONFIG['training_dir']) / 'language_dataset.json'
            if lang_file.exists():
                with open(lang_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    samples = data.get('samples', [])
                
                # Tokenize
                texts = [s.get('response', '') for s in samples]
                encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
                
                # Save as tensor
                torch.save({
                    'input_ids': torch.tensor(encodings['input_ids']),
                    'attention_mask': torch.tensor(encodings['attention_mask']),
                    'metadata': [{'type': s.get('type'), 'term': s.get('term')} for s in samples]
                }, output_dir / 'language_dataset.pt')
            
            logger.info(f"✅ Exported datasets to {output_dir}")
            return str(output_dir)
        
        else:
            raise ValueError(f"Unsupported format: {output_format}")


    def generate_error_interpretation_samples(self, target_count: int = 100) -> List[Dict]:
        """Generate specialized error interpretation samples."""
        samples = []
        
        # Extract error patterns from all YAML files
        error_patterns = []
        for file_name, file_data in self.config_data.items():
            if 'error' in file_name or file_data.get('category') == 'errors':
                data = file_data.get('data', {})
                errors = self._extract_errors_from_structure(data)
                error_patterns.extend(errors)
        
        if not error_patterns:
            logger.warning("No error patterns found")
            return samples
        
        for i in range(min(target_count, len(error_patterns) * 2)):
            try:
                error, category = random.choice(error_patterns)
                
                # Create error-specific prompts
                prompts = [
                    f"Explain the error '{error}' in {category} systems. Include what causes it, how to diagnose it, and step-by-step resolution.",
                    f"A user encounters '{error}'. Provide troubleshooting steps, root causes, and preventive measures.",
                    f"As a system administrator, how would you handle '{error}'? Include immediate actions, investigation steps, and long-term fixes.",
                ]
                
                prompt = random.choice(prompts)
                responses = model_chain.generate_responses(prompt, max_tokens=400)
                
                if responses:
                    best_response = max(responses, key=lambda r: r.get('quality_score', 0))
                    
                    sample = {
                        "type": "error_interpretation",
                        "error": error,
                        "category": category,
                        "prompt": prompt,
                        "response": best_response['response'],
                        "model": best_response.get('model', 'unknown'),
                        "timestamp": datetime.now().isoformat()
                    }
                    samples.append(sample)
                    
            except Exception as e:
                logger.error(f"Error generating error interpretation {i}: {e}")
                continue
        
        return samples

    def _extract_errors_from_structure(self, data: Any) -> List[Tuple[str, str]]:
        """Extract error patterns from data structure."""
        errors = []
        
        def extract_recursive(obj: Any, path: str = ""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    extract_recursive(value, f"{path}/{key}")
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, str) and len(item) > 2:
                        # Determine category from path
                        category = path.split('/')[1] if '/' in path else 'general'
                        errors.append((item.strip(), category))
        
        extract_recursive(data)
        return errors
    
    
    def _create_varied_prompt(self, term: str, category: str, variation: int) -> str:
        """Create varied prompts for diversity."""
        prompts = [
            f"Provide a comprehensive technical explanation of '{term}' in {category}. Include purpose, common issues, monitoring importance, and troubleshooting steps.",
            f"Explain '{term}' from a system administration perspective. Cover what it is, why it matters, how to monitor it, and best practices.",
            f"As an expert in {category}, explain '{term}'. Include real-world examples, common problems, and actionable solutions.",
            f"Define and explain '{term}' for IT professionals. Cover technical details, monitoring strategies, and troubleshooting approaches.",
            f"Describe '{term}' in the context of {category}. Include its function, importance, failure modes, and remediation steps."
        ]
        
        return prompts[variation % len(prompts)]
    
    def _determine_sample_type(self, prompt: str) -> str:
        """Determine sample type from prompt content."""
        prompt_lower = prompt.lower()
        
        if 'error' in prompt_lower or 'troubleshoot' in prompt_lower:
            return 'error_interpretation'
        elif 'conversation' in prompt_lower or 'explain to' in prompt_lower:
            return 'conversational_samples'
        elif 'business' in prompt_lower or 'management' in prompt_lower:
            return 'business_terminology'
        else:
            return 'technical_explanation'
    
    def generate_metrics_dataset(self, target_count: int = None) -> Dict:
        """Generate metrics training dataset."""
        if target_count is None:
            target_count = CONFIG.get('metrics_samples', 10000)
        
        logger.info(f"📊 Generating {target_count} metrics samples")
        
        # Get metrics patterns
        metrics_data = self.config_data.get('metrics_patterns', {}).get('data', {})
        normal_ranges = metrics_data.get('normal_ranges', {})
        anomaly_patterns = metrics_data.get('anomaly_patterns', {})
        
        if not normal_ranges:
            logger.error("No normal_ranges found in metrics_patterns.yaml")
            return {"training_samples": [], "metadata": {}}
        
        samples = []
        anomaly_ratio = CONFIG.get("anomaly_ratio", 0.2)
        current_time = datetime.now() - timedelta(days=30)
        
        for i in range(target_count):
            try:
                # Generate normal or anomaly sample
                if random.random() < anomaly_ratio and anomaly_patterns:
                    sample = self._generate_anomaly_sample(current_time, normal_ranges, anomaly_patterns)
                else:
                    sample = self._generate_normal_sample(current_time, normal_ranges)
                
                samples.append(sample)
                current_time += timedelta(minutes=random.randint(1, 10))
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"📊 Metrics progress: {i + 1}/{target_count}")
                    
            except Exception as e:
                logger.error(f"Error generating metrics sample {i}: {e}")
                continue
        
        # Create dataset
        dataset = {
            "training_samples": samples,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_samples": len(samples),
                "anomaly_ratio": anomaly_ratio,
                "session_id": self.progress.session_id
            }
        }
        
        # Save dataset
        output_file = Path(CONFIG['training_dir']) / 'metrics_dataset.json'
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        self.progress.metrics_completed = len(samples)
        self._save_progress()
        
        logger.info(f"✅ Metrics generation complete: {len(samples)} samples")
        return dataset
    
    def _generate_normal_sample(self, timestamp: datetime, normal_ranges: Dict) -> Dict:
        """Generate normal metrics sample."""
        metrics = {}
        
        for category, metric_ranges in normal_ranges.items():
            if isinstance(metric_ranges, dict):
                for metric_name, range_values in metric_ranges.items():
                    if isinstance(range_values, list) and len(range_values) == 2:
                        min_val, max_val = range_values
                        metrics[metric_name] = random.uniform(min_val, max_val)
        
        return {
            "id": f"normal_{timestamp.strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
            "timestamp": timestamp.isoformat(),
            "metrics": metrics,
            "status": "normal",
            "explanation": "System operating within normal parameters."
        }
    
    def _generate_anomaly_sample(self, timestamp: datetime, normal_ranges: Dict, anomaly_patterns: Dict) -> Dict:
        """Generate anomalous metrics sample."""
        # Start with normal sample
        sample = self._generate_normal_sample(timestamp, normal_ranges)
        
        # Select and apply anomaly pattern
        anomaly_type = random.choice(list(anomaly_patterns.keys()))
        pattern = anomaly_patterns[anomaly_type]
        
        # Apply anomaly metrics
        if isinstance(pattern, dict):
            anomaly_metrics = pattern.get('metrics', {})
            for metric, range_values in anomaly_metrics.items():
                if isinstance(range_values, list) and len(range_values) == 2:
                    min_val, max_val = range_values
                    sample['metrics'][metric] = random.uniform(min_val, max_val)
            
            # Apply correlated effects
            correlated = pattern.get('correlated_effects', {})
            for metric, range_values in correlated.items():
                if isinstance(range_values, list) and len(range_values) == 2:
                    min_val, max_val = range_values
                    sample['metrics'][metric] = random.uniform(min_val, max_val)
        
        sample.update({
            'id': f"anomaly_{anomaly_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
            'status': 'anomaly',
            'anomaly_type': anomaly_type,
            'explanation': pattern.get('description', 'Anomaly detected') if isinstance(pattern, dict) else 'Anomaly detected'
        })
        
        return sample
    
    def _save_language_dataset(self, samples: List[Dict]):
        """Save language dataset with append capability."""
        output_file = Path(CONFIG['training_dir']) / 'language_dataset.json'
        
        # Load existing if present
        existing_samples = []
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    existing_samples = data.get('samples', []) if isinstance(data, dict) else data
            except Exception as e:
                logger.warning(f"Could not load existing samples: {e}")
        
        # Combine samples
        all_samples = existing_samples + samples
        
        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_samples": len(all_samples),
                "new_samples": len(samples),
                "session_id": self.progress.session_id,
                "failed_terms": list(self.failed_terms),
                "generation_stats": self.generation_stats
            },
            "samples": all_samples
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Language dataset saved: {len(samples)} new samples, {len(all_samples)} total")

    def integrate_real_data_sources(self, max_real_samples: int = 200) -> List[Dict]:
        """Integrate real data from Splunk, Jira, Confluence, and Spectrum."""
        try:
            from data_source_integrators import DataSourceOrchestrator
            
            orchestrator = DataSourceOrchestrator()
            real_samples = orchestrator.generate_training_samples_from_data(max_real_samples)
            
            logger.info(f"🎯 Integrated {len(real_samples)} real data samples")
            
            # Convert to standard format
            converted_samples = []
            for sample in real_samples:
                converted = {
                    "type": f"real_data_{sample['type']}",
                    "prompt": sample['prompt'],
                    "context": sample.get('context', {}),
                    "data_driven": True,
                    "quality_score": sample.get('quality_score', 0.8),
                    "session_id": self.progress.session_id,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Generate response using model chain
                responses = model_chain.generate_responses(sample['prompt'], max_tokens=300)
                if responses:
                    best_response = max(responses, key=lambda r: r.get('quality_score', 0))
                    converted['response'] = best_response['response']
                    converted['model'] = best_response.get('model', 'unknown')
                    
                    converted_samples.append(converted)
            
            return converted_samples
            
        except ImportError:
            logger.warning("Data source integrators not available")
            return []
        except Exception as e:
            logger.error(f"Real data integration failed: {e}")
            return []
    
    def generate_complete_dataset(self, language_count: int = None, metrics_count: int = None) -> Tuple[List[Dict], Dict]:
        """Generate both datasets."""
        logger.info("🚀 Starting complete dataset generation")
        
        # Generate language samples
        language_samples = self.generate_language_dataset(language_count)
        
        # Generate metrics samples
        metrics_result = self.generate_metrics_dataset(metrics_count)
        
        # Save final progress
        self._save_progress()
        
        # Show final report
        self.show_final_report()
        
        return language_samples, metrics_result
    
    def show_progress(self):
        """Display current progress."""
        targets, total_target = self._calculate_dynamic_targets()
        existing = self._analyze_existing_dataset()
        
        print(f"\n{'='*60}")
        print("DATASET GENERATION PROGRESS")
        print(f"{'='*60}")
        print(f"Session: {self.progress.session_id}")
        print(f"Started: {self.progress.started_at}")
        print(f"Last Update: {self.progress.last_updated}")
        print(f"\nLanguage Samples:")
        
        for sample_type, target in targets.items():
            current = existing.get(sample_type, 0)
            percent = (current / target * 100) if target > 0 else 100
            status = "✅" if current >= target else "🔄"
            print(f"  {status} {sample_type}: {current}/{target} ({percent:.1f}%)")
        
        print(f"\nMetrics: {self.progress.metrics_completed}")
        print(f"Failed items: {len(self.progress.failed_items)}")
        print(f"Generation stats: {self.generation_stats}")
        print(f"{'='*60}")
    
    def show_final_report(self):
        """Show final generation report."""
        print(f"\n{'='*60}")
        print("GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Language samples: {self.progress.language_completed}")
        print(f"Metrics samples: {self.progress.metrics_completed}")
        print(f"Total errors: {self.generation_stats['errors']}")
        print(f"Skipped terms: {self.generation_stats['skipped_terms']}")
        
        if self.failed_terms:
            print(f"\nFailed terms ({len(self.failed_terms)}):")
            for term in list(self.failed_terms)[:10]:
                print(f"  - {term}")
            if len(self.failed_terms) > 10:
                print(f"  ... and {len(self.failed_terms) - 10} more")
        
        print(f"{'='*60}")
    
    def reset_progress(self):
        """Reset all progress."""
        if self.progress_file.exists():
            self.progress_file.unlink()
        self.progress = self._load_or_create_progress()
        self.failed_terms.clear()
        self.failed_errors.clear()
        self.generation_stats = {
            "samples_generated": 0,
            "api_calls": 0,
            "errors": 0,
            "skipped_terms": 0
        }
        logger.info("✅ Progress reset")

    # Doesn't really work. 
    def retry_failed(self):
        """Clear failed items for retry."""
        self.progress.failed_items.clear()
        self.failed_terms.clear()
        self.failed_errors.clear()
        self._save_progress()
        logger.info("✅ Failed items cleared for retry")

# Backwards compatibility aliases. I rename stuff a lot.
OptimizedDatasetGenerator = DatasetGenerator

# Simple interface functions
def generate_datasets(language_count: int = None, metrics_count: int = None) -> Tuple[List[Dict], Dict]:
    """Generate complete dataset."""
    generator = DatasetGenerator()
    return generator.generate_complete_dataset(language_count, metrics_count)

def show_progress():
    """Show current progress."""
    generator = DatasetGenerator()
    generator.show_progress()

def reset_progress():
    """Reset progress."""
    generator = DatasetGenerator()
    generator.reset_progress()

def retry_failed():
    """Retry failed items."""
    generator = DatasetGenerator()
    generator.retry_failed()