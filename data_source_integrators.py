#!/usr/bin/env python3
"""
Data source integrators for Splunk, Jira, Confluence, and IBM Spectrum Conductor.
Uses python-dotenv for secure token management.
Pure data collection only.
"""

import os
import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import base64

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SplunkIntegrator:
    """Splunk integration for VEMKD logs and system metrics."""
    
    def __init__(self):
        self.base_url = os.getenv('SPLUNK_URL')
        self.token = os.getenv('SPLUNK_TOKEN')
        self.username = os.getenv('SPLUNK_USERNAME')
        self.password = os.getenv('SPLUNK_PASSWORD')
        self.timeout = int(os.getenv('SPLUNK_TIMEOUT', '30'))
        
        if not self.base_url:
            logger.warning("SPLUNK_URL not found in .env file")
            self.enabled = False
        elif not (self.token or (self.username and self.password)):
            logger.warning("SPLUNK_TOKEN or SPLUNK_USERNAME/PASSWORD not found in .env file")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"✅ Splunk integrator initialized: {self.base_url}")
    
    def _get_headers(self):
        """Get authentication headers."""
        if self.token:
            return {
                'Authorization': f'Bearer {self.token}',
                'Content-Type': 'application/json'
            }
        else:
            return {'Content-Type': 'application/json'}
    
    def _get_auth(self):
        """Get authentication for requests."""
        if self.username and self.password:
            return (self.username, self.password)
        return None
    
    def search(self, query: str, earliest_time: str = "-1h", latest_time: str = "now") -> List[Dict]:
        """Execute Splunk search query."""
        if not self.enabled:
            return []
        
        try:
            # Create search job
            search_url = f"{self.base_url}/services/search/jobs"
            search_data = {
                'search': query,
                'earliest_time': earliest_time,
                'latest_time': latest_time,
                'output_mode': 'json'
            }
            
            response = requests.post(
                search_url,
                data=search_data,
                headers=self._get_headers(),
                auth=self._get_auth(),
                timeout=self.timeout,
                verify=False
            )
            
            if response.status_code != 201:
                logger.error(f"Splunk search failed: {response.status_code}")
                return []
            
            # Get search job ID
            search_id = response.json().get('sid')
            if not search_id:
                logger.error("No search ID returned from Splunk")
                return []
            
            # Wait for job completion and get results
            results_url = f"{self.base_url}/services/search/jobs/{search_id}/results"
            results_response = requests.get(
                results_url,
                headers=self._get_headers(),
                auth=self._get_auth(),
                params={'output_mode': 'json'},
                timeout=self.timeout,
                verify=False
            )
            
            if results_response.status_code == 200:
                results = results_response.json()
                return results.get('results', [])
            else:
                logger.error(f"Failed to get Splunk results: {results_response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Splunk search error: {e}")
            return []
    
    def get_vemkd_logs(self, time_range: str = "-1h") -> List[Dict]:
        """Get VEMKD logs from Red Hat Linux systems."""
        query = f'index=linux sourcetype="vemkd" earliest={time_range}'
        return self.search(query, time_range)
    
    def get_system_metrics(self, time_range: str = "-15m") -> List[Dict]:
        """Get system performance metrics."""
        query = f'index=system earliest={time_range} | stats avg(cpu_usage), avg(memory_usage), avg(disk_usage) by host'
        return self.search(query, time_range)

class JiraIntegrator:
    """Jira integration for incident tickets and problem tracking."""
    
    def __init__(self):
        self.base_url = os.getenv('JIRA_URL')
        self.username = os.getenv('JIRA_USERNAME')
        self.token = os.getenv('JIRA_TOKEN')  # API token, not password
        self.timeout = int(os.getenv('JIRA_TIMEOUT', '30'))
        
        if not all([self.base_url, self.username, self.token]):
            logger.warning("JIRA_URL, JIRA_USERNAME, or JIRA_TOKEN not found in .env file")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"✅ Jira integrator initialized: {self.base_url}")
    
    def _get_headers(self):
        """Get authentication headers with base64 encoded credentials."""
        auth_string = f"{self.username}:{self.token}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()
        
        return {
            'Authorization': f'Basic {encoded_auth}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    
    def search_issues(self, jql: str, max_results: int = 100) -> List[Dict]:
        """Search Jira issues using JQL."""
        if not self.enabled:
            return []
        
        try:
            url = f"{self.base_url}/rest/api/2/search"
            params = {
                'jql': jql,
                'maxResults': max_results,
                'fields': 'summary,description,status,priority,created,resolved,assignee'
            }
            
            response = requests.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('issues', [])
            else:
                logger.error(f"Jira search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Jira search error: {e}")
            return []
    
    def get_incident_tickets(self, days_back: int = 7) -> List[Dict]:
        """Get recent incident tickets."""
        jql = f'project = IT AND issuetype = Incident AND created >= -{days_back}d'
        return self.search_issues(jql)
    
    def get_problem_tickets(self, days_back: int = 30) -> List[Dict]:
        """Get recent problem tickets."""
        jql = f'project = IT AND issuetype = Problem AND created >= -{days_back}d'
        return self.search_issues(jql)

class ConfluenceIntegrator:
    """Confluence integration for knowledge base and documentation."""
    
    def __init__(self):
        self.base_url = os.getenv('CONFLUENCE_URL')
        self.username = os.getenv('CONFLUENCE_USERNAME')
        self.token = os.getenv('CONFLUENCE_TOKEN')
        self.timeout = int(os.getenv('CONFLUENCE_TIMEOUT', '30'))
        
        if not all([self.base_url, self.username, self.token]):
            logger.warning("CONFLUENCE_URL, CONFLUENCE_USERNAME, or CONFLUENCE_TOKEN not found in .env file")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"✅ Confluence integrator initialized: {self.base_url}")
    
    def _get_headers(self):
        """Get authentication headers."""
        auth_string = f"{self.username}:{self.token}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()
        
        return {
            'Authorization': f'Basic {encoded_auth}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    
    def search_content(self, query: str, limit: int = 50) -> List[Dict]:
        """Search Confluence content."""
        if not self.enabled:
            return []
        
        try:
            url = f"{self.base_url}/rest/api/content/search"
            params = {
                'cql': f'text ~ "{query}"',
                'limit': limit,
                'expand': 'body.view,space'
            }
            
            response = requests.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])
            else:
                logger.error(f"Confluence search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Confluence search error: {e}")
            return []
    
    def get_troubleshooting_docs(self) -> List[Dict]:
        """Get troubleshooting documentation."""
        return self.search_content("troubleshooting OR error OR problem", limit=100)

class SpectrumIntegrator:
    """IBM Spectrum Conductor integration for workload management."""
    
    def __init__(self):
        self.base_url = os.getenv('SPECTRUM_URL')
        self.username = os.getenv('SPECTRUM_USERNAME')
        self.password = os.getenv('SPECTRUM_PASSWORD')
        self.timeout = int(os.getenv('SPECTRUM_TIMEOUT', '30'))
        
        if not all([self.base_url, self.username, self.password]):
            logger.warning("SPECTRUM_URL, SPECTRUM_USERNAME, or SPECTRUM_PASSWORD not found in .env file")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"✅ Spectrum integrator initialized: {self.base_url}")
    
    def _get_auth(self):
        """Get authentication tuple."""
        return (self.username, self.password)
    
    def get_cluster_info(self) -> Dict:
        """Get cluster information."""
        if not self.enabled:
            return {}
        
        try:
            url = f"{self.base_url}/platform/rest/conductor/v1/clusters"
            response = requests.get(
                url,
                auth=self._get_auth(),
                timeout=self.timeout,
                verify=False
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Spectrum cluster info failed: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Spectrum cluster error: {e}")
            return {}
    
    def get_workloads(self) -> List[Dict]:
        """Get current workloads."""
        if not self.enabled:
            return []
        
        try:
            url = f"{self.base_url}/platform/rest/conductor/v1/workloads"
            response = requests.get(
                url,
                auth=self._get_auth(),
                timeout=self.timeout,
                verify=False
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('workloads', [])
            else:
                logger.error(f"Spectrum workloads failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Spectrum workloads error: {e}")
            return []
    
    def get_resource_groups(self) -> List[Dict]:
        """Get resource group information."""
        if not self.enabled:
            return []
        
        try:
            url = f"{self.base_url}/platform/rest/conductor/v1/resourcegroups"
            response = requests.get(
                url,
                auth=self._get_auth(),
                timeout=self.timeout,
                verify=False
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('resourceGroups', [])
            else:
                logger.error(f"Spectrum resource groups failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Spectrum resource groups error: {e}")
            return []

class DataSourceOrchestrator:
    """Orchestrates all data source integrations for training data generation."""
    
    def __init__(self):
        self.splunk = SplunkIntegrator()
        self.jira = JiraIntegrator()
        self.confluence = ConfluenceIntegrator()
        self.spectrum = SpectrumIntegrator()
        
        # Count enabled integrations
        enabled_count = sum([
            self.splunk.enabled,
            self.jira.enabled,
            self.confluence.enabled,
            self.spectrum.enabled
        ])
        
        logger.info(f"📊 Data source orchestrator initialized with {enabled_count}/4 integrations")
    
    def generate_training_samples_from_data(self, max_samples: int = 200) -> List[Dict]:
        """Generate training samples from real data sources."""
        samples = []
        samples_per_source = max_samples // 4
        
        # Splunk-based samples
        if self.splunk.enabled:
            splunk_samples = self._generate_splunk_samples(samples_per_source)
            samples.extend(splunk_samples)
            logger.info(f"Generated {len(splunk_samples)} Splunk-based samples")
        
        # Jira-based samples
        if self.jira.enabled:
            jira_samples = self._generate_jira_samples(samples_per_source)
            samples.extend(jira_samples)
            logger.info(f"Generated {len(jira_samples)} Jira-based samples")
        
        # Confluence-based samples
        if self.confluence.enabled:
            confluence_samples = self._generate_confluence_samples(samples_per_source)
            samples.extend(confluence_samples)
            logger.info(f"Generated {len(confluence_samples)} Confluence-based samples")
        
        # Spectrum-based samples
        if self.spectrum.enabled:
            spectrum_samples = self._generate_spectrum_samples(samples_per_source)
            samples.extend(spectrum_samples)
            logger.info(f"Generated {len(spectrum_samples)} Spectrum-based samples")
        
        return samples
    
    def _generate_splunk_samples(self, count: int) -> List[Dict]:
        """Generate training samples from Splunk data."""
        samples = []
        
        try:
            # Get VEMKD logs
            vemkd_logs = self.splunk.get_vemkd_logs("-24h")
            
            for log in vemkd_logs[:count//2]:
                sample = {
                    'type': 'vemkd_log_analysis',
                    'prompt': f"Analyze this VEMKD log entry: {log.get('_raw', '')}",
                    'context': {
                        'source': 'splunk_vemkd',
                        'timestamp': log.get('_time'),
                        'host': log.get('host'),
                        'log_level': log.get('log_level')
                    },
                    'quality_score': 0.8
                }
                samples.append(sample)
            
            # Get system metrics
            metrics = self.splunk.get_system_metrics("-1h")
            
            for metric in metrics[:count//2]:
                sample = {
                    'type': 'system_metric_interpretation',
                    'prompt': f"Interpret these system metrics: CPU={metric.get('avg(cpu_usage)', 'N/A')}%, Memory={metric.get('avg(memory_usage)', 'N/A')}%, Disk={metric.get('avg(disk_usage)', 'N/A')}%",
                    'context': {
                        'source': 'splunk_metrics',
                        'host': metric.get('host'),
                        'metric_type': 'system_performance'
                    },
                    'quality_score': 0.9
                }
                samples.append(sample)
                
        except Exception as e:
            logger.error(f"Error generating Splunk samples: {e}")
        
        return samples
    
    def _generate_jira_samples(self, count: int) -> List[Dict]:
        """Generate training samples from Jira tickets."""
        samples = []
        
        try:
            # Get incident tickets
            incidents = self.jira.get_incident_tickets(7)
            
            for ticket in incidents[:count//2]:
                sample = {
                    'type': 'incident_analysis',
                    'prompt': f"Analyze this incident: {ticket.get('fields', {}).get('summary', '')}. Description: {ticket.get('fields', {}).get('description', '')}",
                    'context': {
                        'source': 'jira_incident',
                        'ticket_key': ticket.get('key'),
                        'status': ticket.get('fields', {}).get('status', {}).get('name'),
                        'priority': ticket.get('fields', {}).get('priority', {}).get('name')
                    },
                    'quality_score': 0.85
                }
                samples.append(sample)
            
            # Get problem tickets
            problems = self.jira.get_problem_tickets(30)
            
            for ticket in problems[:count//2]:
                sample = {
                    'type': 'problem_resolution',
                    'prompt': f"Provide resolution steps for this problem: {ticket.get('fields', {}).get('summary', '')}",
                    'context': {
                        'source': 'jira_problem',
                        'ticket_key': ticket.get('key'),
                        'status': ticket.get('fields', {}).get('status', {}).get('name')
                    },
                    'quality_score': 0.9
                }
                samples.append(sample)
                
        except Exception as e:
            logger.error(f"Error generating Jira samples: {e}")
        
        return samples
    
    def _generate_confluence_samples(self, count: int) -> List[Dict]:
        """Generate training samples from Confluence documentation."""
        samples = []
        
        try:
            docs = self.confluence.get_troubleshooting_docs()
            
            for doc in docs[:count]:
                sample = {
                    'type': 'knowledge_base_query',
                    'prompt': f"Based on this documentation, explain: {doc.get('title', '')}",
                    'context': {
                        'source': 'confluence_docs',
                        'space': doc.get('space', {}).get('name'),
                        'doc_id': doc.get('id'),
                        'doc_type': doc.get('type')
                    },
                    'quality_score': 0.75
                }
                samples.append(sample)
                
        except Exception as e:
            logger.error(f"Error generating Confluence samples: {e}")
        
        return samples
    
    def _generate_spectrum_samples(self, count: int) -> List[Dict]:
        """Generate training samples from Spectrum Conductor data."""
        samples = []
        
        try:
            # Get cluster info
            cluster_info = self.spectrum.get_cluster_info()
            if cluster_info:
                sample = {
                    'type': 'spectrum_cluster_analysis',
                    'prompt': f"Analyze this Spectrum cluster status: {json.dumps(cluster_info, indent=2)}",
                    'context': {
                        'source': 'spectrum_cluster',
                        'cluster_type': 'conductor'
                    },
                    'quality_score': 0.8
                }
                samples.append(sample)
            
            # Get workloads
            workloads = self.spectrum.get_workloads()
            for workload in workloads[:count//2]:
                sample = {
                    'type': 'spectrum_workload_analysis',
                    'prompt': f"Analyze this Spectrum workload: {json.dumps(workload, indent=2)}",
                    'context': {
                        'source': 'spectrum_workload',
                        'workload_id': workload.get('id'),
                        'workload_state': workload.get('state')
                    },
                    'quality_score': 0.85
                }
                samples.append(sample)
            
            # Get resource groups
            resource_groups = self.spectrum.get_resource_groups()
            for rg in resource_groups[:count//2]:
                sample = {
                    'type': 'spectrum_resource_analysis',
                    'prompt': f"Analyze this Spectrum resource group: {json.dumps(rg, indent=2)}",
                    'context': {
                        'source': 'spectrum_resources',
                        'rg_name': rg.get('name'),
                        'rg_type': rg.get('type')
                    },
                    'quality_score': 0.8
                }
                samples.append(sample)
                
        except Exception as e:
            logger.error(f"Error generating Spectrum samples: {e}")
        
        return samples
    
    def test_all_connections(self) -> Dict[str, bool]:
        """Test all data source connections."""
        results = {}
        
        # Test Splunk
        if self.splunk.enabled:
            try:
                test_results = self.splunk.search("| head 1", "-1m")
                results['splunk'] = len(test_results) >= 0  # Any result including empty is success
            except:
                results['splunk'] = False
        else:
            results['splunk'] = False
        
        # Test Jira
        if self.jira.enabled:
            try:
                test_results = self.jira.search_issues("project IS NOT EMPTY", 1)
                results['jira'] = len(test_results) >= 0
            except:
                results['jira'] = False
        else:
            results['jira'] = False
        
        # Test Confluence
        if self.confluence.enabled:
            try:
                test_results = self.confluence.search_content("test", 1)
                results['confluence'] = len(test_results) >= 0
            except:
                results['confluence'] = False
        else:
            results['confluence'] = False
        
        # Test Spectrum
        if self.spectrum.enabled:
            try:
                cluster_info = self.spectrum.get_cluster_info()
                results['spectrum'] = isinstance(cluster_info, dict)
            except:
                results['spectrum'] = False
        else:
            results['spectrum'] = False
        
        return results

def create_env_template():
    """Create a template .env file with all required variables."""
    env_template = """# Data Source Integration Configuration
# Copy this file to .env and fill in your actual values

# Splunk Configuration
SPLUNK_URL=https://your-splunk-instance:8089
SPLUNK_TOKEN=your_splunk_token_here
# OR use username/password instead of token
SPLUNK_USERNAME=your_splunk_username
SPLUNK_PASSWORD=your_splunk_password
SPLUNK_TIMEOUT=30

# Jira Configuration
JIRA_URL=https://your-company.atlassian.net
JIRA_USERNAME=your_email@company.com
JIRA_TOKEN=your_jira_api_token
JIRA_TIMEOUT=30

# Confluence Configuration
CONFLUENCE_URL=https://your-company.atlassian.net/wiki
CONFLUENCE_USERNAME=your_email@company.com
CONFLUENCE_TOKEN=your_confluence_api_token
CONFLUENCE_TIMEOUT=30

# IBM Spectrum Conductor Configuration
SPECTRUM_URL=https://your-spectrum-cluster:8443
SPECTRUM_USERNAME=your_spectrum_username
SPECTRUM_PASSWORD=your_spectrum_password
SPECTRUM_TIMEOUT=30
"""
    
    with open('.env.template', 'w') as f:
        f.write(env_template)
    
    print("✅ Created .env.template file")
    print("📝 Copy to .env and fill in your actual credentials")

def interactive_menu():
    """Interactive menu for standalone testing."""
    print("""
🔧 DATA SOURCE INTEGRATOR - STANDALONE MODE
============================================

1. Test all connections
2. Test specific connection
3. Collect sample data
4. Generate training samples
5. View connection status
6. Create .env template
7. Exit

Choose an option (1-7): """, end="")

def test_specific_connection(orchestrator):
    """Test a specific data source connection."""
    print("\nAvailable data sources:")
    print("1. Splunk")
    print("2. Jira") 
    print("3. Confluence")
    print("4. Spectrum")
    
    choice = input("Choose data source (1-4): ").strip()
    
    if choice == "1":
        if orchestrator.splunk.enabled:
            print("🧪 Testing Splunk connection...")
            try:
                results = orchestrator.splunk.search("| head 5", "-1h")
                print(f"✅ Splunk: Retrieved {len(results)} sample records")
                if results:
                    print("Sample data:")
                    for i, result in enumerate(results[:2], 1):
                        print(f"  {i}. {result}")
            except Exception as e:
                print(f"❌ Splunk test failed: {e}")
        else:
            print("❌ Splunk not configured")
    
    elif choice == "2":
        if orchestrator.jira.enabled:
            print("🧪 Testing Jira connection...")
            try:
                issues = orchestrator.jira.search_issues("ORDER BY created DESC", 3)
                print(f"✅ Jira: Retrieved {len(issues)} recent issues")
                for issue in issues:
                    print(f"  • {issue.get('key')}: {issue.get('fields', {}).get('summary', 'No summary')}")
            except Exception as e:
                print(f"❌ Jira test failed: {e}")
        else:
            print("❌ Jira not configured")
    
    elif choice == "3":
        if orchestrator.confluence.enabled:
            print("🧪 Testing Confluence connection...")
            try:
                docs = orchestrator.confluence.search_content("linux", 3)
                print(f"✅ Confluence: Retrieved {len(docs)} documents")
                for doc in docs:
                    print(f"  • {doc.get('title', 'No title')} ({doc.get('type', 'Unknown type')})")
            except Exception as e:
                print(f"❌ Confluence test failed: {e}")
        else:
            print("❌ Confluence not configured")
    
    elif choice == "4":
        if orchestrator.spectrum.enabled:
            print("🧪 Testing Spectrum connection...")
            try:
                cluster_info = orchestrator.spectrum.get_cluster_info()
                workloads = orchestrator.spectrum.get_workloads()
                print(f"✅ Spectrum: Cluster info available, {len(workloads)} workloads")
                if cluster_info:
                    print(f"  Cluster status: {cluster_info}")
            except Exception as e:
                print(f"❌ Spectrum test failed: {e}")
        else:
            print("❌ Spectrum not configured")
    else:
        print("Invalid choice")

def collect_sample_data(orchestrator):
    """Collect and display sample data from all sources."""
    print("\n📊 COLLECTING SAMPLE DATA FROM ALL SOURCES")
    print("=" * 50)
    
    # Splunk samples
    if orchestrator.splunk.enabled:
        print("\n🔍 Splunk Data:")
        try:
            vemkd_logs = orchestrator.splunk.get_vemkd_logs("-2h")
            metrics = orchestrator.splunk.get_system_metrics("-1h")
            print(f"  • VEMKD logs: {len(vemkd_logs)} entries")
            print(f"  • System metrics: {len(metrics)} entries")
            
            if vemkd_logs:
                print(f"  Sample VEMKD log: {vemkd_logs[0].get('_raw', 'No data')[:100]}...")
            if metrics:
                print(f"  Sample metric: {metrics[0]}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Jira samples  
    if orchestrator.jira.enabled:
        print("\n🎫 Jira Data:")
        try:
            incidents = orchestrator.jira.get_incident_tickets(7)
            problems = orchestrator.jira.get_problem_tickets(30)
            print(f"  • Recent incidents: {len(incidents)}")
            print(f"  • Recent problems: {len(problems)}")
            
            if incidents:
                incident = incidents[0]
                print(f"  Sample incident: {incident.get('key')} - {incident.get('fields', {}).get('summary', 'No summary')}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Confluence samples
    if orchestrator.confluence.enabled:
        print("\n📚 Confluence Data:")
        try:
            docs = orchestrator.confluence.get_troubleshooting_docs()
            print(f"  • Troubleshooting docs: {len(docs)}")
            
            if docs:
                doc = docs[0]
                print(f"  Sample doc: {doc.get('title', 'No title')} in {doc.get('space', {}).get('name', 'Unknown space')}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Spectrum samples
    if orchestrator.spectrum.enabled:
        print("\n⚡ Spectrum Data:")
        try:
            cluster_info = orchestrator.spectrum.get_cluster_info()
            workloads = orchestrator.spectrum.get_workloads()
            resource_groups = orchestrator.spectrum.get_resource_groups()
            
            print(f"  • Cluster info: {'Available' if cluster_info else 'Not available'}")
            print(f"  • Active workloads: {len(workloads)}")
            print(f"  • Resource groups: {len(resource_groups)}")
            
            if workloads:
                workload = workloads[0]
                print(f"  Sample workload: {workload.get('id', 'No ID')} - {workload.get('state', 'Unknown state')}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

def view_connection_status(orchestrator):
    """Display detailed connection status."""
    print("\n🔌 CONNECTION STATUS DETAILS")
    print("=" * 50)
    
    # Splunk status
    print(f"\n🔍 Splunk:")
    print(f"  URL: {orchestrator.splunk.base_url or 'Not configured'}")
    print(f"  Token auth: {'Yes' if orchestrator.splunk.token else 'No'}")
    print(f"  User auth: {'Yes' if orchestrator.splunk.username else 'No'}")
    print(f"  Enabled: {'✅' if orchestrator.splunk.enabled else '❌'}")
    
    # Jira status
    print(f"\n🎫 Jira:")
    print(f"  URL: {orchestrator.jira.base_url or 'Not configured'}")
    print(f"  Username: {orchestrator.jira.username or 'Not configured'}")
    print(f"  Token: {'Yes' if orchestrator.jira.token else 'No'}")
    print(f"  Enabled: {'✅' if orchestrator.jira.enabled else '❌'}")
    
    # Confluence status
    print(f"\n📚 Confluence:")
    print(f"  URL: {orchestrator.confluence.base_url or 'Not configured'}")
    print(f"  Username: {orchestrator.confluence.username or 'Not configured'}")
    print(f"  Token: {'Yes' if orchestrator.confluence.token else 'No'}")
    print(f"  Enabled: {'✅' if orchestrator.confluence.enabled else '❌'}")
    
    # Spectrum status
    print(f"\n⚡ Spectrum:")
    print(f"  URL: {orchestrator.spectrum.base_url or 'Not configured'}")
    print(f"  Username: {orchestrator.spectrum.username or 'Not configured'}")
    print(f"  Password: {'Yes' if orchestrator.spectrum.password else 'No'}")
    print(f"  Enabled: {'✅' if orchestrator.spectrum.enabled else '❌'}")

if __name__ == "__main__":
    import sys
    
    # Check for .env file
    if not Path('.env').exists():
        print("❌ No .env file found.")
        create_env_template()
        print("📝 Created .env.template for you.")
        print("Please copy .env.template to .env and fill in your credentials.")
        
        create_env = input("\nCreate empty .env file now? (y/n): ").strip().lower()
        if create_env == 'y':
            Path('.env').touch()
            print("✅ Created empty .env file. Please edit it with your credentials.")
        
        print("\n💡 After configuring .env, run this script again to test connections.")
        exit(1)
    
    # Initialize orchestrator
    try:
        orchestrator = DataSourceOrchestrator()
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {e}")
        exit(1)
    
    # Check if running as standalone script
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            print("🧪 Testing all connections...")
            results = orchestrator.test_all_connections()
            for source, success in results.items():
                status = "✅" if success else "❌"
                print(f"  {status} {source}")
        
        elif command == "collect":
            collect_sample_data(orchestrator)
        
        elif command == "generate":
            sample_count = int(sys.argv[2]) if len(sys.argv) > 2 else 20
            print(f"📊 Generating {sample_count} training samples...")
            samples = orchestrator.generate_training_samples_from_data(sample_count)
            print(f"✅ Generated {len(samples)} samples")
            
            # Save to file
            output_file = f"sample_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(samples, f, indent=2)
            print(f"💾 Saved to {output_file}")
        
        elif command == "status":
            view_connection_status(orchestrator)
        
        else:
            print("Usage:")
            print("  python data_source_integrators.py test      # Test all connections")
            print("  python data_source_integrators.py collect   # Collect sample data") 
            print("  python data_source_integrators.py generate [count]  # Generate training samples")
            print("  python data_source_integrators.py status    # View connection status")
            print("  python data_source_integrators.py           # Interactive mode")
    
    else:
        # Interactive mode
        while True:
            try:
                interactive_menu()
                choice = input().strip()
                
                if choice == "1":
                    print("\n🧪 Testing all connections...")
                    results = orchestrator.test_all_connections()
                    for source, success in results.items():
                        status = "✅" if success else "❌"
                        print(f"  {status} {source}")
                
                elif choice == "2":
                    test_specific_connection(orchestrator)
                
                elif choice == "3":
                    collect_sample_data(orchestrator)
                
                elif choice == "4":
                    sample_count = input("Number of training samples to generate (default 20): ").strip()
                    sample_count = int(sample_count) if sample_count else 20
                    
                    print(f"\n📊 Generating {sample_count} training samples...")
                    samples = orchestrator.generate_training_samples_from_data(sample_count)
                    print(f"✅ Generated {len(samples)} samples")
                    
                    save_file = input("Save to file? (y/n): ").strip().lower()
                    if save_file == 'y':
                        output_file = f"sample_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(output_file, 'w') as f:
                            json.dump(samples, f, indent=2)
                        print(f"💾 Saved to {output_file}")
                
                elif choice == "5":
                    view_connection_status(orchestrator)
                
                elif choice == "6":
                    create_env_template()
                
                elif choice == "7":
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("Invalid choice. Please enter 1-7.")
                
                input("\nPress Enter to continue...")
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("Press Enter to continue...")