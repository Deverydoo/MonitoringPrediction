"""
Server Profile Detection

Extracts server profile from server name patterns.
"""

import streamlit as st


@st.cache_data(ttl=300)  # Cache for 5 minutes - profiles rarely change
def get_server_profile(server_name: str) -> str:
    """
    Extract server profile from server name.

    Args:
        server_name: Server hostname (e.g., 'ppml0001', 'ppdb003')

    Returns:
        Profile name (e.g., 'ML Compute', 'Database')
    """
    if server_name.startswith('ppml'): return 'ML Compute'
    if server_name.startswith('ppdb'): return 'Database'
    if server_name.startswith('ppweb'): return 'Web API'
    if server_name.startswith('ppcon'): return 'Conductor'
    if server_name.startswith('ppetl'): return 'ETL/Ingest'
    if server_name.startswith('pprisk'): return 'Risk Analytics'
    return 'Generic'
