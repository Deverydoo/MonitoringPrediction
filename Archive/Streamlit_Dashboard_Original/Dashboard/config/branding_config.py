"""
Customer Branding Configuration

This module allows easy customization of dashboard branding for different customers.
Change ACTIVE_BRANDING to switch between customer themes.

Created: October 29, 2025
"""

# =============================================================================
# BRANDING PROFILES
# =============================================================================

BRANDING_PROFILES = {
    'nordiq': {
        'name': 'ArgusAI',
        'primary_color': '#0EA5E9',  # NordIQ Blue
        'secondary_color': '#3B82F6',
        'header_text': '🧭 ArgusAI',
        'logo_emoji': '🧭',
        'tagline': 'Predictive System Monitoring',
        'accent_border': '#0EA5E9'
    },

    'wells_fargo': {
        'name': 'Wells Fargo',
        'primary_color': '#D71E28',  # Wells Fargo Official Red
        'secondary_color': '#B71C1C',  # Darker red for hover states
        'header_text': '🏛️ Wells Fargo',
        'logo_emoji': '🏛️',
        'tagline': 'Established 1852',
        'accent_border': '#D71E28'
    },

    'generic': {
        'name': 'Enterprise',
        'primary_color': '#4B5563',  # Professional gray
        'secondary_color': '#374151',
        'header_text': '📊 Enterprise Dashboard',
        'logo_emoji': '📊',
        'tagline': 'Predictive Infrastructure Monitoring',
        'accent_border': '#4B5563'
    }
}

# =============================================================================
# ACTIVE BRANDING (Change this to switch customer themes)
# =============================================================================

# Options: 'nordiq', 'wells_fargo', 'generic'
ACTIVE_BRANDING = 'wells_fargo'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_active_branding():
    """
    Get the active branding configuration.

    Returns:
        dict: Branding configuration for active customer
    """
    return BRANDING_PROFILES.get(ACTIVE_BRANDING, BRANDING_PROFILES['nordiq'])


def get_custom_css():
    """
    Generate custom CSS for active branding.

    Returns:
        str: CSS styles as string
    """
    branding = get_active_branding()

    css = f"""
    <style>
        /* {branding['name']} Branded Header Bar */
        header[data-testid="stHeader"] {{
            background-color: {branding['primary_color']} !important;
            border-bottom: 3px solid {branding['secondary_color']} !important;
        }}

        /* Branding for header text */
        header[data-testid="stHeader"] * {{
            color: white !important;
        }}

        /* Style the app header toolbar */
        .stApp > header {{
            background-color: {branding['primary_color']} !important;
        }}

        /* Sidebar accent border */
        section[data-testid="stSidebar"] {{
            border-right: 3px solid {branding['accent_border']} !important;
        }}

        /* Brand color for metrics and important UI elements */
        div[data-testid="stMetricValue"] {{
            color: {branding['primary_color']} !important;
        }}

        /* Brand color for buttons */
        button[kind="primary"] {{
            background-color: {branding['primary_color']} !important;
            border-color: {branding['primary_color']} !important;
        }}

        button[kind="primary"]:hover {{
            background-color: {branding['secondary_color']} !important;
            border-color: {branding['secondary_color']} !important;
        }}

        /* Brand color for links */
        a {{
            color: {branding['primary_color']} !important;
        }}

        /* Add customer logo/name in header */
        header[data-testid="stHeader"]::before {{
            content: "{branding['header_text']} | ";
            font-weight: bold;
            font-size: 1.2em;
            padding-right: 10px;
        }}

        /* Optional: Style tab buttons with brand color */
        button[data-baseweb="tab"] {{
            border-bottom: 2px solid transparent;
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            border-bottom: 2px solid {branding['primary_color']} !important;
            color: {branding['primary_color']} !important;
        }}
    </style>
    """

    return css


def get_header_title():
    """
    Get the formatted header title with logo.

    Returns:
        str: Header title with logo emoji
    """
    branding = get_active_branding()
    return f"{branding['logo_emoji']} {branding['name']} - Predictive Monitoring"


def get_about_text():
    """
    Get the about text for the menu.

    Returns:
        str: About text with customer branding
    """
    branding = get_active_branding()
    return f"{branding['name']} | {branding['tagline']} | Predictive Infrastructure Monitoring powered by Temporal Fusion Transformer"


# =============================================================================
# COLOR PALETTE REFERENCE
# =============================================================================

# Common enterprise colors for quick reference:
ENTERPRISE_COLORS = {
    # Financial Institutions
    'wells_fargo_red': '#D71E28',
    'bank_of_america_red': '#E31837',
    'chase_blue': '#117ACA',
    'citi_blue': '#056DAE',
    'capital_one_red': '#DB0011',

    # Tech Companies
    'microsoft_blue': '#00A4EF',
    'google_blue': '#4285F4',
    'amazon_orange': '#FF9900',
    'ibm_blue': '#0F62FE',
    'oracle_red': '#F80000',

    # Healthcare
    'uhc_blue': '#002677',
    'anthem_blue': '#003DA5',
    'humana_green': '#00A758',

    # Generic Professional
    'corporate_blue': '#1E3A8A',
    'corporate_gray': '#4B5563',
    'corporate_green': '#059669',
    'corporate_red': '#DC2626'
}

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

"""
To change branding:

1. Edit ACTIVE_BRANDING at top of this file
2. Restart dashboard: daemon.bat restart dashboard
3. That's it!

To add new customer:

BRANDING_PROFILES['acme_corp'] = {
    'name': 'ACME Corporation',
    'primary_color': '#FF6B35',
    'secondary_color': '#E55527',
    'header_text': '🚀 ACME Corp',
    'logo_emoji': '🚀',
    'tagline': 'Innovation Delivered',
    'accent_border': '#FF6B35'
}

Then set: ACTIVE_BRANDING = 'acme_corp'
"""
