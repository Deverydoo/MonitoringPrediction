"""
Explainable AI (XAI) Components for TFT Monitoring System

Provides transparency and interpretability for TFT model predictions through:
1. SHAP values - Feature importance analysis
2. Attention weights - Temporal focus visualization
3. Counterfactual scenarios - What-if analysis

No LLMs required - pure statistical explanations!
"""

from .shap_explainer import TFTShapExplainer
from .attention_visualizer import AttentionVisualizer
from .counterfactual_generator import CounterfactualGenerator

__all__ = [
    'TFTShapExplainer',
    'AttentionVisualizer',
    'CounterfactualGenerator'
]

__version__ = '1.0.0'
