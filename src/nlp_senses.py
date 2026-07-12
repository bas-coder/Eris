"""
Platform Architecture: NLP Fundamentals Senses (FinBERT)
Workspace: Eris
Environment: Conda fx39 (Python 3.9 win-64)
Dependencies: transformers, torch (CPU-Optimized)
"""

import os
import torch
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class FundamentalsEngine:
    """
    Parses macroeconomic headlines into quantifiable sentiment matrices.
    Hard-coded to execute pure CPU inference based on fx39 system telemetry.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        print(f"[NLP SYSTEM] Initializing FinBERT Neural Senses ({model_name})...")
        
        # Explicitly lock compute to the CPU to match the 2.8.0+cpu installation
        self.device = torch.device("cpu")
        
        try:
            # First execution will download the model to your local HuggingFace cache
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            
            # Set model to evaluation mode (shuts off dropout layers for deterministic execution)
            self.model.eval() 
            print("[NLP SYSTEM] Senses Online. Awaiting market intelligence.")
        except Exception as e:
            print(f"[FATAL NLP ERROR] Failed to load model weights. Ensure internet connection for first-time caching. {e}")
            raise

    def analyze_headline(self, text: str) -> float:
        """
        Ingests a headline, processes it through FinBERT, and outputs
        a ruthless mathematical sentiment score between -1.0 (Bearish) and 1.0 (Bullish).
        """
        if not text or text.isspace():
            return 0.0

        # Tokenize and push to designated CPU compute device
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        # Context manager disables gradient tracking. 
        # This reduces memory consumption and drastically speeds up CPU inference.
        with torch.no_grad(): 
            outputs = self.model(**inputs)
        
        # Extract logits and apply softmax to get raw probabilities
        # FinBERT output mapping: [Positive, Negative, Neutral]
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Map tensors back to standard Python floats for the Gym Environment matrix
        positive_prob = float(probs[0][0].item())
        negative_prob = float(probs[0][1].item())
        
        # Calculate final delta score. A completely neutral headline yields ~0.0.
        sentiment_score = positive_prob - negative_prob
        
        return round(sentiment_score, 4)

# Quick diagnostic test block. Runs only if this specific file is executed directly.
if __name__ == "__main__":
    test_engine = FundamentalsEngine()
    test_headline = "US Non-Farm Payrolls massively exceed expectations, signaling aggressive rate hikes."
    score = test_engine.analyze_headline(test_headline)
    print(f"Test Headline: '{test_headline}'")
    print(f"Calculated Sentiment Delta: {score}") # Should yield a highly positive score