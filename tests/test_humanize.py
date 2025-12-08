import os
# Set a dummy OpenAI API key so that imports do not fail
os.environ.setdefault("OPENAI_API_KEY", "dummy-key")

import sys
# Add repository root to PYTHONPATH so tests can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import humanize

def test_humanize_reason_he_none():
    # When no reason is provided, the function should return a default Hebrew message
    assert humanize.humanize_reason_he(None) == "ללא גורמים חריגים"

def test_where_from_likely_fault_known():
    # Should map Engine to its Hebrew description
    assert humanize.where_from_likely_fault("Engine failure") == "במנוע"

def test_where_from_likely_fault_unknown():
    # Unknown maps to general fault description
    assert humanize.where_from_likely_fault("Unknown glitch") == "תקלה כללית"
    # Completely unrelated should return 'unidentified system'
    assert humanize.where_from_likely_fault("Unrelated fault") == "במערכת לא מזוהה"
