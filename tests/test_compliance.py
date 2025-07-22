"""
Test suite for AAIRE Compliance Engine
"""

import pytest
import asyncio
from src.compliance_engine import ComplianceEngine, ComplianceResult

@pytest.fixture
def compliance_engine():
    return ComplianceEngine()

@pytest.mark.asyncio
async def test_tax_advice_filtering(compliance_engine):
    """Test that tax advice queries are properly filtered"""
    
    queries = [
        "How can I minimize my tax liability?",
        "What tax planning strategies should I use?",
        "Give me tax advice for insurance companies"
    ]
    
    for query in queries:
        result = await compliance_engine.check_query(query)
        assert result.blocked == True
        assert result.rule_name == "tax_advice"
        assert "tax advice" in result.response.lower()

@pytest.mark.asyncio
async def test_legal_advice_filtering(compliance_engine):
    """Test that legal advice queries are properly filtered"""
    
    queries = [
        "Should I sue this company?",
        "Give me legal advice about this contract",
        "What legal interpretation should I use?"
    ]
    
    for query in queries:
        result = await compliance_engine.check_query(query)
        assert result.blocked == True
        assert result.rule_name == "legal_advice"
        assert "legal" in result.response.lower()

@pytest.mark.asyncio
async def test_allowed_queries(compliance_engine):
    """Test that legitimate accounting queries are not filtered"""
    
    queries = [
        "How should insurance reserves be calculated under IFRS 17?",
        "What are the measurement models in IFRS 17?",
        "Explain the contractual service margin calculation"
    ]
    
    for query in queries:
        result = await compliance_engine.check_query(query)
        assert result.blocked == False
        assert result.rule_name is None

@pytest.mark.asyncio
async def test_professional_judgment_disclaimer(compliance_engine):
    """Test professional judgment disclaimer detection"""
    
    queries = [
        "What should we do in this situation?",
        "Which method is better for our company?",
        "What would you recommend?"
    ]
    
    for query in queries:
        response = "Based on IFRS 17, you can use the PAA approach."
        result = await compliance_engine.add_professional_judgment_disclaimer(response, query)
        assert "Professional Judgment Required" in result

def test_rule_management(compliance_engine):
    """Test adding and removing compliance rules"""
    
    # Add new rule
    compliance_engine.add_rule(
        name="test_rule",
        patterns=["test pattern"],
        response_template="Test response"
    )
    
    assert "test_rule" in compliance_engine.rules
    
    # Remove rule
    compliance_engine.remove_rule("test_rule")
    assert "test_rule" not in compliance_engine.rules

@pytest.mark.asyncio
async def test_response_validation(compliance_engine):
    """Test response validation for compliance issues"""
    
    # Response with advice language
    response_with_advice = "You should definitely use this approach."
    query = "What should I do?"
    
    validation = await compliance_engine.validate_response(response_with_advice, query)
    assert validation["compliant"] == False
    assert any(issue["type"] == "advice_language" for issue in validation["issues"])
    
    # Response without citations
    response_without_citations = "Insurance reserves should be measured at fair value."
    validation = await compliance_engine.validate_response(response_without_citations, query)
    assert any(issue["type"] == "missing_citations" for issue in validation["issues"])

if __name__ == "__main__":
    pytest.main([__file__])