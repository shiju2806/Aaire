"""
Compliance Configuration - MVP-FR-017 through MVP-FR-020
Following SRS v2.0 specifications
"""

import re
from typing import Dict, List, Pattern
from enum import Enum

class LogLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

class ComplianceRule:
    def __init__(self, name: str, patterns: List[str], response_template: str, log_level: LogLevel):
        self.name = name
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        self.response_template = response_template
        self.log_level = log_level
    
    def matches(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self.patterns)

# MVP Compliance Rules - As specified in SRS
COMPLIANCE_RULES = {
    "tax_advice": ComplianceRule(
        name="tax_advice",
        patterns=[
            r"tax\s+advice",
            r"tax\s+planning", 
            r"minimize\s+tax",
            r"tax\s+strategy",
            r"tax\s+optimization"
        ],
        response_template="""
I can explain the accounting treatment of tax-related items according to {standard}, 
but I cannot provide tax advice. Please consult a tax professional for tax planning 
and tax strategy questions.

**Disclaimer**: This response provides accounting guidance only and should not be 
construed as tax advice.
        """.strip(),
        log_level=LogLevel.WARNING
    ),
    
    "legal_advice": ComplianceRule(
        name="legal_advice", 
        patterns=[
            r"legal\s+advice",
            r"sue|lawsuit",
            r"legal\s+interpretation",
            r"litigation",
            r"legal\s+opinion"
        ],
        response_template="""
I can explain accounting standards and their requirements, but I cannot provide 
legal interpretations or legal advice. Please consult qualified legal counsel 
for matters involving legal interpretation.

**Disclaimer**: This response provides accounting guidance only and should not be 
construed as legal advice.
        """.strip(),
        log_level=LogLevel.WARNING
    ),
    
    "investment_advice": ComplianceRule(
        name="investment_advice",
        patterns=[
            r"investment\s+advice",
            r"buy|sell|trade",
            r"investment\s+recommendation",
            r"portfolio\s+advice"
        ],
        response_template="""
I provide accounting and actuarial guidance only. I cannot provide investment 
advice or recommendations. Please consult a qualified investment advisor.

**Disclaimer**: This response provides accounting guidance only and should not be 
construed as investment advice.
        """.strip(),
        log_level=LogLevel.WARNING
    ),
    
    "actuarial_certification": ComplianceRule(
        name="actuarial_certification",
        patterns=[
            r"actuarial\s+opinion",
            r"certify|certification",
            r"actuarial\s+statement"
        ],
        response_template="""
I can explain actuarial concepts and methodologies, but I cannot provide 
actuarial opinions or certifications. Such work must be performed by 
qualified actuaries following applicable standards of practice.

**Disclaimer**: This response is for educational purposes only and does not 
constitute professional actuarial advice.
        """.strip(),
        log_level=LogLevel.WARNING
    )
}

# Professional Judgment Disclaimers - MVP-FR-018
PROFESSIONAL_JUDGMENT_TRIGGERS = [
    r"should\s+we",
    r"what\s+would\s+you\s+do",
    r"your\s+recommendation",
    r"best\s+practice",
    r"which\s+method\s+is\s+better"
]

PROFESSIONAL_JUDGMENT_DISCLAIMER = """
**Professional Judgment Required**: The application of accounting standards often 
requires professional judgment based on specific facts and circumstances. This 
response provides general guidance that should be evaluated by qualified 
accounting professionals in the context of your specific situation.
"""

# Audit Trail Configuration - MVP-FR-024
AUDIT_EVENTS = {
    "query_submitted": {
        "log_level": LogLevel.INFO,
        "retention_days": 2555  # 7 years as specified
    },
    "compliance_triggered": {
        "log_level": LogLevel.WARNING, 
        "retention_days": 2555
    },
    "document_uploaded": {
        "log_level": LogLevel.INFO,
        "retention_days": 2555
    },
    "user_login": {
        "log_level": LogLevel.INFO,
        "retention_days": 90  # Hot storage
    }
}