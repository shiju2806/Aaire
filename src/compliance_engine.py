"""
Compliance Engine for AAIRE - MVP-FR-017 through MVP-FR-020
Implements rule-based content filtering and professional judgment detection
"""

import re
import structlog
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from config.compliance import (
    COMPLIANCE_RULES,
    PROFESSIONAL_JUDGMENT_TRIGGERS,
    PROFESSIONAL_JUDGMENT_DISCLAIMER,
    LogLevel
)

logger = structlog.get_logger()

@dataclass
class ComplianceResult:
    blocked: bool
    rule_name: Optional[str]
    response: str
    log_level: LogLevel
    triggered_patterns: List[str]

class ComplianceEngine:
    def __init__(self):
        """Initialize compliance engine with rules from configuration"""
        self.rules = COMPLIANCE_RULES
        self.professional_judgment_triggers = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in PROFESSIONAL_JUDGMENT_TRIGGERS
        ]
        
        logger.info("Compliance engine initialized", 
                   rule_count=len(self.rules),
                   professional_judgment_triggers=len(self.professional_judgment_triggers))
    
    async def check_query(self, query: str) -> ComplianceResult:
        """
        Check query against compliance rules - MVP-FR-017
        Returns blocking result if compliance rules are triggered
        """
        
        # Check each compliance rule
        for rule_name, rule in self.rules.items():
            if rule.matches(query):
                triggered_patterns = [
                    pattern.pattern for pattern in rule.patterns 
                    if pattern.search(query)
                ]
                
                # Log the compliance trigger
                await self._log_compliance_event(
                    rule_name=rule_name,
                    query=query,
                    triggered_patterns=triggered_patterns,
                    log_level=rule.log_level
                )
                
                return ComplianceResult(
                    blocked=True,
                    rule_name=rule_name,
                    response=rule.response_template.format(standard="applicable standards"),
                    log_level=rule.log_level,
                    triggered_patterns=triggered_patterns
                )
        
        # If no blocking rules triggered, check for professional judgment disclaimers
        professional_judgment_needed = any(
            trigger.search(query) for trigger in self.professional_judgment_triggers
        )
        
        return ComplianceResult(
            blocked=False,
            rule_name=None,
            response="",
            log_level=LogLevel.INFO,
            triggered_patterns=[],
        )
    
    async def add_professional_judgment_disclaimer(self, response: str, query: str) -> str:
        """
        Add professional judgment disclaimer if needed - MVP-FR-018
        """
        needs_disclaimer = any(
            trigger.search(query) for trigger in self.professional_judgment_triggers
        )
        
        if needs_disclaimer:
            return f"{response}\n\n{PROFESSIONAL_JUDGMENT_DISCLAIMER}"
        
        return response
    
    async def validate_response(self, response: str, query: str) -> Dict[str, Any]:
        """
        Validate generated response for compliance issues
        """
        issues = []
        
        # Check for potential advice language in response
        advice_patterns = [
            r"you should",
            r"I recommend",
            r"I advise",
            r"my recommendation",
            r"the best approach is"
        ]
        
        for pattern in advice_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                issues.append({
                    "type": "advice_language",
                    "pattern": pattern,
                    "severity": "warning",
                    "message": "Response contains advice-like language"
                })
        
        # Check for unsupported claims (responses without citations)
        if "[" not in response or "]" not in response:
            issues.append({
                "type": "missing_citations",
                "severity": "warning", 
                "message": "Response lacks proper citations"
            })
        
        # Check for regulatory disclaimer compliance
        regulatory_keywords = ["must", "required", "shall", "mandatory"]
        has_regulatory_language = any(
            keyword in response.lower() for keyword in regulatory_keywords
        )
        
        if has_regulatory_language and "professional judgment" not in response.lower():
            issues.append({
                "type": "missing_disclaimer",
                "severity": "warning",
                "message": "Regulatory language without professional judgment disclaimer"
            })
        
        return {
            "compliant": len(issues) == 0,
            "issues": issues,
            "requires_disclaimer": any(
                trigger.search(query) for trigger in self.professional_judgment_triggers
            )
        }
    
    async def _log_compliance_event(
        self, 
        rule_name: str, 
        query: str, 
        triggered_patterns: List[str],
        log_level: LogLevel
    ):
        """Log compliance rule triggers for audit purposes - MVP-FR-019"""
        
        event_data = {
            "compliance_event_type": "compliance_triggered",
            "rule_name": rule_name,
            "query": query[:200],  # Truncate for logs
            "triggered_patterns": triggered_patterns,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if log_level == LogLevel.WARNING:
            logger.warning("Compliance rule triggered", **event_data)
        elif log_level == LogLevel.ERROR:
            logger.error("Compliance rule triggered", **event_data)
        else:
            logger.info("Compliance rule triggered", **event_data)
    
    def add_rule(
        self, 
        name: str, 
        patterns: List[str], 
        response_template: str, 
        log_level: LogLevel = LogLevel.WARNING
    ):
        """
        Add new compliance rule - MVP-FR-020 (Configuration-based rule updates)
        """
        from config.compliance import ComplianceRule
        
        self.rules[name] = ComplianceRule(
            name=name,
            patterns=patterns,
            response_template=response_template,
            log_level=log_level
        )
        
        logger.info("Added new compliance rule", rule_name=name, pattern_count=len(patterns))
    
    def remove_rule(self, name: str):
        """Remove compliance rule"""
        if name in self.rules:
            del self.rules[name]
            logger.info("Removed compliance rule", rule_name=name)
    
    def update_rule(
        self, 
        name: str, 
        patterns: Optional[List[str]] = None,
        response_template: Optional[str] = None,
        log_level: Optional[LogLevel] = None
    ):
        """Update existing compliance rule"""
        if name not in self.rules:
            raise ValueError(f"Rule {name} does not exist")
        
        rule = self.rules[name]
        
        if patterns is not None:
            rule.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        if response_template is not None:
            rule.response_template = response_template
        if log_level is not None:
            rule.log_level = log_level
        
        logger.info("Updated compliance rule", rule_name=name)
    
    def get_rule_stats(self) -> Dict[str, Any]:
        """Get compliance rule statistics"""
        return {
            "total_rules": len(self.rules),
            "rules": {
                name: {
                    "pattern_count": len(rule.patterns),
                    "log_level": rule.log_level.value
                }
                for name, rule in self.rules.items()
            },
            "professional_judgment_triggers": len(self.professional_judgment_triggers)
        }
    
    async def test_query_against_rules(self, query: str) -> Dict[str, Any]:
        """
        Test a query against all rules for debugging/testing purposes
        """
        results = {}
        
        for rule_name, rule in self.rules.items():
            matched_patterns = []
            for pattern in rule.patterns:
                if pattern.search(query):
                    matched_patterns.append(pattern.pattern)
            
            results[rule_name] = {
                "triggered": len(matched_patterns) > 0,
                "matched_patterns": matched_patterns
            }
        
        # Check professional judgment triggers
        pj_triggered = [
            trigger.pattern for trigger in self.professional_judgment_triggers
            if trigger.search(query)
        ]
        
        results["professional_judgment"] = {
            "triggered": len(pj_triggered) > 0,
            "matched_patterns": pj_triggered
        }
        
        return results