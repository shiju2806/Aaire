#!/usr/bin/env python3
"""
Configuration Audit Tool
Scans codebase for hardcoded configuration values that should use centralized config
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import json

class ConfigurationAuditor:
    def __init__(self, src_path: str = "/Users/shijuprakash/AAIRE/src"):
        self.src_path = Path(src_path)
        self.violations = []

        # Patterns to detect hardcoded configuration values
        self.patterns = {
            "thresholds": [
                r'threshold\s*=\s*([0-9]+\.[0-9]+)',
                r'([0-9]+\.[0-9]+)\s*.*threshold',
                r'min.*=\s*([0-9]+\.[0-9]+)',
                r'max.*=\s*([0-9]+\.[0-9]+)'
            ],
            "model_params": [
                r'temperature\s*=\s*([0-9]+\.[0-9]*)',
                r'max_tokens\s*=\s*([0-9]+)',
                r'model.*=\s*["\']([^"\']+)["\']',
                r'top_[kp]\s*=\s*([0-9]+\.[0-9]*)'
            ],
            "limits": [
                r'limit\s*=\s*([0-9]+)',
                r'batch_size\s*=\s*([0-9]+)',
                r'chunk_size\s*=\s*([0-9]+)',
                r'timeout\s*=\s*([0-9]+)'
            ],
            "weights": [
                r'weight\s*=\s*([0-9]+\.[0-9]+)',
                r'alpha\s*=\s*([0-9]+\.[0-9]+)',
                r'beta\s*=\s*([0-9]+\.[0-9]+)'
            ]
        }

        # Constructor patterns to detect missing config injection
        self.constructor_patterns = [
            r'def __init__\(self\):',  # No parameters at all
            r'def __init__\(self,(?!.*config)[^)]*\):',  # Parameters but no config
        ]

    def scan_file(self, file_path: Path) -> Dict:
        """Scan a single file for configuration violations"""
        violations = {
            "file": str(file_path),
            "hardcoded_values": [],
            "missing_config_injection": [],
            "line_count": 0
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                violations["line_count"] = len(lines)

                # Check for hardcoded values
                for category, patterns in self.patterns.items():
                    for pattern in patterns:
                        for line_num, line in enumerate(lines, 1):
                            matches = re.finditer(pattern, line, re.IGNORECASE)
                            for match in matches:
                                violations["hardcoded_values"].append({
                                    "category": category,
                                    "line": line_num,
                                    "content": line.strip(),
                                    "value": match.group(1) if match.groups() else match.group(0),
                                    "pattern": pattern
                                })

                # Check for missing config injection
                for pattern in self.constructor_patterns:
                    matches = re.finditer(pattern, content, re.MULTILINE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        violations["missing_config_injection"].append({
                            "line": line_num,
                            "content": lines[line_num-1].strip(),
                            "pattern": pattern
                        })

        except Exception as e:
            violations["error"] = str(e)

        return violations

    def scan_directory(self) -> Dict:
        """Scan entire src directory for violations"""
        results = {
            "summary": {
                "files_scanned": 0,
                "files_with_violations": 0,
                "total_hardcoded_values": 0,
                "total_missing_config": 0
            },
            "violations_by_category": {},
            "high_priority_files": [],
            "detailed_results": []
        }

        # Scan all Python files
        for py_file in self.src_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            results["summary"]["files_scanned"] += 1
            violations = self.scan_file(py_file)

            if violations["hardcoded_values"] or violations["missing_config_injection"]:
                results["summary"]["files_with_violations"] += 1
                results["summary"]["total_hardcoded_values"] += len(violations["hardcoded_values"])
                results["summary"]["total_missing_config"] += len(violations["missing_config_injection"])

                # Categorize violations
                for violation in violations["hardcoded_values"]:
                    category = violation["category"]
                    if category not in results["violations_by_category"]:
                        results["violations_by_category"][category] = []
                    results["violations_by_category"][category].append({
                        "file": violations["file"],
                        "line": violation["line"],
                        "value": violation["value"]
                    })

                # Mark high priority files (validators, core components)
                if any(keyword in str(py_file).lower() for keyword in ["validator", "quality", "alignment"]):
                    results["high_priority_files"].append({
                        "file": violations["file"],
                        "hardcoded_count": len(violations["hardcoded_values"]),
                        "missing_config_count": len(violations["missing_config_injection"])
                    })

                results["detailed_results"].append(violations)

        return results

    def generate_report(self, results: Dict) -> str:
        """Generate human-readable audit report"""
        report = []
        report.append("🔍 CONFIGURATION AUDIT REPORT")
        report.append("=" * 50)

        # Summary
        summary = results["summary"]
        report.append(f"\n📊 SUMMARY:")
        report.append(f"  Files scanned: {summary['files_scanned']}")
        report.append(f"  Files with violations: {summary['files_with_violations']}")
        report.append(f"  Hardcoded values found: {summary['total_hardcoded_values']}")
        report.append(f"  Missing config injection: {summary['total_missing_config']}")

        # High priority files
        if results["high_priority_files"]:
            report.append(f"\n🚨 HIGH PRIORITY FILES:")
            for file_info in sorted(results["high_priority_files"],
                                  key=lambda x: x['hardcoded_count'], reverse=True):
                report.append(f"  {file_info['file']}:")
                report.append(f"    Hardcoded values: {file_info['hardcoded_count']}")
                report.append(f"    Missing config: {file_info['missing_config_count']}")

        # Violations by category
        if results["violations_by_category"]:
            report.append(f"\n📋 VIOLATIONS BY CATEGORY:")
            for category, violations in results["violations_by_category"].items():
                report.append(f"  {category.upper()}: {len(violations)} violations")
                for v in violations[:3]:  # Show first 3 examples
                    report.append(f"    {v['file']}:{v['line']} = {v['value']}")
                if len(violations) > 3:
                    report.append(f"    ... and {len(violations) - 3} more")

        return "\n".join(report)

    def export_json(self, results: Dict, output_path: str = "config_audit.json"):
        """Export detailed results to JSON for processing"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

def main():
    auditor = ConfigurationAuditor()

    print("🔍 Scanning codebase for configuration violations...")
    results = auditor.scan_directory()

    # Generate and display report
    report = auditor.generate_report(results)
    print(report)

    # Export detailed results
    auditor.export_json(results)
    print(f"\n📄 Detailed results exported to: config_audit.json")

    # Specific focus on quality validators
    print(f"\n🎯 QUALITY VALIDATOR ANALYSIS:")
    validator_files = [r for r in results["detailed_results"]
                      if "validator" in r["file"].lower()]

    for validator in validator_files:
        if validator["hardcoded_values"]:
            print(f"  {validator['file']}:")
            for hv in validator["hardcoded_values"]:
                print(f"    Line {hv['line']}: {hv['content']}")

if __name__ == "__main__":
    main()