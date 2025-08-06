#!/usr/bin/env python3
"""
GDPR Compliance Checker for Enterprise CI/CD
Validates GDPR compliance requirements in the codebase.
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class GDPRComplianceChecker:
    """GDPR compliance validation for fraud analytics platform."""
    
    def __init__(self):
        self.violations = []
        self.warnings = []
        self.compliance_score = 100
        
        # GDPR requirements checklist
        self.requirements = {
            'data_subject_rights': False,
            'consent_management': False,
            'data_retention': False,
            'data_portability': False,
            'privacy_by_design': False,
            'audit_logging': False,
            'data_protection_officer': False,
            'breach_notification': False,
        }
    
    def check_data_subject_rights(self) -> bool:
        """Check if data subject rights are implemented (Articles 15-22)."""
        print("🔍 Checking data subject rights implementation...")
        
        required_methods = [
            'handle_right_to_access',
            'handle_right_to_rectification', 
            'handle_right_to_erasure',
            'handle_right_to_portability',
            'handle_right_to_object',
        ]
        
        found_methods = []
        
        # Search for GDPR rights implementation
        for py_file in Path('.').rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for method in required_methods:
                    if method in content:
                        found_methods.append(method)
                        
            except (UnicodeDecodeError, PermissionError):
                continue
        
        missing_methods = set(required_methods) - set(found_methods)
        
        if missing_methods:
            self.violations.append(
                f"Missing data subject rights implementation: {', '.join(missing_methods)}"
            )
            self.compliance_score -= 15
            return False
        
        print("✅ Data subject rights implementation found")
        return True
    
    def check_consent_management(self) -> bool:
        """Check if consent management is implemented (Article 7)."""
        print("🔍 Checking consent management...")
        
        consent_patterns = [
            r'consent.*given',
            r'consent.*withdrawn',
            r'record.*consent',
            r'ConsentModel',
            r'consent_date',
        ]
        
        consent_found = False
        
        for py_file in Path('.').rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in consent_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        consent_found = True
                        break
                        
                if consent_found:
                    break
                    
            except (UnicodeDecodeError, PermissionError):
                continue
        
        if not consent_found:
            self.violations.append("No consent management implementation found")
            self.compliance_score -= 20
            return False
        
        print("✅ Consent management implementation found")
        return True
    
    def check_data_retention(self) -> bool:
        """Check if data retention policies are implemented (Article 5)."""
        print("🔍 Checking data retention policies...")
        
        retention_patterns = [
            r'retention.*policy',
            r'delete.*after',
            r'DataRetentionPolicy',
            r'retention_period',
            r'auto_delete',
        ]
        
        retention_found = False
        
        for py_file in Path('.').rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in retention_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        retention_found = True
                        break
                        
                if retention_found:
                    break
                    
            except (UnicodeDecodeError, PermissionError):
                continue
        
        if not retention_found:
            self.violations.append("No data retention policy implementation found")
            self.compliance_score -= 15
            return False
        
        print("✅ Data retention policy implementation found")
        return True
    
    def check_audit_logging(self) -> bool:
        """Check if audit logging is implemented (Article 30)."""
        print("🔍 Checking audit logging...")
        
        audit_patterns = [
            r'DataAccessLog',
            r'audit.*log',
            r'log.*access',
            r'processing.*record',
            r'_log_data_access',
        ]
        
        audit_found = False
        
        for py_file in Path('.').rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in audit_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        audit_found = True
                        break
                        
                if audit_found:
                    break
                    
            except (UnicodeDecodeError, PermissionError):
                continue
        
        if not audit_found:
            self.violations.append("No audit logging implementation found")
            self.compliance_score -= 20
            return False
        
        print("✅ Audit logging implementation found")
        return True
    
    def check_privacy_by_design(self) -> bool:
        """Check if privacy by design principles are followed (Article 25)."""
        print("🔍 Checking privacy by design implementation...")
        
        privacy_patterns = [
            r'data.*minimization',
            r'purpose.*limitation',
            r'pseudonymization',
            r'anonymization',
            r'encryption',
            r'privacy.*impact',
        ]
        
        privacy_features = 0
        
        for py_file in Path('.').rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in privacy_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        privacy_features += 1
                        
            except (UnicodeDecodeError, PermissionError):
                continue
        
        if privacy_features < 3:
            self.warnings.append(
                f"Limited privacy by design features found ({privacy_features}/6)"
            )
            self.compliance_score -= 10
            return False
        
        print(f"✅ Privacy by design features found ({privacy_features}/6)")
        return True
    
    def check_data_portability(self) -> bool:
        """Check if data portability is implemented (Article 20)."""
        print("🔍 Checking data portability...")
        
        portability_patterns = [
            r'export.*data',
            r'data.*portability',
            r'machine.*readable',
            r'_export_portable_data',
            r'portable.*format',
        ]
        
        portability_found = False
        
        for py_file in Path('.').rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in portability_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        portability_found = True
                        break
                        
                if portability_found:
                    break
                    
            except (UnicodeDecodeError, PermissionError):
                continue
        
        if not portability_found:
            self.violations.append("No data portability implementation found")
            self.compliance_score -= 15
            return False
        
        print("✅ Data portability implementation found")
        return True
    
    def check_sensitive_data_handling(self) -> bool:
        """Check if sensitive data is properly handled."""
        print("🔍 Checking sensitive data handling...")
        
        violations_found = []
        
        # Check for hardcoded sensitive data
        sensitive_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
        ]
        
        for py_file in Path('.').rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in sensitive_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        violations_found.append(f"Hardcoded sensitive data in {py_file}")
                        
            except (UnicodeDecodeError, PermissionError):
                continue
        
        if violations_found:
            for violation in violations_found:
                self.violations.append(violation)
            self.compliance_score -= len(violations_found) * 5
            return False
        
        print("✅ No hardcoded sensitive data found")
        return True
    
    def check_documentation(self) -> bool:
        """Check if GDPR documentation exists."""
        print("🔍 Checking GDPR documentation...")
        
        required_docs = [
            'PRIVACY_POLICY.md',
            'DATA_PROCESSING_AGREEMENT.md',
            'COOKIE_POLICY.md',
            'SECURITY.md',
        ]
        
        missing_docs = []
        
        for doc in required_docs:
            if not Path(doc).exists() and not Path(f'docs/{doc}').exists():
                missing_docs.append(doc)
        
        if missing_docs:
            self.warnings.append(f"Missing GDPR documentation: {', '.join(missing_docs)}")
            self.compliance_score -= len(missing_docs) * 3
        
        if len(missing_docs) < len(required_docs):
            print(f"✅ GDPR documentation found ({len(required_docs) - len(missing_docs)}/{len(required_docs)})")
            return True
        
        return False
    
    def run_compliance_check(self) -> Dict[str, any]:
        """Run complete GDPR compliance check."""
        print("🛡️ Starting GDPR Compliance Check")
        print("=" * 50)
        
        # Run all checks
        self.requirements['data_subject_rights'] = self.check_data_subject_rights()
        self.requirements['consent_management'] = self.check_consent_management()
        self.requirements['data_retention'] = self.check_data_retention()
        self.requirements['audit_logging'] = self.check_audit_logging()
        self.requirements['privacy_by_design'] = self.check_privacy_by_design()
        self.requirements['data_portability'] = self.check_data_portability()
        
        # Additional checks
        self.check_sensitive_data_handling()
        self.check_documentation()
        
        # Calculate compliance percentage
        implemented_requirements = sum(self.requirements.values())
        total_requirements = len(self.requirements)
        compliance_percentage = (implemented_requirements / total_requirements) * 100
        
        # Adjust score based on violations
        final_score = min(self.compliance_score, compliance_percentage)
        
        return {
            'compliance_score': final_score,
            'requirements_met': implemented_requirements,
            'total_requirements': total_requirements,
            'violations': self.violations,
            'warnings': self.warnings,
            'requirements_status': self.requirements,
        }


def main():
    """Main function to run GDPR compliance check."""
    checker = GDPRComplianceChecker()
    results = checker.run_compliance_check()
    
    print("\n📊 GDPR Compliance Results")
    print("=" * 50)
    print(f"Compliance Score: {results['compliance_score']:.1f}/100")
    print(f"Requirements Met: {results['requirements_met']}/{results['total_requirements']}")
    
    if results['violations']:
        print(f"\n❌ Violations ({len(results['violations'])}):")
        for violation in results['violations']:
            print(f"   • {violation}")
    
    if results['warnings']:
        print(f"\n⚠️  Warnings ({len(results['warnings'])}):")
        for warning in results['warnings']:
            print(f"   • {warning}")
    
    print(f"\n📋 Requirements Status:")
    for requirement, status in results['requirements_status'].items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {requirement.replace('_', ' ').title()}")
    
    # Set exit code based on compliance score
    if results['compliance_score'] >= 80:
        print("\n✅ GDPR Compliance Check PASSED")
        sys.exit(0)
    else:
        print("\n❌ GDPR Compliance Check FAILED")
        print(f"   Required: 80, Actual: {results['compliance_score']:.1f}")
        sys.exit(1)


if __name__ == "__main__":
    main()