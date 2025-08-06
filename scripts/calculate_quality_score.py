#!/usr/bin/env python3
"""
Code Quality Score Calculator for Enterprise CI/CD
Calculates overall code quality score based on multiple metrics.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str) -> tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr."""
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True
    )
    return result.returncode, result.stdout, result.stderr


def calculate_complexity_score() -> float:
    """Calculate complexity score from radon output."""
    try:
        with open('complexity-report.json', 'r') as f:
            complexity_data = json.load(f)
        
        total_complexity = 0
        total_functions = 0
        
        for file_path, file_data in complexity_data.items():
            for item in file_data:
                if item['type'] == 'function':
                    total_complexity += item['complexity']
                    total_functions += 1
        
        if total_functions == 0:
            return 100.0
        
        avg_complexity = total_complexity / total_functions
        
        # Score based on average complexity (lower is better)
        if avg_complexity <= 5:
            return 100.0
        elif avg_complexity <= 10:
            return 80.0
        elif avg_complexity <= 15:
            return 60.0
        elif avg_complexity <= 20:
            return 40.0
        else:
            return 20.0
            
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return 50.0  # Default score if analysis fails


def calculate_maintainability_score() -> float:
    """Calculate maintainability score from radon output."""
    try:
        with open('maintainability-report.json', 'r') as f:
            maintainability_data = json.load(f)
        
        total_mi = 0
        total_files = 0
        
        for file_path, mi_score in maintainability_data.items():
            if isinstance(mi_score, (int, float)):
                total_mi += mi_score
                total_files += 1
        
        if total_files == 0:
            return 100.0
        
        avg_mi = total_mi / total_files
        
        # Maintainability Index scoring
        if avg_mi >= 85:
            return 100.0
        elif avg_mi >= 70:
            return 80.0
        elif avg_mi >= 50:
            return 60.0
        elif avg_mi >= 25:
            return 40.0
        else:
            return 20.0
            
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return 50.0


def calculate_test_coverage_score() -> float:
    """Calculate test coverage score."""
    try:
        # Run coverage report
        exit_code, stdout, stderr = run_command("coverage report --format=json")
        
        if exit_code != 0:
            return 50.0
        
        coverage_data = json.loads(stdout)
        coverage_percent = coverage_data.get('totals', {}).get('percent_covered', 0)
        
        # Score based on coverage percentage
        if coverage_percent >= 90:
            return 100.0
        elif coverage_percent >= 80:
            return 90.0
        elif coverage_percent >= 70:
            return 80.0
        elif coverage_percent >= 60:
            return 70.0
        elif coverage_percent >= 50:
            return 60.0
        else:
            return max(coverage_percent, 20.0)
            
    except (json.JSONDecodeError, KeyError):
        return 50.0


def calculate_lint_score() -> float:
    """Calculate linting score based on flake8 output."""
    try:
        exit_code, stdout, stderr = run_command("flake8 apps/ fraud_platform/ --statistics")
        
        if exit_code == 0:
            return 100.0  # No linting errors
        
        # Count total errors
        lines = stdout.strip().split('\n')
        total_errors = 0
        
        for line in lines:
            if line and line[0].isdigit():
                error_count = int(line.split()[0])
                total_errors += error_count
        
        # Score based on error density
        total_files = len(list(Path('apps').rglob('*.py'))) + len(list(Path('fraud_platform').rglob('*.py')))
        error_density = total_errors / max(total_files, 1)
        
        if error_density == 0:
            return 100.0
        elif error_density <= 1:
            return 90.0
        elif error_density <= 3:
            return 70.0
        elif error_density <= 5:
            return 50.0
        else:
            return 30.0
            
    except (ValueError, IndexError):
        return 50.0


def calculate_security_score() -> float:
    """Calculate security score based on bandit output."""
    try:
        exit_code, stdout, stderr = run_command("bandit -r apps/ fraud_platform/ -f json")
        
        if exit_code == 0:
            return 100.0  # No security issues
        
        try:
            bandit_data = json.loads(stdout)
            results = bandit_data.get('results', [])
            
            high_severity = len([r for r in results if r.get('issue_severity') == 'HIGH'])
            medium_severity = len([r for r in results if r.get('issue_severity') == 'MEDIUM'])
            low_severity = len([r for r in results if r.get('issue_severity') == 'LOW'])
            
            # Weighted scoring
            security_score = 100 - (high_severity * 20) - (medium_severity * 10) - (low_severity * 5)
            return max(security_score, 0.0)
            
        except json.JSONDecodeError:
            return 70.0  # Partial score if JSON parsing fails
            
    except Exception:
        return 70.0


def calculate_documentation_score() -> float:
    """Calculate documentation score based on docstring coverage."""
    try:
        exit_code, stdout, stderr = run_command("interrogate apps/ fraud_platform/ --quiet")
        
        # Parse interrogate output for coverage percentage
        lines = stdout.strip().split('\n')
        for line in lines:
            if 'Overall' in line and '%' in line:
                # Extract percentage from line like "Overall: 85.5%"
                percentage_str = line.split('%')[0].split()[-1]
                percentage = float(percentage_str)
                return min(percentage, 100.0)
        
        return 50.0  # Default if parsing fails
        
    except (ValueError, IndexError):
        return 50.0


def calculate_overall_quality_score() -> float:
    """Calculate overall quality score with weighted components."""
    
    print("🔍 Calculating code quality metrics...")
    
    # Calculate individual scores
    complexity_score = calculate_complexity_score()
    maintainability_score = calculate_maintainability_score()
    test_coverage_score = calculate_test_coverage_score()
    lint_score = calculate_lint_score()
    security_score = calculate_security_score()
    documentation_score = calculate_documentation_score()
    
    # Print individual scores
    print(f"📊 Quality Metrics:")
    print(f"   Complexity Score: {complexity_score:.1f}/100")
    print(f"   Maintainability Score: {maintainability_score:.1f}/100")
    print(f"   Test Coverage Score: {test_coverage_score:.1f}/100")
    print(f"   Lint Score: {lint_score:.1f}/100")
    print(f"   Security Score: {security_score:.1f}/100")
    print(f"   Documentation Score: {documentation_score:.1f}/100")
    
    # Weighted overall score
    weights = {
        'complexity': 0.15,
        'maintainability': 0.15,
        'test_coverage': 0.25,
        'lint': 0.20,
        'security': 0.20,
        'documentation': 0.05,
    }
    
    overall_score = (
        complexity_score * weights['complexity'] +
        maintainability_score * weights['maintainability'] +
        test_coverage_score * weights['test_coverage'] +
        lint_score * weights['lint'] +
        security_score * weights['security'] +
        documentation_score * weights['documentation']
    )
    
    print(f"🎯 Overall Quality Score: {overall_score:.1f}/100")
    
    return overall_score


def main():
    """Main function to calculate and output quality score."""
    try:
        overall_score = calculate_overall_quality_score()
        
        # Write score to file for CI/CD pipeline
        with open('quality_score.txt', 'w') as f:
            f.write(str(int(overall_score)))
        
        # Set appropriate exit code
        if overall_score >= 80:
            print("✅ Quality gate PASSED")
            sys.exit(0)
        else:
            print("❌ Quality gate FAILED")
            print(f"   Required: 80, Actual: {overall_score:.1f}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error calculating quality score: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()