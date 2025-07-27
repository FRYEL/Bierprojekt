#!/usr/bin/env python3
"""
Dynamic LLM-Powered Data Quality Evaluator

This system uses LLMs to:
1. Analyze a dataset and automatically discover plausibility rules
2. Apply these rules to evaluate data quality with consistent scoring
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class PlausibilityRule:
    """Represents a discovered plausibility rule"""
    rule_id: str
    title: str
    description: str
    detection_logic: str
    affected_columns: List[str]
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create from dictionary"""
        return cls(**data)


class RuleDiscoveryAgent:
    """Discovers plausibility rules from a dataset using an LLM"""
    
    DISCOVERY_PROMPT = """You are a Data Quality Expert analyzing a dataset to discover plausibility rules. Your task is to identify logical inconsistencies, impossible combinations, and data quality issues that could exist in this type of data.

You will be given:
1. Dataset schema with column names and types
2. Statistical summary of the data
3. Sample rows from the dataset
4. The dataset context/domain

Your task is to generate plausibility rules that can detect data quality issues. Focus on:
- Logical contradictions between columns
- Business rule violations
- Statistical anomalies
- Domain-specific constraints
- Impossible value combinations

Output a JSON array of rules with this structure:
[
    {
        "rule_id": "unique_identifier",
        "title": "Short descriptive title",
        "description": "Detailed explanation of what this rule checks",
        "detection_logic": "Python-like pseudocode explaining how to detect violations",
        "affected_columns": ["column1", "column2", ...]
    }
]

Only output the JSON array, no other text."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def analyze_dataset(self, df: pd.DataFrame, context: str = "") -> Dict[str, Any]:
        """Analyze dataset to prepare information for rule discovery"""
        analysis = {
            "schema": {},
            "statistics": {},
            "sample_rows": [],
            "context": context
        }
        
        # Schema information
        for col in df.columns:
            analysis["schema"][col] = {
                "dtype": str(df[col].dtype),
                "unique_values": int(df[col].nunique()),
                "null_count": int(df[col].isnull().sum()),
                "sample_values": df[col].dropna().unique()[:5].tolist() if df[col].nunique() < 20 else None
            }
        
        # Statistical summary for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            analysis["statistics"][col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "percentiles": {
                    "25%": float(df[col].quantile(0.25)),
                    "50%": float(df[col].quantile(0.50)),
                    "75%": float(df[col].quantile(0.75))
                }
            }
        
        # Sample rows
        sample_size = min(10, len(df))
        sample_indices = np.random.choice(df.index, size=sample_size, replace=False)
        analysis["sample_rows"] = df.loc[sample_indices].to_dict('records')
        
        return analysis
    
    def discover_rules(self, df: pd.DataFrame, context: str = None, max_retries: int = 3) -> List[PlausibilityRule]:
        """Discover plausibility rules from the dataset"""
        # Analyze the dataset
        print("Analyzing dataset structure...")
        analysis = self.analyze_dataset(df, context or "")
        
        # Prepare context information
        context_info = ""
        if context:
            context_info = f"Dataset Context: {context}\n\n"
        
        # Create prompt
        user_prompt = f"""{context_info}Schema Information:
{json.dumps(analysis['schema'], indent=2)}

Statistical Summary:
{json.dumps(analysis['statistics'], indent=2)}

Sample Rows:
{json.dumps(analysis['sample_rows'], indent=2)}

Based on this analysis, generate plausibility rules to detect data quality issues."""

        for attempt in range(max_retries):
            try:
                print(f"Discovering rules (attempt {attempt + 1}/{max_retries})...")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.DISCOVERY_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Try to extract JSON from response
                import re
                
                # Try to find JSON array in the response
                json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group()
                else:
                    # If no array found, try to parse the whole response
                    json_text = response_text
                
                # Parse rules
                rules_data = json.loads(json_text)
                rules = [PlausibilityRule.from_dict(rule_data) for rule_data in rules_data]
                
                print(f"Discovered {len(rules)} plausibility rules")
                return rules
                
            except Exception as e:
                print(f"Error in rule discovery: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)


class DataEvaluationAgent:
    """Evaluates data rows using discovered rules"""
    
    EVALUATION_PROMPT_TEMPLATE = """You are a Data Quality Analyst. Evaluate the given data row against the provided plausibility rules.

Rules to check:
{rules_json}

For each rule, determine if it's violated by examining the data according to the detection_logic.

Your response must be a JSON object with these exact keys:
- "score": Integer 0-100 (start at 100, deduct {points_per_violation} points for each violated rule)
- "violations": List of violated rule IDs
- "details": Object mapping rule_id to explanation of why it was violated
- "summary": One-sentence summary of the data quality

Each rule violation deducts exactly {points_per_violation} points from the score.
The minimum score is 0.

Only output the JSON object."""
    
    def __init__(self, rules: List[PlausibilityRule], api_key: str = None, model: str = "gpt-4o-mini"):
        self.rules = rules
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        # Calculate dynamic points per violation based on number of rules
        self.points_per_violation = round(100 / len(rules)) if rules else 25
        print(f"Points per violation: {self.points_per_violation} (based on {len(rules)} rules)")
        
        # Prepare rules for prompt
        self.rules_json = json.dumps([rule.to_dict() for rule in rules], indent=2)
    
    def evaluate_row(self, row_data: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """Evaluate a single row against the rules"""
        prompt = self.EVALUATION_PROMPT_TEMPLATE.format(
            rules_json=self.rules_json,
            points_per_violation=self.points_per_violation
        )
        user_prompt = f"Evaluate this data row:\n{json.dumps(row_data, indent=2)}"
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Try to extract JSON from response
                import re
                
                # Try to find JSON object in the response
                json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', response_text, re.DOTALL)
                if json_match:
                    result_text = json_match.group()
                else:
                    # If no object found, try to parse the whole response
                    result_text = response_text
                
                result = json.loads(result_text)
                return result
                
            except Exception as e:
                if attempt == max_retries - 1:
                    return {
                        "score": 0,
                        "violations": ["ERROR"],
                        "details": {"ERROR": str(e)},
                        "summary": f"Evaluation failed: {str(e)}"
                    }
                time.sleep(2 ** attempt)
    
    def evaluate_dataframe(self, df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
        """Evaluate all rows in the dataframe"""
        result_df = df.copy()
        
        scores = []
        violations = []
        details = []
        summaries = []
        
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            result = self.evaluate_row(row_dict)
            
            scores.append(result['score'])
            violations.append(json.dumps(result['violations']))
            details.append(json.dumps(result['details']))
            summaries.append(result['summary'])
            
            if progress_callback:
                progress_callback(idx + 1, total_rows)
            else:
                print(f"Evaluated row {idx + 1}/{total_rows}")
            
            time.sleep(0.1)  # Rate limiting
        
        result_df['quality_score'] = scores
        result_df['rule_violations'] = violations
        result_df['violation_details'] = details
        result_df['quality_summary'] = summaries
        
        return result_df


class DynamicDataEvaluator:
    """Main orchestrator for the dynamic evaluation system"""
    
    def __init__(self, api_key: str = None, discovery_model: str = "gpt-4o", 
                 evaluation_model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.discovery_agent = RuleDiscoveryAgent(api_key, discovery_model)
        self.evaluation_model = evaluation_model
    
    def discover_and_save_rules(self, df: pd.DataFrame, context: str, 
                               output_path: str) -> List[PlausibilityRule]:
        """Discover rules and save them to a file"""
        rules = self.discovery_agent.discover_rules(df, context)
        
        # Save rules
        rules_data = [rule.to_dict() for rule in rules]
        with open(output_path, 'w') as f:
            json.dump(rules_data, f, indent=2)
        
        print(f"Saved {len(rules)} rules to {output_path}")
        return rules
    
    def load_rules(self, rules_path: str) -> List[PlausibilityRule]:
        """Load rules from a file"""
        with open(rules_path, 'r') as f:
            rules_data = json.load(f)
        
        rules = [PlausibilityRule.from_dict(rule_data) for rule_data in rules_data]
        print(f"Loaded {len(rules)} rules from {rules_path}")
        return rules
    
    def evaluate_with_rules(self, df: pd.DataFrame, rules: List[PlausibilityRule]) -> pd.DataFrame:
        """Evaluate dataframe with given rules"""
        evaluation_agent = DataEvaluationAgent(rules, self.api_key, self.evaluation_model)
        return evaluation_agent.evaluate_dataframe(df)
    
    def full_pipeline(self, df: pd.DataFrame, context: str, 
                     rules_output: str = "discovered_rules.json") -> Tuple[List[PlausibilityRule], pd.DataFrame]:
        """Run the full pipeline: discover rules and evaluate"""
        # Discover rules
        rules = self.discover_and_save_rules(df, context, rules_output)
        
        # Display discovered rules
        print("\n=== Discovered Rules ===")
        for rule in rules:
            print(f"\n{rule.title}")
            print(f"  Description: {rule.description}")
            print(f"  Affected columns: {', '.join(rule.affected_columns)}")
        
        points_per_violation = round(100 / len(rules)) if rules else 25
        print(f"\nScoring: Each rule violation deducts {points_per_violation} points")
        
        # Evaluate data
        print("\n=== Evaluating Data ===")
        result_df = self.evaluate_with_rules(df, rules)
        
        return rules, result_df


def load_context(context_input: str) -> str:
    """Load context from file or return as string"""
    if context_input and os.path.isfile(context_input):
        print(f"Loading context from file: {context_input}")
        with open(context_input, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return context_input


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic data quality evaluation using LLMs"
    )
    parser.add_argument("input_file", help="Input CSV file for rule discovery OR evaluation")
    parser.add_argument("-e", "--evaluate-file", 
                       help="CSV file to evaluate (if different from input_file)")
    parser.add_argument("-c", "--context", 
                       help="Dataset context - can be a text string or path to a text file (optional)")
    parser.add_argument("-o", "--output", default="evaluated_output.csv",
                       help="Output CSV file")
    parser.add_argument("-r", "--rules-file", 
                       help="Use existing rules file instead of discovering")
    parser.add_argument("--save-rules", default="discovered_rules.json",
                       help="Where to save discovered rules")
    parser.add_argument("--discovery-model", default="gpt-4o",
                       help="Model for rule discovery")
    parser.add_argument("--evaluation-model", default="gpt-4o-mini",
                       help="Model for data evaluation")
    parser.add_argument("-k", "--api-key", help="OpenAI API key")
    parser.add_argument("--sample", type=int, help="Process only N rows for rule discovery")
    
    args = parser.parse_args()
    
    # Determine which file to use for evaluation
    eval_file = args.evaluate_file if args.evaluate_file else args.input_file
    
    # Load context if provided
    context = None
    if args.context:
        context = load_context(args.context)
        if context:
            print(f"Context loaded: {context[:100]}..." if len(context) > 100 else f"Context: {context}")
    
    # Initialize evaluator
    evaluator = DynamicDataEvaluator(
        api_key=args.api_key,
        discovery_model=args.discovery_model,
        evaluation_model=args.evaluation_model
    )
    
    # Check if we should use existing rules or discover new ones
    if args.rules_file and os.path.exists(args.rules_file):
        print(f"Using existing rules from {args.rules_file}")
        rules = evaluator.load_rules(args.rules_file)
        
        # Load evaluation data
        print(f"Loading evaluation data from {eval_file}...")
        eval_df = pd.read_csv(eval_file)
        
        result_df = evaluator.evaluate_with_rules(eval_df, rules)
    else:
        # Load data for rule discovery
        print(f"Loading data for rule discovery from {args.input_file}...")
        discovery_df = pd.read_csv(args.input_file)
        
        if args.sample:
            discovery_df = discovery_df.head(args.sample)
            print(f"Using sample of {len(discovery_df)} rows for rule discovery")
        
        # Discover rules
        rules = evaluator.discover_and_save_rules(discovery_df, context, args.save_rules)
        
        # Display discovered rules
        print("\n=== Discovered Rules ===")
        for rule in rules:
            print(f"\n{rule.title}")
            print(f"  Description: {rule.description}")
            print(f"  Affected columns: {', '.join(rule.affected_columns)}")
        
        points_per_violation = round(100 / len(rules)) if rules else 25
        print(f"\nScoring: Each rule violation deducts {points_per_violation} points")
        
        # Load evaluation data
        print(f"\n=== Loading evaluation data from {eval_file} ===")
        eval_df = pd.read_csv(eval_file)
        
        # Evaluate data
        print("\n=== Evaluating Data ===")
        result_df = evaluator.evaluate_with_rules(eval_df, rules)
    
    # Save results
    result_df.to_csv(args.output, index=False)
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Total rows evaluated: {len(result_df)}")
    print(f"Average quality score: {result_df['quality_score'].mean():.1f}")
    print(f"Perfect quality rows (100): {(result_df['quality_score'] == 100).sum()}")
    print(f"Poor quality rows (<50): {(result_df['quality_score'] < 50).sum()}")
    
    # Violation statistics
    all_violations = []
    for v in result_df['rule_violations']:
        all_violations.extend(json.loads(v))
    
    if all_violations:
        from collections import Counter
        violation_counts = Counter(all_violations)
        print("\nMost common rule violations:")
        for rule_id, count in violation_counts.most_common(5):
            rule = next((r for r in rules if r.rule_id == rule_id), None)
            if rule:
                print(f"  - {rule.title}: {count} violations")
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()



python dynamic_evaluator.py data/real_beer_data.csv \
    --evaluate-file data/synthetic_beer_data.csv \
    --rules-file beer_rules.json \
    --output synthetic_evaluation.csv