#!/usr/bin/env python3
"""
LLM-Powered Semantic Evaluator for Synthetic Beer Preference Data

This script reads synthetic beer preference data from a CSV file,
evaluates each row using an LLM for semantic and logical consistency,
and outputs the results with scores and explanations.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple, Any

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a Data Quality Analyst specializing in survey data. Your task is to evaluate data rows from a survey about the beer consumption preferences of Generation Z in Germany for semantic and logical consistency.

You will be given a data row in JSON format. The data is based on a questionnaire where most scales range from 1 to 5. You must evaluate the data against the following rules:

**Evaluation Rules:**
1.  **Abstinence Contradiction:** A user who states that completely avoiding alcohol is 'very important' (alkoholverzicht_wichtig == 5) cannot also show a 'high' or 'very high' preference for normal alcoholic beer (praef_bier_alkohol >= 4). The value 5 in 'alkoholverzicht_wichtig' corresponds to the answer 'trifft voll und ganz zu'[cite: 36].
2.  **Disinterest vs. Frequency:** A user who has 'no interest at all' in beer (involvement_interesse == 1) and for whom beer plays 'no role at all' (involvement_rolle == 1) cannot be a frequent consumer. A violation occurs if 'konsumhaeufigkeit' is 'täglich' or 'mehrmals pro Woche'[cite: 21, 22, 34]. The value 1 in the involvement questions corresponds to 'trifft überhaupt nicht zu'[cite: 34].
3.  **Health Contradiction:** A user who considers healthy nutrition 'very important' (gesundheit_wichtig == 5) and for whom avoiding alcohol is also 'very important' (alkoholverzicht_wichtig == 5) cannot state that beer consumption is 'very advantageous' (involvement_vorteilhaft >= 4). The value 5 in 'gesundheit_wichtig' and 'alkoholverzicht_wichtig' corresponds to 'trifft voll und ganz zu', while a value >= 4 in 'involvement_vorteilhaft' corresponds to 'trifft eher zu' or 'trifft voll und ganz zu'[cite: 34, 36].
4.  **Sustainability Value Mismatch:** A user who states that sustainable production has a 'very strong influence' on their purchase decision (einfluss_nachhaltigkeit == 5) should be willing to pay more for organic products. A violation occurs if their preparedness to pay a premium for organic drinks is low (bio_mehrpreis_bereit <= 2). The value 5 for 'einfluss_nachhaltigkeit' means 'sehr starker Einfluss', while a value <= 2 for 'bio_mehrpreis_bereit' means they are 'eher nicht' or 'überhaupt nicht' prepared to pay more[cite: 36, 39].


**Your Response Format:**
You MUST respond with a JSON object only. Do not provide any conversational text or explanations outside of the JSON structure. The JSON object must have the following three keys:
- "score": An integer from 0 to 100, where 100 is a perfectly logical and consistent data row, and 0 is completely nonsensical. Deduct 25 points for each broken rule.
- "broken_rules": A JSON list of strings, where each string is the title of a rule that was violated (e.g., ["Abstinence Contradiction", "Health Contradiction"]). If no rules are broken, return an empty list [].
- "explanation": A brief, one-sentence string explaining your reasoning for the score.

Do not be conversational. Only return the JSON object."""

# The USER_PROMPT_TEMPLATE can remain the same as it is generic.
USER_PROMPT_TEMPLATE = """Please evaluate the following beer preference data row for semantic consistency based on the rules you have been given.
Data:
{data_row_as_json_string}"""


class BeerDataEvaluator:
    """Evaluates beer preference data using an LLM."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        """
        Initialize the evaluator with OpenAI API credentials.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment.
            model: The OpenAI model to use for evaluation.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass it as parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
    def evaluate_row(self, row_data: Dict[str, Any], max_retries: int = 3) -> Tuple[int, List[str], str]:
        """
        Evaluate a single row of beer preference data.
        
        Args:
            row_data: Dictionary containing the beer preference data
            max_retries: Maximum number of API call retries
            
        Returns:
            Tuple of (score, broken_rules, explanation)
        """
        # Convert row data to JSON string for the prompt
        data_json = json.dumps(row_data, indent=2)
        user_prompt = USER_PROMPT_TEMPLATE.format(data_row_as_json_string=data_json)
        
        for attempt in range(max_retries):
            try:
                # Make API call to OpenAI
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistency
                    max_tokens=500
                )
                
                # Parse the response
                response_text = response.choices[0].message.content.strip()
                
                # Try to parse as JSON
                try:
                    result = json.loads(response_text)
                    
                    # Validate the response structure
                    if not all(key in result for key in ["score", "broken_rules", "explanation"]):
                        raise ValueError("Missing required keys in response")
                    
                    score = int(result["score"])
                    broken_rules = result["broken_rules"]
                    explanation = result["explanation"]
                    
                    return score, broken_rules, explanation
                    
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error parsing LLM response on attempt {attempt + 1}: {e}")
                    print(f"Response was: {response_text}")
                    
                    if attempt == max_retries - 1:
                        # Return default values on final failure
                        return 0, ["Error: Failed to parse LLM response"], "Could not evaluate due to parsing error"
                    
            except Exception as e:
                print(f"API call error on attempt {attempt + 1}: {e}")
                
                if attempt == max_retries - 1:
                    # Return default values on final failure
                    return 0, ["Error: API call failed"], f"Could not evaluate due to API error: {str(e)}"
                
                # Wait before retrying (exponential backoff)
                time.sleep(2 ** attempt)
                
    def evaluate_dataframe(self, df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
        """
        Evaluate all rows in a DataFrame.
        
        Args:
            df: DataFrame containing beer preference data
            progress_callback: Optional callback function(current, total) for progress updates
            
        Returns:
            DataFrame with added evaluation columns
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Initialize new columns
        scores = []
        broken_rules_list = []
        explanations = []
        
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            # Convert row to dictionary
            row_dict = row.to_dict()
            
            # Evaluate the row
            score, broken_rules, explanation = self.evaluate_row(row_dict)
            
            # Store results
            scores.append(score)
            broken_rules_list.append(json.dumps(broken_rules))  # Store as JSON string
            explanations.append(explanation)
            
            # Progress update
            if progress_callback:
                progress_callback(idx + 1, total_rows)
            else:
                print(f"Evaluated row {idx + 1}/{total_rows}")
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        # Add results to DataFrame
        result_df['score'] = scores
        result_df['broken_rules'] = broken_rules_list
        result_df['explanation'] = explanations
        
        return result_df


def main():
    """Main function to run the beer data evaluator."""
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic beer preference data using an LLM"
    )
    parser.add_argument(
        "input_file",
        help="Path to the input CSV file containing beer preference data"
    )
    parser.add_argument(
        "-o", "--output",
        default="evaluated_output.csv",
        help="Path to the output CSV file (default: evaluated_output.csv)"
    )
    parser.add_argument(
        "-m", "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "-k", "--api-key",
        help="OpenAI API key (alternatively set OPENAI_API_KEY environment variable)"
    )
    
    args = parser.parse_args()
    
    # === DEBUG CODE START ===
    print(f"\n=== DEBUG: Checking output path ===")
    print(f"Output path: {args.output}")
    print(f"Output path exists: {os.path.exists(args.output)}")
    print(f"Output path is file: {os.path.isfile(args.output)}")
    print(f"Output path is directory: {os.path.isdir(args.output)}")
    
    # If it's a directory, suggest a filename
    if os.path.isdir(args.output):
        suggested_path = os.path.join(args.output, "evaluated_output.csv")
        print(f"Output is a directory! Suggested path: {suggested_path}")
        print(f"Using suggested path instead...")
        args.output = suggested_path
    
    print(f"Final output path: {args.output}")
    print(f"=== DEBUG CODE END ===\n")
    # === DEBUG CODE END ===
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
    
    try:
        # Read the input CSV
        print(f"Reading data from {args.input_file}...")
        df = pd.read_csv(args.input_file)
        # df = df.head(5)

        # FOR DEBUGGING: Process only the first 5 rows
        print(f"Loaded {len(df)} rows of data.")
        
        # Initialize evaluator
        print("Initializing LLM evaluator...")
        evaluator = BeerDataEvaluator(api_key=args.api_key, model=args.model)
        
        # Evaluate the data
        print("Starting evaluation...")
        print("This may take a while depending on the number of rows and API rate limits.")
        
        def progress_update(current, total):
            print(f"Progress: {current}/{total} rows evaluated ({current/total*100:.1f}%)")
        
        result_df = evaluator.evaluate_dataframe(df, progress_callback=progress_update)
        
        # Save the results
        print(f"\nSaving results to {args.output}...")
        result_df.to_csv(args.output, index=False)
        
        # Print summary statistics
        print("\n=== Evaluation Summary ===")
        print(f"Total rows evaluated: {len(result_df)}")
        print(f"Average score: {result_df['score'].mean():.1f}")
        print(f"Rows with perfect score (100): {(result_df['score'] == 100).sum()}")
        print(f"Rows with issues (score < 100): {(result_df['score'] < 100).sum()}")
        
        # Show most common broken rules
        all_broken_rules = []
        for rules_str in result_df['broken_rules']:
            rules = json.loads(rules_str)
            all_broken_rules.extend([r for r in rules if not r.startswith("Error:")])
        
        if all_broken_rules:
            from collections import Counter
            rule_counts = Counter(all_broken_rules)
            print("\nMost common broken rules:")
            for rule, count in rule_counts.most_common(5):
                print(f"  - {rule}: {count} occurrences")
        
        print(f"\nEvaluation complete! Results saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()