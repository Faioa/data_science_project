import pandas as pd
import ast

def extract_names(column):
    if pd.isnull(column):
        return []
    try:
        parsed = ast.literal_eval(column)
        return [d['name'] for d in parsed]
    except (ValueError, SyntaxError):
        return []