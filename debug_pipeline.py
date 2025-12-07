import os
import sys
from app.pipeline import process_document

# Create a dummy excel file if needed
import pandas as pd
df = pd.DataFrame({'Data': [1, 2, 3]})
df.to_excel("debug_test.xlsx", index=False)

try:
    print("Testing process_document...")
    result, model = process_document("debug_test.xlsx", "format_template.json")
    print("Success!")
    print("Result keys:", result.keys())
    print("Classification:", result.get("classification") or result.get("metadata", {}).get("classification"))
except Exception as e:
    print("Caught error:")
    import traceback
    traceback.print_exc()
