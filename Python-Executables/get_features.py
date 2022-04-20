import sys
import pandas as pd

filename = sys.argv[1]

# Read Dataset
df = pd.read_csv(f"Dataset\\{filename}")

print("->".join(df.columns.tolist()))

sys.stdout.flush()