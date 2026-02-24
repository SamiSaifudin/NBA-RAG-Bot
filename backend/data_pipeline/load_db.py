import os
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data_pipeline", "box_scores_2025_26.csv")
DB_PATH = os.path.join(BASE_DIR, "data_pipeline", "nba.db")

df = pd.read_csv(CSV_PATH)
conn = sqlite3.connect(DB_PATH)
df.to_sql('boxscores', conn, if_exists='replace', index=False)
conn.close()

print(f"Loaded {len(df)} rows into boxscores table.")