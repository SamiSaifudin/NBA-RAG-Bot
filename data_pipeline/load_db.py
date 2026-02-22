import sqlite3
import pandas as pd

df = pd.read_csv('data_pipeline/box_scores_2025_26.csv')
conn = sqlite3.connect('data_pipeline/nba.db')
df.to_sql('boxscores', conn, if_exists='replace', index=False)
conn.close()