import os
import boto3
import sqlite3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "nba.db")
CSV_PATH = os.path.join(BASE_DIR, "box_scores_2025_26.csv")

# Download from S3 if CSV doesn't exist locally
if not os.path.exists(CSV_PATH):
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION')
    )

    print("CSV not found locally, downloading from S3...")
    s3.download_file(os.getenv('S3_BUCKET_NAME'), 'box_scores_2025_26.csv', CSV_PATH)
    print("Downloaded successfully")
else:
    print("Using local CSV")

df = pd.read_csv(CSV_PATH)
conn = sqlite3.connect(DB_PATH)
df.to_sql('boxscores', conn, if_exists='replace', index=False)
conn.close()

print(f"Loaded {len(df)} rows into boxscores table.")