import os
import boto3
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "nba.db")

date_str = datetime.now().strftime("%Y_%m_%d")
csv_path = os.path.join(BASE_DIR, "box_scores", f"box_scores_{date_str}.csv")

print(f"Today's Date: {date_str}")
print(f"Reading: {csv_path}")

# Download from S3 if CSV doesn't exist locally
if not os.path.exists(csv_path):
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION')
    )

    print(f"CSV not found locally, downloading box_scores_{date_str}.csv from S3...")
    s3.download_file(os.getenv('S3_BUCKET_NAME'), f"box_scores_{date_str}.csv", csv_path)
    print("Downloaded successfully")
else:
    print("Using local CSV")

df = pd.read_csv(csv_path)

engine = create_engine(os.getenv('DATABASE_URL'))

# Check for existing IDs
try:
    with engine.connect() as conn:
        result = conn.execute(text('SELECT DISTINCT "gameId" FROM boxscores')) #('0022500001', 28, ....)
        existing_ids = [row[0] for row in result]
        new_rows = df[~df['gameId'].isin(existing_ids)]
        print(f"Found {len(existing_ids)} existing rows, {len(new_rows)} new rows")
except Exception as e:
    print(f"Table doesn't exist yet, inserting all rows: {e}")
    new_rows = df

# Insert in a separate connection
if not new_rows.empty:
    with engine.begin() as conn:
        new_rows.to_sql('boxscores', conn, if_exists='append', index=False)
        print(f"Inserted {len(new_rows)} new rows")
else:
    print("No new games to insert")