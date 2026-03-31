import os
import boto3
import pandas as pd
from datetime import datetime
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

FETCH_BATCH_SIZE = 100
UPSERT_BATCH_SIZE = 90
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

date_str = datetime.now().strftime("%Y_%m_%d")
csv_path = os.path.join(BASE_DIR, "box_scores", f"box_scores_{date_str}.csv")

print(f"Today's Date: {date_str}")
print(f"Reading: {csv_path}")

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index('clutchquery')

# format date: '2026-02-23' -> 'Monday, February 23rd, 2026'
def format_game_date(iso_date_str: str) -> str:
    if not iso_date_str:
        return ""
    dt = datetime.strptime(iso_date_str, "%Y-%m-%d")

    day = dt.day
    
    if 11 <= day <= 13:
        suffix = "th"
    else:
        last_digit = day % 10
        if last_digit == 1:
            suffix = "st"
        elif last_digit == 2:
            suffix = "nd"
        elif last_digit == 3:
            suffix = "rd"
        else:
            suffix = "th"

    return f"{dt.strftime('%A, %B')} {day}{suffix}, {dt.year}"

# Converts row to text so we can embed
def row_to_text(row: pd.Series) -> str:
    name = f"{row.get('firstName', '')} {row.get('lastName', row.get('familyName', ''))}".strip()
    minutes = row.get("minutes") or "0"

    points = row.get("points", 0)
    assists = row.get("assists", 0)

    steals = row.get("steals", 0)
    blocks = row.get("blocks", 0)

    turnovers = row.get("turnovers", 0)
    plus_minus = row.get("plusMinusPoints", 0)

    fouls = row.get("foulsPersonal", 0)

    fg_pct = (row.get("fieldGoalsPercentage") or 0) * 100
    fg_made = row.get("fieldGoalsMade", 0)
    fg_att = row.get("fieldGoalsAttempted", 0)

    fg3_pct = (row.get("threePointersPercentage", 0) or 0) * 100
    fg3_made = row.get("threePointersMade", 0)
    fg3_att = row.get("threePointersAttempted", 0)

    ft_pct = (row.get("freeThrowsPercentage", 0) or 0) * 100
    ft_made = row.get("freeThrowsMade", 0)
    ft_att = row.get("freeThrowsAttempted", 0)

    total_rebounds = row.get("reboundsTotal", 0)
    offensive_rebounds = row.get("reboundsOffensive", 0)
    defensive_rebounds = row.get("reboundsDefensive", 0)

    team_name = row.get("teamName", "")
    opponent = row.get("opponent", "")
    game_id = row.get("gameId", "")

    raw_date = row.get("game_date", "")
    game_date = format_game_date(raw_date)

    ts_pct = (row.get("trueShootingPercentage") or 0) * 100
    
    return (
        f"{name} played for {team_name} vs {opponent} on {game_date} (game {game_id}). "
        f"In {minutes} minutes he scored {points} points: "
        f"FG {fg_made}/{fg_att} ({fg_pct:.1f}%), 3PT {fg3_made}/{fg3_att} ({fg3_pct:.1f}%), FT {ft_made}/{ft_att} ({ft_pct:.1f}%), True Shooting: {ts_pct:.1f}%. "
        f"He had {total_rebounds} rebounds ({offensive_rebounds} offensive, {defensive_rebounds} defensive), "
        f"{assists} assists, {steals} steals, {blocks} blocks, {turnovers} turnovers, {fouls} fouls. "
        f"+/-: {plus_minus}."
    )

# Download from S3 if CSV doesn't exist locally
if not os.path.exists(csv_path):
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION')
    )

    print("CSV not found locally, downloading from S3...")
    s3.download_file(os.getenv('S3_BUCKET_NAME'), f"box_scores_{date_str}.csv", csv_path)
    print("Downloaded successfully")
else:
    print("Using local CSV")

# Load the CSV
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows")
print(df.columns.tolist())


df['text'] = df.apply(row_to_text, axis=1)
df['vector_id'] = df['gameId'].astype(str) + "_" + df['personId'].astype(str)

# Fetch existing IDs from Pinecone
existing_ids = set()
stats = index.describe_index_stats()
if stats['total_vector_count'] > 0:
    
    all_ids = df['vector_id'].tolist()
    for i in range(0, len(all_ids), FETCH_BATCH_SIZE):
        batch_ids = all_ids[i:i+FETCH_BATCH_SIZE]
        fetch_result = index.fetch(ids=batch_ids)
        existing_ids.update(fetch_result['vectors'].keys())
    
    new_rows = df[~df['vector_id'].isin(existing_ids)]
    print(f"Found {len(existing_ids)} existing vectors, {len(new_rows)} new rows to embed")
else:
    new_rows = df
    print(f"Index is empty, embedding all {len(new_rows)} rows")

# Upsert to Pinecone
if len(new_rows) > 0:
    print("Embedding and uploading to Pinecone...")
    for i in range(0, len(df), UPSERT_BATCH_SIZE):
        batch = new_rows.iloc[i:i+UPSERT_BATCH_SIZE]
        
        try:
            embeddings = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=batch['text'].tolist(),
                parameters={"input_type": "passage"}
            )
            
            embeddings = [e['values'] for e in embeddings]
            
            vectors = [
                {
                    "id": str(row['vector_id']),
                    "values": embeddings[j],
                    "metadata": {
                        "text": row['text'],
                        "firstName": str(row['firstName']),
                        "lastName": str(row['lastName']),
                        "teamName": str(row['teamName']),
                        "game_date": str(row['game_date']),
                        "opponent": str(row['opponent'])
                    }
                }
                for j, (_, row) in enumerate(batch.iterrows())
            ]

            index.upsert(vectors=vectors)
            print(f"Uploaded batch {i} to {min(i+UPSERT_BATCH_SIZE, len(df))}")
        except Exception as e:
            print(f"Batch {i} failed: {e}, skipping...")
            continue

    print(f"Done! {len(new_rows)} chunks embedded and stored in Pinecone.")
else:
    print(f"No new rows to upsert.")