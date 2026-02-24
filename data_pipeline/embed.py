import os
import chromadb
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTION_NAME = "nba_boxscores_6"
csv_file = os.path.join(BASE_DIR, "box_scores_2025_26.csv")
chroma_db_path = os.path.join(BASE_DIR, "chroma_db")
transformer_model = 'BAAI/bge-small-en-v1.5'

# Helper function to format date: '2026-02-23' -> 'Monday, February 23rd, 2026'.
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

# Load the CSV
df = pd.read_csv(csv_file)
print(f"Loaded {len(df)} rows")
print(df.columns.tolist())

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer(transformer_model)

def row_to_text(row):
    name = f"{row.get('firstName', '')} {row.get('lastName', row.get('familyName', ''))}".strip()
    minutes = row.get("minutes") or "0"

    points = row.get("points", 0)
    assists = row.get("assists", 0)

    steals = row.get("steals", 0)
    blocks = row.get("blocks", 0)

    turnovers = row.get("turnovers", 0)
    plus_minus = row.get("plusMinusPoints", 0)

    fouls = row.get("foulsPersonal", 0)

    fg_pct = row.get("fieldGoalsPercentage", 0)
    fg_made = row.get("fieldGoalsMade", 0)
    fg_att = row.get("fieldGoalsAttempted", 0)

    fg3_pct = row.get("threePointersPercentage", 0)
    fg3_made = row.get("threePointersMade", 0)
    fg3_att = row.get("threePointersAttempted", 0)

    ft_pct = row.get("freeThrowsPercentage", 0)
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

    ts_pct = row.get("trueShootingPercentage")
    ts_str = f", TS% {ts_pct * 100:.1f}%" if ts_pct is not None and not pd.isna(ts_pct) else ""

    print(f"{team_name} vs {opponent}")
    
    return (
        f"{name} played for {team_name} vs {opponent} on {game_date} (game {game_id}). "
        f"In {minutes} minutes he scored {points} points: "
        f"FG {fg_made}/{fg_att} ({fg_pct}%), 3PT {fg3_made}/{fg3_att} ({fg3_pct}%), FT {ft_made}/{ft_att} ({ft_pct}%), True Shooting: {ts_str}. "
        f"He had {total_rebounds} rebounds ({offensive_rebounds} offensive, {defensive_rebounds} defensive), "
        f"{assists} assists, {steals} steals, {blocks} blocks, {turnovers} turnovers, {fouls} fouls. "
        f"+/-: {plus_minus}."
    )

df['text'] = df.apply(row_to_text, axis=1)

print("Embedding chunks...")
embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)

print("Storing in Chroma...")
chroma = chromadb.PersistentClient(path=chroma_db_path)
collection = chroma.get_or_create_collection(COLLECTION_NAME)

collection.add(
    ids=df.index.astype(str).tolist(),
    documents=df['text'].tolist(),
    embeddings=embeddings.tolist(),
    metadatas=df[['firstName', 'lastName', 'teamName', 'gameId']].to_dict('records')
)

print(f"Done! {len(df)} chunks embedded and stored.")