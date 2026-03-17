import os
import time
import boto3
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
from nba_api.stats.endpoints import boxscoretraditionalv3, leaguegamefinder

load_dotenv()

SEASON = "2025-26"
RATE_LIMIT_DELAY = 3
TESTING_LIMIT = 10
SEASON_TYPES = ("Regular Season", "Playoffs", "PlayIn")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

date_str = datetime.now().strftime("%Y_%m_%d")
yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y_%m_%d")

csv_path = os.path.join(BASE_DIR, "box_scores", f"box_scores_{yesterday_str}.csv")

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

if not os.path.exists(csv_path):
    print("CSV not found locally, downloading from S3...")
    s3.download_file(os.getenv('S3_BUCKET_NAME'), f"box_scores_{yesterday_str}.csv", csv_path)
    print("Downloaded successfully")
else:
    print("Using local CSV")

headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Referer': 'https://www.nba.com/',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
        'Origin': 'https://www.nba.com',
        'Connection': 'keep-alive',
}

# Get all game IDs and dates for yesterday's NBA games (Regular Season, Playoffs, PlayIn)
def get_all_game_ids_and_dates(season: str) -> tuple[list[str], dict[str, str]]:
    all_game_ids = []
    game_date_map = {}

    yesterday = (datetime.now() - timedelta(days=1)).strftime('%m/%d/%Y')

    for season_type in SEASON_TYPES:
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable="00",
            season_type_nullable=season_type,
            timeout=30,
            date_from_nullable=yesterday,
            date_to_nullable=yesterday,
            headers=headers
        )
        games_df = gamefinder.get_data_frames()[0]

        if games_df is not None and not games_df.empty:
            game_info = games_df[["GAME_ID", "GAME_DATE"]].drop_duplicates()
            game_ids = game_info["GAME_ID"].tolist()
            all_game_ids.extend(game_ids)
            
            for _, row in game_info.iterrows():
                game_date_map[row["GAME_ID"]] = (row["GAME_DATE"], season_type)
            
            print(f"  {season_type}: {len(game_ids)} games")

        time.sleep(RATE_LIMIT_DELAY)

    unique_game_ids = list(dict.fromkeys(all_game_ids))
    return unique_game_ids, game_date_map


# Fetch traditional box scores (player stats) for every game
def fetch_box_scores_for_season(season: str) -> pd.DataFrame:
    game_ids, game_date_map = get_all_game_ids_and_dates(season)
    print(f"Found {len(game_ids)} games for {season} season")

    all_player_stats = []
    failed_games = []

    for i, game_id in enumerate(game_ids, 1):
        try:
            boxscore = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, headers=headers)

            players_df = boxscore.player_stats.get_data_frame()
            if players_df is not None and not players_df.empty:

                drop_cols = ["nameI", "teamCity", "teamTricode", "teamSlug", "playerSlug", "comment", "jerseyNum"]
                players_df = players_df.drop(columns=[c for c in drop_cols if c in players_df.columns])
                players_df = players_df.rename(columns={"familyName": "lastName"})

                teams_in_game = players_df[["teamId", "teamName"]].drop_duplicates()
                team_a_id, team_a_name = teams_in_game.iloc[0]
                team_b_id, team_b_name = teams_in_game.iloc[1]

                players_df["opponent"] = players_df["teamId"].map( #Logic: If teamId is Team A’s id → opponent name is Team B’s name.
                    {team_a_id: team_b_name, team_b_id: team_a_name}
                )
                players_df["opponent_id"] = players_df["teamId"].map( #Logic: If teamId is Team A’s id → opponent id is Team B’s id.
                    {team_a_id: team_b_id, team_b_id: team_a_id}
                )
                
                game_date, season_type = game_date_map.get(game_id, None)

                players_df["game_date"] = game_date
                players_df["season_type"] = season_type

                # True Shooting %: PTS / (2 * (FGA + 0.44 * FTA)): When this player used a possession to shoot, how efficient were they?”
                ts_denom = 2 * (players_df["fieldGoalsAttempted"] + 0.44 * players_df["freeThrowsAttempted"])
                players_df["trueShootingPercentage"] = players_df["points"] / ts_denom.replace(0, float("nan"))

                all_player_stats.append(players_df)

            num_players = len(players_df) if players_df is not None else 0
            print(f"  [{i}/{len(game_ids)}] Game {game_id}: {num_players} players")
            
        except Exception as e:
            failed_games.append((game_id, str(e)))
            print(f"  [{i}/{len(game_ids)}] Game {game_id}: FAILED - {e}")
        time.sleep(RATE_LIMIT_DELAY)

    if failed_games:
        print(f"\nFailed to fetch {len(failed_games)} games: {failed_games}")

    if not all_player_stats:
        return pd.DataFrame()

    return pd.concat(all_player_stats, ignore_index=True)

# Upload the csv file to S3 bucket
def upload_to_s3(local_path):
    s3.upload_file(local_path, os.getenv('S3_BUCKET_NAME'), f"box_scores_{date_str}.csv")
    print(f"Uploaded to S3 successfully")

if __name__ == "__main__":
    box_scores = fetch_box_scores_for_season(SEASON)
    print(f"\nTotal rows: {len(box_scores)}")
    print(f"Columns: {list(box_scores.columns)}")
    print(box_scores.head(10))

    existing_df = pd.read_csv(csv_path)

    updated_df = pd.concat([existing_df, box_scores], ignore_index=True)

    output_path = os.path.join(BASE_DIR, "box_scores", f"box_scores_{date_str}.csv")
    updated_df.to_csv(output_path, index=False)

    print(f"\nSaved to {output_path}")
    upload_to_s3(output_path)
