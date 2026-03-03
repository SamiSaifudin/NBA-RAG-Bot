import os
import json
import sqlite3
import asyncio
import chromadb
from groq import AsyncGroq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

CURRENT_SEASON = "2025-2026"
COLLECTION_NAME = "nba_boxscores"
VALID_TOOLS = {"query_sql_db", "query_vector_db"}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "data_pipeline", "chroma_db")
SQLITE_DB_PATH = os.path.join(BASE_DIR, "data_pipeline", "nba.db")

chroma = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma.get_or_create_collection(COLLECTION_NAME)
conn = sqlite3.connect(SQLITE_DB_PATH)

# Tools Definition
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_vector_db",
            "description": "Use this for descriptive questions about a specific game or player performance in a single game.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_sql_db",
            "description": "Use this for aggregation questions like averages, totals, or comparisons across multiple games.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"}
                },
                "required": ["sql"]
            }
        }
    }
]

async def query_vector_db(query: str) -> str:
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )
    context = "\n".join(results['documents'][0])
    
    response = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an NBA stats assistant. Answer using only the context provided. If the context does not explicitly mention the requested game or stats, say you don't know and do NOT invent stats or opponents."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content

def query_sql_db(sql: str) -> str:
    try:
        sql = sql.replace("\\'", "''")
        print(f"Executing SQL: {sql}")

        cursor = conn.execute(sql)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        
        result = [dict(zip(columns, row)) for row in rows]
        print(f"SQL Result: {result}")
        return str(result)
    except Exception as e:
        print(f"SQL Error: {e}")
        return f"SQL error: {e}"


async def run_bot(question: str, history: list[dict]) -> str:
    schema = """
    Table: boxscores
    Columns:
    - gameId (text), Example: 0022500825
    - teamId (text), Example: 1610612741
    - teamName (text), Example: Bulls
    - personId (text), Example: 1630171
    - firstName (text), Example: Isaac
    - lastName (text), Example: Okoro
    - position (text), Example: F
    - minutes (text), Example: 33:00
    - fieldGoalsMade (integer), Example: 4
    - fieldGoalsAttempted (integer), Example: 10
    - fieldGoalsPercentage (float), Example: 0.4
    - threePointersMade (integer), Example: 3
    - threePointersAttempted (integer), Example: 7
    - threePointersPercentage (float), Example: 0.429
    - freeThrowsMade (integer), Example: 1
    - freeThrowsAttempted (integer), Example: 1
    - freeThrowsPercentage (float), Example: 1.0
    - reboundsOffensive (integer), Example: 3
    - reboundsDefensive (integer), Example: 3
    - reboundsTotal (integer), Example: 6
    - assists (integer), Example: 1
    - steals (integer), Example: 1
    - blocks (integer), Example: 0
    - turnovers (integer), Example: 2
    - foulsPersonal (integer), Example: 3
    - points (integer), Example: 12
    - plusMinusPoints (float), Example: -15.0
    - opponent (text), Example: Knicks
    - opponent_id (text), Example: 1610612752
    - game_date (text), Example: 2026-02-22
    - season_type (text), Example: Regular Season
    - trueShootingPercentage (float), Example: 0.574712643678161
    """

    # Router decides which tool to use
    response = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"""You are an NBA stats assistant for the {CURRENT_SEASON} season.

                You must always call exactly one of these two tools:

                1. query_sql_db: 
                Use for any question involving:
                - Averages, totals, counts, or rankings
                - Phrases like 'this season', 'average', 'total', 'per game'
                - Stats across multiple games
                - Comparing multiple players or teams

                2. query_vector_db: 
                - Use ONLY when the user is asking about one specific game AND a specific player with a clear date or opponent mentioned (e.g. 'vs the Rockets on February 5th').

                SQL Rules:
                - Always SELECT all relevant columns needed to fully answer the question, never just SELECT a single column
                - For player performance questions always include: game_date, opponent, points, assists, reboundsTotal, fieldGoalsPercentage, threePointersPercentage, freeThrowsPercentage, trueShootingPercentage, plusMinusPoints
                - Percentages are stored as floats (e.g. 0.55 = 55%)

                Rules:
                - Tool names must not contain any whitespace, tabs, or special characters
                - Use exact tool names: query_sql_db and query_vector_db
                - Always call a tool, never respond directly
                - When in doubt, use query_sql_db

                If the user asks a follow up question, use the conversation history to understand what they are referring to before deciding which tool to use.

                IMPORTANT: Use the tool name EXACTLY as written above. No parentheses, no equals signs, no extra characters.

                Example Vector DB Entry:
                Victor Wembanyama played for Spurs vs Kings on Sunday, February 21st, 2026 (game 0022500815). 
                In 29:45 minutes he scored 28 points: FG 11/20 (55.0%), 3PT 1/5 (20.0%), FT 5/7 (71.4%), True Shooting: 60.7%. 
                He had 15 rebounds (1 offensive, 14 defensive), 6 assists, 1 steal, 4 blocks, 1 turnover, 3 fouls. 
                +/-: 32.0.

                Database schema:
                {schema}
            """},
            *history,
            {"role": "user", "content": question}
        ],
        tools=tools,
        tool_choice="required",
        parallel_tool_calls=False
    )
    
    tool_call = response.choices[0].message.tool_calls[0]
    tool_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    
    if tool_name not in VALID_TOOLS:
        raise ValueError(f"Unknown tool requested by model: {tool_name}")

    if tool_name.lower() == "query_sql_db":
        print(f"Routing to SQL: {args['sql']}")
        raw_result = query_sql_db(args['sql'])
    elif tool_name.lower() == "query_vector_db":
        print(f"Routing to Vector DB: {args['query']}")
        raw_result = await query_vector_db(args['query'])
    
    final_response = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an NBA stats assistant. Answer the question naturally using the data provided. If the question is a follow up, use the conversation history for context."},
            *history,
            {"role": "user", "content": f"Question: {question}\nData: {raw_result}"}
        ]
    )
    
    return final_response.choices[0].message.content

async def main():
    test_questions = [
        "How many points did Bam Adebayo score vs the Grizzlies on February 21st, 2026?",
        "What were LaMelo Ball's TS%% vs the Rockets last Thursday?",
        "What are LaMelo Ball's total points vs the Rockets this season?",
        "Which player scored the most total points this season?",
    ]
    
    for q in test_questions:
        print(f"Q: {q}")
        answer = await run_bot(q)
        print(f"A: {answer}")
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(main())