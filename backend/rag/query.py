import os
import json
import asyncio
from groq import AsyncGroq
from openai import AsyncOpenAI
from pinecone import Pinecone
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

RETRY_LIMIT = 3
CURRENT_SEASON = "2025-2026"
VALID_TOOLS = {"query_sql_db", "query_vector_db"}

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index('clutchquery')
engine = create_engine(os.getenv('DATABASE_URL'))

today = datetime.now()
current_date = today.strftime("%A, %B %d, %Y")

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
                    "query": {"type": "string"},
                    "game_date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format e.g. 2026-03-15"
                    },
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"}
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

async def query_vector_db(query: str, game_date: str = None, first_name: str = None, last_name: str = None) -> str:
    filter = {}
    if game_date: filter["game_date"] = game_date
    if first_name: filter["firstName"] = first_name
    if last_name: filter["lastName"] = last_name

    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={"input_type": "query"}
    )
    query_embedding = embedding[0]['values']

    results = index.query(
        vector=query_embedding,
        top_k=10,
        include_metadata=True,
        filter=filter if filter else None
    )

    if not results['matches']:
        print("No results with filter, retrying without...")
        results = index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True
        )

    context = "\n".join([match['metadata']['text'] for match in results['matches']])

    print(context)
    
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an NBA stats assistant. Answer using only the context provided. If the context does not explicitly mention the requested game or stats, say you don't know and do NOT invent stats or opponents."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nFilter: {filter}"}
        ]
    )
    return response.choices[0].message.content

def query_sql_db(sql: str) -> str:
    try:
        sql = sql.replace("\\'", "''")
        print(f"Executing SQL: {sql}")

        with engine.connect() as conn:
            cursor = conn.execute(text(sql))
            columns = list(cursor.keys())
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

    for attempt in range(RETRY_LIMIT):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"""You are an NBA stats assistant for the {CURRENT_SEASON} season which started in October 2025. Today's date is {current_date}.

                        You must always call exactly one of these two tools:

                        1. query_sql_db: 
                        Use for any question involving:
                        - Averages, totals, counts, or rankings
                        - Phrases like 'this season', 'average', 'total', 'per game'
                        - Stats across multiple games
                        - Comparing multiple players or teams
                        - Always compute shooting percentages using totals, not averages of percentages. For example, NEVER DO: SELECT AVG("freeThrowsPercentage"). Instead do: SELECT 
                        100.0 * SUM(ft_made) / NULLIF(SUM(ft_attempted), 0) AS ft_percentage

                        2. query_vector_db: 
                        - Use ONLY when the user is asking about one specific game AND a specific player with a clear date or opponent mentioned (e.g. 'vs the Rockets on February 5th').

                        SQL Rules:
                        - Always SELECT all relevant columns needed to fully answer the question, never just SELECT a single column
                        - For player performance questions always include: game_date, opponent, points, assists, reboundsTotal, fieldGoalsPercentage, threePointersPercentage, freeThrowsPercentage, trueShootingPercentage, plusMinusPoints
                        - Percentages are stored as floats (e.g. 0.55 = 55%)

                        Vector DB Rules:
                        - Convert relative date references to ACTUAL dates in query. 
                            * "yesterday" → the actual date
                            * "last Tuesday" → the actual date
                            * "X days ago" → the actual date
                            * "on Wednesday" → the actual date
                        - Query should be descriptive (TRY TO MATCH FORMAT OF ACTUAL VECTOR DB ENTRY)
                        - Try to include date and opponent if they are given

                        General Rules:
                        - Tool names must not contain any whitespace, tabs, or special characters
                        - Use exact tool names: query_sql_db and query_vector_db
                        - Always call a tool, never respond directly
                        - When in doubt, use query_sql_db
                        - Never use relative terms like 'yesterday' or '2 days ago' in your queries.
                        - ALWAYS USE FULL DATES, i.e., February 21st, 2026
                        - True Shooting (TS)% Formula: PTS / (2 * (FGA + 0.44 * FTA))
                        - Convert known nickname's to the player's real name. 
                        - DO NOT GUESS OPONENTS. 

                        If the user asks a follow up question, use the conversation history to understand what they are referring to before deciding which tool to use.

                        IMPORTANT: Use the tool name EXACTLY as written above. No parentheses, no equals signs, no extra characters.

                        Example Vector DB Entry (ALL VECTOR DB ROWS LOOK LIKE THIS):
                        Victor Wembanyama played for Spurs vs Kings on Sunday, February 21st, 2026 (game 0022500815). 
                        In 29:45 minutes he scored 28 points: FG 11/20 (55.0%), 3PT 1/5 (20.0%), FT 5/7 (71.4%), True Shooting: 60.7%. 
                        He had 15 rebounds (1 offensive, 14 defensive), 6 assists, 1 steal, 4 blocks, 1 turnover, 3 fouls. 
                        +/-: 32.0.

                        Database schema:
                        {schema}

                        Always wrap column names in double quotes since Postgres is case sensitive (e.g. "firstName", "teamName")
                    """},
                    *history,
                    {"role": "user", "content": question}
                ],
                tools=tools,
                tool_choice="required",
                parallel_tool_calls=False
            )
            break
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt == 2:
                return "I had trouble processing that question. Please try rephrasing it."
    
    tool_call = response.choices[0].message.tool_calls[0]
    tool_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    
    if tool_name not in VALID_TOOLS:
        raise ValueError(f"Unknown tool requested by model: {tool_name}")

    if tool_name.lower() == "query_sql_db":
        print(f"Routing to SQL: {args['sql']}")
        raw_result = query_sql_db(args['sql'])
    elif tool_name.lower() == "query_vector_db":
        print(f"Routing to Vector DB: {args['query']}, First Name: {args.get('first_name', None)}, Last Name: {args.get('last_name', None)}, Game Date: {args.get('game_date', None)}")
        raw_result = await query_vector_db(args['query'], args.get('game_date', None), args.get('first_name', None), args.get('last_name', None))
    
    final_response = await groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"You are an NBA stats assistant for the {CURRENT_SEASON} season. Today's date is {current_date}. Answer the question naturally using the data provided. If the question is a follow up, use the conversation history for context. BE CONCISE!"},
            *history,
            {"role": "user", "content": f"Question: {question}\nData: {raw_result}"}
        ]
    )

    return final_response.choices[0].message.content

async def main():
    test_questions = [
        "How many points did Bam Adebayo score vs the Wizards on March 10th, 2026?",
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