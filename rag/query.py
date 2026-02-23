import sqlite3
import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Initialize everything
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "data_pipeline", "chroma_db")
SQLITE_DB_PATH = os.path.join(BASE_DIR, "data_pipeline", "nba.db")
VALID_TOOLS = {"query_sql_db", "query_vector_db"}

chroma = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma.get_or_create_collection("nba_boxscores_6")
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

def query_vector_db(query):
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )
    context = "\n".join(results['documents'][0])

    print(context)
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an NBA stats assistant. Answer using only the context provided. If the context does not explicitly mention the requested game or stats, say you don't know and do NOT invent stats or opponents."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content

def query_sql_db(sql):
    try:
        result = conn.execute(sql).fetchall()
        print(str(result))
        return str(result)
    except Exception as e:
        return f"SQL error: {e}"


def run_bot(question):
    schema = """
    Table: boxscores
    Columns:
    - gameId (text)
    - teamId (text)
    - teamName (text)
    - personId (text)
    - firstName (text)
    - lastName (text)
    - position (text)
    - minutes (text)
    - fieldGoalsMade (integer)
    - fieldGoalsAttempted (integer)
    - fieldGoalsPercentage (float)
    - threePointersMade (integer)
    - threePointersAttempted (integer)
    - threePointersPercentage (float)
    - freeThrowsMade (integer)
    - freeThrowsAttempted (integer)
    - freeThrowsPercentage (float)
    - reboundsOffensive (integer)
    - reboundsDefensive (integer)
    - reboundsTotal (integer)
    - assists (integer)
    - steals (integer)
    - blocks (integer)
    - turnovers (integer)
    - foulsPersonal (integer)
    - points (integer)
    - plusMinusPoints (float)
    - opponent (text)
    - opponent_id (text)
    - game_date (text)
    - season_type (text)
    - trueShootingPercentage (float)
    """

    # Step 1: Router decides which tool to use
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"""You are an NBA stats assistant. 
            Use 'query_sql_db' for any question that asks about stats over multiple games, uses phrases like 'this season', 'over his last X games', 'average', 'total', 'sum', 'per game', or compares multiple players or teams.”. 
            Use 'query_vector_db' for descriptive questions when the user is clearly asking about one specific game (e.g. 'vs the Rockets on February 5th' or 'in that game'). If the time frame is a season, multiple games, or not clearly a single game, do not use this tool..
            
            You must use only the two provided tools, with exact names:
            > - query_sql_db
            > - query_vector_db
            Do not invent or modify tool names

            Here is the SQL database schema:
            {schema}
            """},
            {"role": "user", "content": question}
        ],
        tools=tools,
        tool_choice="required"
    )
    
    tool_call = response.choices[0].message.tool_calls[0] # Extracts the first tool call from the response
    tool_name = tool_call.function.name # Gets the name of the tool the LLM picked
    args = json.loads(tool_call.function.arguments) # Get the arguments created by the LLM (SQL or Vector DB Query)
    
    if tool_name not in VALID_TOOLS:
        raise ValueError(f"Unknown tool requested by model: {tool_name}")

    # Run tool
    if tool_name.lower() == "query_sql_db":
        print(f"Routing to SQL: {args['sql']}")
        raw_result = query_sql_db(args['sql'])
    elif tool_name.lower() == "query_vector_db":
        print(f"Routing to Vector DB: {args['query']}")
        raw_result = query_vector_db(args['query'])
    
    # Generate natural language answer
    final_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an NBA stats assistant. Answer the question naturally using the data provided."},
            {"role": "user", "content": f"Question: {question}\nData: {raw_result}"}
        ]
    )
    
    return final_response.choices[0].message.content

if __name__ == "__main__":
    #print(run_bot("How many points did Bam Adebayo score vs the Grizzlies?"))
    #print(run_bot("What was KAT's average TS%% this season?"))
    #print(run_bot("What is LeBron's FG% this season?")) GOOD
    #print(run_bot("What is Lamelo's TS%% vs the Rockets on 2026-02-05?"))
    #print(run_bot("What is Lamelo's TS% this season vs the Rockets?"))
    print(run_bot("What was LaMelo's TS%% in his first game against the Rockets?"))