# https://python.langchain.com/docs/integrations/tools/sql_database/
import sqlite3
import os
from dotenv import load_dotenv
import requests
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
import os

def get_engine_for_chinook_db():
    """Pull sql file, populate in-memory database, and create engine."""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

engine = get_engine_for_chinook_db()

db = SQLDatabase(engine)
print("Available tables:", db.get_table_names())

# Load environment variables from .env file
load_dotenv()

from langchain_openai import ChatOpenAI
from agent_utils import create_chat_openai_from_env

llm = create_chat_openai_from_env(default_model="gpt-5")

# Use the SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
toolkit.get_tools()

# -----
from langchain import hub

prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")

assert len(prompt_template.messages) == 1
print("Input variables:", prompt_template.input_variables)
system_message = prompt_template.format(dialect="SQLite", top_k=5)

from langgraph.prebuilt import create_react_agent
from agent_utils import create_react_agent_compat

agent_executor = create_react_agent_compat(llm, toolkit.get_tools(), system_message)

# Loop to continuously ask for user queries
# example_query = "Which country's customers spent the most?"
while True:
    user_input = input("Enter your query (or 'q' to quit): ")
    if user_input.strip().lower() == 'q':
        print("Exiting...")
        break

    # Run the agent_executor on user_input
    events = agent_executor.stream(
        {"messages": [("user", user_input)]},
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()
