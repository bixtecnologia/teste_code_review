# https://python.langchain.com/docs/integrations/tools/sql_database/
import sqlite3
import os
from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from langchain import hub

db = SQLDatabase.from_uri("sqlite:///example.db")
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
prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")

assert len(prompt_template.messages) == 1
print("Input variables:", prompt_template.input_variables)
system_message = prompt_template.format(dialect="SQLite", top_k=5)

# Domain-specific guidance (can be overridden via env SQL_AGENT_HINTS)
domain_hint = os.getenv(
    "SQL_AGENT_HINTS",
    "When referencing prices, use the table 'product_itens'. For product descriptions, use the table 'products'.",
)
if domain_hint:
    system_message = f"{system_message}\n\nDomain guidance: {domain_hint}"

from langgraph.prebuilt import create_react_agent
from agent_utils import create_react_agent_compat

agent_executor = create_react_agent_compat(llm, toolkit.get_tools(), system_message)

# Loop to continuously ask for user queries
# example_query = "tell me quote which is pending" and "tell me pending quote which was created in feb"
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
