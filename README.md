🚍 Akton Predictive Maintenance – LLM Bus Analytics Agent

Akton Agent is a full end-to-end predictive maintenance system built as a final project, combining LLM reasoning, SQL planning, data modeling, and real-time analytics for bus fleet management.
It provides deep insights, failure predictions, part-replacement analysis, and natural-language interaction in both Hebrew and English.

🔥 Key Features
🧠 LLM-Driven Query Agent

Understands Hebrew and English

Handles typos, slang, spacing errors, and broken phrasing

Normalizes text and extracts intent

Generates safe, schema-restricted SQL queries dynamically

Chooses between predefined dashboards and custom SQL

🔧 Predictive Maintenance Insights

7-day and 30-day failure predictions

Maintenance-flag analysis

Most-replaced parts detection

Failure types and likely fault causes

Risk scoring per bus / per day

🗂 Strict SQL Pipeline

The agent uses a highly controlled SQL generation layer:

Mandatory join sequence

No unauthorized tables or columns

Special rules for part-replacement queries

Full date-range support

Complete defense against harmful SQL

🎛 Streamlit Dashboard

Interactive visual dashboards

Risk overview

Bus-level drill-down

Part replacement reports

Time-range selection

Integrated chat agent with live DB querying

📁 Project Structure
akton_agent/
│
├── app_streamlit.py        # Streamlit UI + chat interface
├── agent_queries.py        # Predefined SQL dashboards
├── intents.py              # NLP intent detection (Hebrew + English)
├── failure_mapping.py      # Fault-to-part mapping logic
├── schema_meta.py          # Database schema definition
├── db.py                   # PostgreSQL connection & execution
├── utils_logging.py        # Logging utilities
└── config.py               # Environment + settings

🧰 Technology Stack

Language: Python

Framework: Streamlit

AI: OpenAI LLM (SQL Planner + NLU)

Database: PostgreSQL

Visualization: Streamlit Charts

Logic: SQL generation, schema validation, intent parsing

🎯 Project Goals

Demonstrate real-world predictive maintenance with AI

Combine Data Engineering, LLMs, ML predictions and BI dashboards

Build a safe, reliable, fully functional chat-based analytics agent

Provide a simple interface for fleet managers to access advanced insights

🧑‍💻 Author

Elishay Amar
Data Analyst & Full-Stack Developer
Final Project — CyberPro Data Analyst Program

LinkedIn: (add your link)

📄 License

This project is provided for educational and demonstration purposes.
