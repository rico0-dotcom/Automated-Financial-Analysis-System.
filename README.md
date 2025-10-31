AI Financial Analysis System:

### Test Case
Tested on Walmart’s annual report (10-K, includes 3 years of financials).

- Notebook with full run output: `Financial_analytic_Ai.ipynb`
- Parsing algorithm: `financial_parsing_horizontal.py`
- Ratio engine: `ratio.py`


End-to-end pipeline for parsing raw financial statements, standardizing line items, computing ratios, and generating analyst-style insights.

Overview

This system automates key steps in financial analysis:

Ingest Excel/CSV financial statements

Parse and normalize financial items

Map to standardized taxonomy using hybrid matching

Support Human-in-the-Loop (HITL) corrections

Compute ratios and trends

Generate analyst-style summaries using LLMs

Export dashboards and insights

The goal is to eliminate manual report processing and enable scalable financial analytics.

Key Features
🔎 Data Ingestion & Parsing

Upload raw statements in Excel/CSV

Extract Income Statement, Balance Sheet, Cash Flow sections

🧠 Hybrid Financial Term Mapping

Three-layer matching engine:

Regex pattern matching

Fuzzy similarity

Transformer embedding similarity

Unmatched terms routed to reviewer.

✅ Human-in-the-Loop Learning

Terms below confidence threshold trigger manual input

Accepted corrections update mapping rules for future runs

📊 Ratio & Trend Engine

Core financial ratios

Custom ratio builder

Multi-period trend evaluation

📝 Automated Insights

LLM-generated commentary on ratios, trends, and risks

Analyst-style interpretation for reports

🚀 Deployment-Ready

Streamlit frontend

Modular backend

Scales to multi-GB data

Tech Stack:
Core->	Python, Pandas, NumPy
NLP	->spaCy, regex, fuzzy matching
LLM	->Azure OpenAI.
Data->	Excel, CSV
