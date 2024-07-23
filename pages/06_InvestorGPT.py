import os
from typing import Type

import requests
import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.tools import BaseTool
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from pydantic import BaseModel, Field

from src.utils import load_file

markdown_file = load_file("./markdowns/investor_gpt.md")
st.set_page_config(
    page_title="InvestorGPT",
    page_icon="📈",
)
st.markdown(markdown_file)


llm = ChatOpenAI(
    temperature=0.1,
    model_name="gpt-3.5-turbo-1106",
)
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")


# def add(text):
#     # Use a regular expression to find all numbers in the text
#     # This pattern matches integers and decimal numbers
#     numbers = re.findall(r"-?\d+\.?\d*", text)

#     # Convert the extracted strings to floats and sum them
#     total = sum(float(number) for number in numbers)

#     return total


class StockMarketSymbolSearchToolSchema(BaseModel):
    query: str = Field(description="The query string to search for")


class StockMarketSymbolSearchTool(BaseTool):
    name = "StockMarketSymbolSearchTool"
    description = """
    Use this tool to find the stock market symbol for a given company.
    It takes a query string as an argument.
    Example query: Stock market Symbol for Apple Company
    """
    args_schema: Type[StockMarketSymbolSearchToolSchema] = (
        StockMarketSymbolSearchToolSchema
    )

    def _run(self, query):
        search = DuckDuckGoSearchAPIWrapper()
        return search.run(query)


class CompanyOverviewSchema(BaseModel):
    symbol: str = Field(
        description="Stock symbol of the company. Example: AAPL",
    )


class CompanyOverviewTool(BaseTool):
    name = "CompanyOverview"
    description = """
    Use this tool to get an overview of the financials of the company. You should enter stock symbol.
    """
    args_schema: Type[CompanyOverviewSchema] = CompanyOverviewSchema

    def _run(self, symbol):
        req = requests.get(
            f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        )
        return req.json()


class CompanyIncomeStatementTool(BaseTool):
    name = "CompanyIncomeStatement"
    description = """
    Use this tool to get the income statement of a company. You should enter stock symbol.
    """
    args_schema: Type[CompanyOverviewSchema] = CompanyOverviewSchema

    def _run(self, symbol):
        req = requests.get(
            f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        )
        return req.json()["annualReports"]


class CompanyStockPerformanceTool(BaseTool):
    name = "CompanyStockPerformance"
    description = """
    Use this tool to get the weekly stock performance of a company. You should enter stock symbol.
    """
    args_schema: Type[CompanyOverviewSchema] = CompanyOverviewSchema

    def _run(self, symbol):
        req = requests.get(
            f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        )
        weekly_time_series = req.json()["Weekly Time Series"]
        # convert dictionary to a list of items and sort them by date in descending order to get the latest weeks first
        sorted_weeks = sorted(
            weekly_time_series.items(), key=lambda x: x[0], reverse=True
        )
        # return the last 156 weeks (equivalent to 3 years)
        return sorted_weeks[:156]


agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,  # try to recover when parsing errors occur
    tools=[
        StockMarketSymbolSearchTool(),
        CompanyOverviewTool(),
        CompanyIncomeStatementTool(),
        CompanyStockPerformanceTool(),
    ],
    agent_kwargs={
        "system_message": SystemMessage(
            content="""
            You are an experienced hedge fund manager analyzing stocks for potential investment opportunities.

            Your task is to conduct a thorough evaluation of a company's financial health and market performance to determine if it is worth investing in.
            """
        ),
    },
)

company = st.text_input("Write the name of the public company you are interested in.")

if company:
    result = agent.invoke(company)
    st.write(result["output"])
