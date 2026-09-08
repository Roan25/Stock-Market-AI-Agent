import json
import os
import yfinance as yf
import pandas as pd
from duckduckgo_search import DDGS
from langchain_core.tools import tool

@tool
def get_stock_price_and_fundamentals(ticker: str) -> str:
    """Fetches current price, P/E ratio, market cap, and 52-week high/low range for a given stock ticker.
    For Indian NSE stocks, automatically appends .NS if omitted (e.g., 'RELIANCE', 'TCS')."""
    try:
        clean_ticker = ticker.strip().upper()
        # If it looks like an Indian stock ticker without exchange suffix, attempt .NS first
        if not clean_ticker.endswith(".NS") and not clean_ticker.endswith(".BO") and len(clean_ticker) <= 15:
            # Try fetching with .NS first, fallback to original if not found
            stock = yf.Ticker(f"{clean_ticker}.NS")
            info = stock.info
            if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
                stock = yf.Ticker(clean_ticker)
                info = stock.info
        else:
            stock = yf.Ticker(clean_ticker)
            info = stock.info

        current_price = info.get("currentPrice") or info.get("regularMarketPrice", "N/A")
        pe_ratio = info.get("trailingPE", "N/A")
        market_cap = info.get("marketCap", "N/A")
        week_high = info.get("fiftyTwoWeekHigh", "N/A")
        week_low = info.get("fiftyTwoWeekLow", "N/A")
        currency = info.get("currency", "")

        return (
            f"Ticker: {stock.ticker}\n"
            f"Current Price: {current_price} {currency}\n"
            f"Trailing P/E: {pe_ratio}\n"
            f"Market Cap: {market_cap}\n"
            f"52-Week High: {week_high}\n"
            f"52-Week Low: {week_low}"
        )
    except Exception as e:
        return f"Error retrieving data for {ticker}: {str(e)}"

@tool
def get_financial_statements(ticker: str) -> str:
    """Returns a parsed summary of the latest income statement and balance sheet via yfinance.
    For Indian stocks, appends .NS if needed."""
    try:
        clean_ticker = ticker.strip().upper()
        if not clean_ticker.endswith(".NS") and not clean_ticker.endswith(".BO"):
            stock = yf.Ticker(f"{clean_ticker}.NS")
            if stock.financials is None or stock.financials.empty:
                stock = yf.Ticker(clean_ticker)
        else:
            stock = yf.Ticker(clean_ticker)

        # Income Statement
        income_stmt = stock.financials
        income_summary = {}
        if income_stmt is not None and not income_stmt.empty:
            recent_income = income_stmt.iloc[:, 0].dropna()
            income_summary = {k: recent_income[k] for k in list(recent_income.keys())[:8]}

        # Balance Sheet
        balance_sheet = stock.balance_sheet
        balance_summary = {}
        if balance_sheet is not None and not balance_sheet.empty:
            recent_balance = balance_sheet.iloc[:, 0].dropna()
            balance_summary = {k: recent_balance[k] for k in list(recent_balance.keys())[:8]}

        result = {
            "ticker": stock.ticker,
            "income_statement_recent": income_summary if income_summary else "No income statement data available",
            "balance_sheet_recent": balance_summary if balance_summary else "No balance sheet data available"
        }
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return f"Error retrieving financial statements for {ticker}: {str(e)}"

@tool
def search_market_news(query: str) -> str:
    """Uses DuckDuckGo to search for recent macroeconomic news or earnings reports."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No recent news found for this query."
        return "\n\n".join([f"Title: {r['title']}\nSnippet: {r['body']}" for r in results])
    except Exception as e:
        return f"Error searching web: {str(e)}"

@tool
def analyze_portfolio(portfolio_json: str) -> str:
    """Accepts a JSON string of tickers and their percentage weights, e.g., '{\"AAPL\": 60, \"MSFT\": 40}'.
    Returns a text summary of sector diversification and basic risk."""
    try:
        data = json.loads(portfolio_json)
        if not isinstance(data, dict):
            return "Error: Input must be a JSON dictionary mapping tickers to percentage weights."

        total_weight = sum(float(w) for w in data.values())
        breakdown = [f"{ticker.upper()}: {weight}%" for ticker, weight in data.items()]
        
        status = "Balanced" if abs(total_weight - 100.0) < 0.5 else f"Warning: Total weight is {total_weight}% (should sum to 100%)"
        
        # Sector lookups
        sectors = {}
        concentrated = []
        for ticker, weight in data.items():
            if float(weight) > 30.0:
                concentrated.append(ticker.upper())
            try:
                t = ticker.strip().upper()
                if not t.endswith(".NS") and not t.endswith(".BO"):
                    s = yf.Ticker(f"{t}.NS")
                    sec = s.info.get("sector")
                    if not sec:
                        s = yf.Ticker(t)
                        sec = s.info.get("sector", "Unknown")
                else:
                    s = yf.Ticker(t)
                    sec = s.info.get("sector", "Unknown")
                sectors[sec] = sectors.get(sec, 0) + float(weight)
            except Exception:
                sectors["Unknown"] = sectors.get("Unknown", 0) + float(weight)

        sector_summary = "\n".join([f"  - {sec}: {round(w, 2)}%" for sec, w in sectors.items()])
        risk_notes = []
        if concentrated:
            risk_notes.append(f"High concentration risk in {', '.join(concentrated)} (>30% allocation).")
        if len(sectors) < 3:
            risk_notes.append("Sector concentration risk: Holdings cover fewer than 3 sectors.")
        else:
            risk_notes.append("Adequate sector diversification.")

        return (
            f"Portfolio Holdings:\n" + "\n".join(breakdown) + "\n\n"
            f"Allocation Status: {status}\n\n"
            f"Sector Diversification:\n{sector_summary}\n\n"
            f"Risk Assessment:\n" + "\n".join(risk_notes)
        )
    except json.JSONDecodeError as e:
        return f"Invalid JSON format for portfolio. Details: {str(e)}"
    except Exception as e:
        return f"Error analyzing portfolio: {str(e)}"

@tool
def query_nifty50_historical_dataset(ticker: str) -> str:
    """Queries the local Kaggle NIFTY-50 (2000-2021) dataset by Rohan Rao for long-term historical trading records.
    Expects CSV files in './data/{TICKER}.csv' (e.g., RELIANCE.csv, TCS.csv)."""
    try:
        symbol = ticker.strip().upper().replace(".NS", "").replace(".BO", "")
        # Check potential local paths
        possible_paths = [
            f"./data/{symbol}.csv",
            f"./data/NIFTY50/{symbol}.csv",
            f"C:/Users/rohan/.gemini/antigravity/scratch/buffett_agent/data/{symbol}.csv"
        ]
        
        target_path = None
        for p in possible_paths:
            if os.path.exists(p):
                target_path = p
                break

        if not target_path:
            return (
                f"Kaggle NIFTY-50 archive file for '{symbol}.csv' not found locally in './data/'. "
                f"To query long-term 2000-2021 historical data, download the dataset from "
                f"https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data and place the CSV files in './data/'."
            )

        df = pd.read_csv(target_path)
        if df.empty:
            return f"Dataset for {symbol} is empty."

        # Compute summary metrics from Kaggle dataset
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        start_date = df["Date"].min().strftime("%Y-%m-%d")
        end_date = df["Date"].max().strftime("%Y-%m-%d")
        total_days = len(df)
        all_time_high = df["High"].max()
        all_time_low = df["Low"].min()
        avg_vwap = df["VWAP"].mean()
        latest_row = df.iloc[-1]

        return (
            f"Kaggle NIFTY-50 Historical Archive for {symbol} ({start_date} to {end_date}):\n"
            f"Total Trading Records: {total_days} sessions\n"
            f"Historical Low: {all_time_low}\n"
            f"Historical High: {all_time_high}\n"
            f"Historical Average VWAP: {round(avg_vwap, 2)}\n"
            f"Archive Final Date: {end_date}\n"
            f"Archive Final Close: {latest_row['Close']}\n"
            f"Archive Deliverable Volume %: {latest_row.get('%Deliverble', 'N/A')}\n"
        )
    except Exception as e:
        return f"Error reading Kaggle NIFTY-50 historical dataset for {ticker}: {str(e)}"
