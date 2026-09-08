import json
import yfinance as yf
from duckduckgo_search import DDGS
from langchain_core.tools import tool

@tool
def get_stock_price_and_fundamentals(ticker: str) -> str:
    """Fetches current price, P/E ratio, market cap, and 52-week high/low range for a given stock ticker.
    Supports US tickers (e.g., 'AAPL', 'MSFT') and Indian NSE tickers (e.g., 'RELIANCE', 'TCS')."""
    try:
        clean_ticker = ticker.strip().upper()
        stock = yf.Ticker(clean_ticker)
        info = stock.info

        # Fallback to .NS for Indian tickers if base ticker returned no pricing data
        if (not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None)) and "." not in clean_ticker:
            fallback_stock = yf.Ticker(f"{clean_ticker}.NS")
            fallback_info = fallback_stock.info
            if fallback_info and (fallback_info.get("regularMarketPrice") is not None or fallback_info.get("currentPrice") is not None):
                stock = fallback_stock
                info = fallback_info

        # Safely handle None values returned for unprofitable or OTC tickers
        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or "N/A"
        pe_ratio = info.get("trailingPE") or "N/A"
        market_cap = info.get("marketCap") or "N/A"
        week_high = info.get("fiftyTwoWeekHigh") or "N/A"
        week_low = info.get("fiftyTwoWeekLow") or "N/A"
        currency = info.get("currency") or ""

        return (
            f"Ticker: {stock.ticker}\n"
            f"Current Price: {current_price} {currency}\n"
            f"Trailing P/E: {pe_ratio}\n"
            f"Market Cap: {market_cap}\n"
            f"52-Week High: {week_high}\n"
            f"52-Week Low: {week_low}"
        )
    except Exception as e:
        return f"Error retrieving fundamentals for {ticker}: {str(e)}"

@tool
def get_financial_statements(ticker: str) -> str:
    """Returns a parsed summary of the latest annual Income Statement, Balance Sheet, and Cash Flow (Owner Earnings) via yfinance.
    Supports US tickers and Indian NSE tickers."""
    try:
        clean_ticker = ticker.strip().upper()
        stock = yf.Ticker(clean_ticker)
        if (stock.financials is None or stock.financials.empty) and "." not in clean_ticker:
            fallback_stock = yf.Ticker(f"{clean_ticker}.NS")
            if fallback_stock.financials is not None and not fallback_stock.financials.empty:
                stock = fallback_stock

        # 1. Key Income Statement Line Items
        income_summary = {}
        if stock.financials is not None and not stock.financials.empty:
            income_col = stock.financials.iloc[:, 0].dropna()
            target_income_keys = [
                "Total Revenue", "Cost Of Revenue", "Gross Profit",
                "Operating Income", "Net Income", "Diluted EPS", "Basic EPS"
            ]
            for key in target_income_keys:
                if key in income_col:
                    income_summary[key] = float(income_col[key])

        # 2. Key Balance Sheet Line Items
        balance_summary = {}
        if stock.balance_sheet is not None and not stock.balance_sheet.empty:
            bs_col = stock.balance_sheet.iloc[:, 0].dropna()
            target_bs_keys = [
                "Total Assets", "Cash And Cash Equivalents",
                "Total Liabilities Net Minority Interest", "Total Debt",
                "Net Debt", "Stockholders Equity"
            ]
            for key in target_bs_keys:
                if key in bs_col:
                    balance_summary[key] = float(bs_col[key])

        # 3. Key Cash Flow Line Items (Buffett "Owner Earnings" & Free Cash Flow)
        cashflow_summary = {}
        if stock.cashflow is not None and not stock.cashflow.empty:
            cf_col = stock.cashflow.iloc[:, 0].dropna()
            target_cf_keys = [
                "Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"
            ]
            for key in target_cf_keys:
                if key in cf_col:
                    cashflow_summary[key] = float(cf_col[key])
            
            # Calculate Owner Earnings proxy if Free Cash Flow isn't explicitly listed
            if "Free Cash Flow" not in cashflow_summary and "Operating Cash Flow" in cashflow_summary and "Capital Expenditure" in cashflow_summary:
                cashflow_summary["Calculated Owner Earnings (FCF)"] = cashflow_summary["Operating Cash Flow"] - abs(cashflow_summary["Capital Expenditure"])

        result = {
            "ticker": stock.ticker,
            "income_statement_latest": income_summary if income_summary else "No income statement data available",
            "balance_sheet_latest": balance_summary if balance_summary else "No balance sheet data available",
            "cash_flow_and_owner_earnings": cashflow_summary if cashflow_summary else "No cash flow data available"
        }
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return f"Error retrieving financial statements for {ticker}: {str(e)}"

@tool
def search_market_news(query: str) -> str:
    """Uses DuckDuckGo to search for recent macroeconomic news, company updates, or earnings reports."""
    results = []
    try:
        with DDGS() as ddgs:
            news_results = list(ddgs.news(query, max_results=4))
            if news_results:
                results = news_results
    except Exception:
        pass

    if not results:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=4))
        except Exception as e:
            return f"Market search rate-limited or temporarily unavailable: {str(e)}"

    if not results:
        return "No recent news found for this query."
    return "\n\n".join([f"Title: {r['title']}\nSnippet: {r.get('body', r.get('snippet', ''))}" for r in results])

@tool
def analyze_portfolio(portfolio_json: str) -> str:
    """Accepts a JSON string of tickers and their percentage weights, e.g., '{\"AAPL\": 60, \"MSFT\": 40}'.
    Returns a text summary of sector diversification and basic risk."""
    try:
        # Gracefully handle string or direct dictionary inputs from LLM tool callers
        if isinstance(portfolio_json, dict):
            data = portfolio_json
        elif isinstance(portfolio_json, str):
            # Clean possible escaped quotes or formatting artifacts
            cleaned = portfolio_json.strip().replace("'", '"')
            data = json.loads(cleaned)
        else:
            return "Error: Input must be a JSON string mapping tickers to percentage weights."

        if not isinstance(data, dict):
            return "Error: Portfolio must be a dictionary of ticker symbols to allocation percentages."

        total_weight = sum(float(w) for w in data.values())
        breakdown = [f"{ticker.upper()}: {weight}%" for ticker, weight in data.items()]
        
        status = "Balanced" if abs(total_weight - 100.0) < 0.5 else f"Warning: Total weight is {total_weight}% (should sum to 100%)"
        
        # Sector analysis
        sectors = {}
        concentrated = []
        for ticker, weight in data.items():
            alloc = float(weight)
            if alloc > 30.0:
                concentrated.append(ticker.upper())
            try:
                t = ticker.strip().upper()
                s = yf.Ticker(t)
                sec = s.info.get("sector")
                if not sec and "." not in t:
                    s = yf.Ticker(f"{t}.NS")
                    sec = s.info.get("sector")
                sector_name = sec if sec else "Unknown"
                sectors[sector_name] = sectors.get(sector_name, 0) + alloc
            except Exception:
                sectors["Unknown"] = sectors.get("Unknown", 0) + alloc

        sector_summary = "\n".join([f"  - {sec}: {round(w, 2)}%" for sec, w in sorted(sectors.items(), key=lambda x: x[1], reverse=True)])
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
