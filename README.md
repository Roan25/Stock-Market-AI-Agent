# 📈 Value Investing Stock Market AI Agent

A production-ready, local multi-tool Stock Market AI Agent built with **Python**, **LangChain**, **Ollama**, and **Streamlit**. The agent operates under a strict **Warren Buffett** value-investing persona, analyzing companies based on intrinsic value, economic moats, and margins of safety rather than short-term market noise.

---

## 🚀 Key Features

- **100% Local Inference**: Powered by local open-weight LLMs via Ollama (`llama3.1`, `qwen2.5:7b`, `mistral`) running on your GPU. Zero external API keys needed for inference; total privacy for your portfolio data.
- **Custom LangChain Financial Tools**:
  - `get_stock_price_and_fundamentals`: Fetches real-time price, trailing P/E ratio, market cap, and 52-week high/low using `yfinance`. Auto-resolves Indian NSE tickers (e.g., `RELIANCE`, `TCS` $\rightarrow$ `.NS`).
  - `get_financial_statements`: Extracts and parses the latest annual income statement and balance sheet.
  - `search_market_news`: Searches real-time macroeconomic and earnings news using DuckDuckGo (`DDGS`).
  - `analyze_portfolio`: Analyzes user-provided JSON portfolios, calculating sector diversification and flagging concentration risks (>30% weight).
  - `query_nifty50_historical_dataset`: Integrates with the [Kaggle Rohan Rao NIFTY-50 dataset](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data) for long-term historical analysis (2000–2021).
- **Persistent Multi-Turn Memory**: Utilizes `ConversationBufferMemory` to maintain full conversational context across interactions.
- **Real-Time Tool Thought Streaming**: Uses `StreamlitCallbackHandler` to render expandable real-time tool execution logs and thought processes.

---

## 📁 Repository Structure

```
Stock-Market-AI-Agent/
├── .gitignore              # Standard Python gitignore
├── README.md               # Documentation & setup guide
├── requirements.txt        # Core project dependencies
├── tools.py                # 5 Custom LangChain tools
├── agent.py                # ChatOllama Agent & Buffett persona
└── app.py                  # Streamlit chat interface
```

---

## 🛠️ Quickstart Guide

### 1. Prerequisites
- **Python 3.10+**
- **Ollama**: Download and install from [ollama.com](https://ollama.com/).
- Pull your preferred model:
  ```bash
  ollama pull llama3.1
  ```

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/Roan25/Stock-Market-AI-Agent.git
cd Stock-Market-AI-Agent
pip install -r requirements.txt
```

### 3. Run the Application
Launch the Streamlit web application:
```bash
streamlit run app.py
```

---

## 💡 Example Queries

- *"Analyze Apple (AAPL) fundamentals and recent earnings news from a value-investing perspective."*
- *"Give me a breakdown of Reliance Industries based on its balance sheet and P/E ratio."*
- *"Analyze my portfolio: `{\"AAPL\": 45, \"MSFT\": 35, \"GOOGL\": 20}`"*
- *"What is the economic moat and intrinsic value outlook for Nvidia (NVDA)?"*
