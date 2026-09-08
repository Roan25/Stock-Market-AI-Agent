# 📈 Value Investing Stock Market AI Agent

A production-ready, local multi-tool Stock Market AI Agent built with **Python**, **LangChain**, **Ollama**, and **Streamlit**. The agent operates under a strict **Warren Buffett** value-investing persona, analyzing companies based on intrinsic value, economic moats, and margins of safety rather than short-term market noise.

---

## 🚀 Key Features

- **100% Local Inference**: Powered by local open-weight LLMs via Ollama (`llama3.1`, `qwen2.5:7b`, `mistral`) running on your GPU. Zero external API keys needed for inference; total privacy for your portfolio data.
- **The 4 Core LangChain Financial Tools**:
  - `get_stock_price_and_fundamentals`: Fetches real-time price, trailing P/E ratio, market cap, and 52-week high/low using `yfinance`. Auto-resolves Indian NSE tickers (e.g., `RELIANCE`, `TCS` $\rightarrow$ `.NS`).
  - `get_financial_statements`: Extracts and parses full Income Statements, Balance Sheets, and Cash Flow Statements (calculating Buffett's **Owner Earnings / Free Cash Flow**).
  - `search_market_news`: Searches real-time macroeconomic and earnings news using DuckDuckGo (`DDGS`).
  - `analyze_portfolio`: Analyzes user-provided JSON portfolios, calculating sector diversification and flagging concentration risks (>30% weight).
- **Synchronized Bounded Memory**: Utilizes `ConversationBufferWindowMemory(k=5)` with `num_ctx=8192` to prevent Ollama context window overflow, seamlessly re-hydrated whenever switching models in the UI.
- **Real-Time Tool Thought Streaming**: Uses `StreamlitCallbackHandler` to render expandable real-time tool execution logs and thought processes.

---

## 📁 Repository Structure

```
Stock-Market-AI-Agent/
├── .gitignore              # Standard Python & environment exclusions
├── README.md               # Documentation & setup guide
├── requirements.txt        # Core project dependencies
├── tools.py                # 4 Core LangChain tools
├── agent.py                # ChatOllama Agent, Buffett persona & bounded memory
└── app.py                  # Streamlit chat interface with memory synchronization
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
- *"Give me a breakdown of Reliance Industries based on its balance sheet, P/E ratio, and Owner Earnings."*
- *"Analyze my portfolio: `{\"AAPL\": 45, \"MSFT\": 35, \"GOOGL\": 20}`"*
- *"What is the economic moat and intrinsic value outlook for Nvidia (NVDA)?"*
