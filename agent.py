try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    from langchain_classic.memory import ConversationBufferMemory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from tools import (
    analyze_portfolio,
    get_financial_statements,
    get_stock_price_and_fundamentals,
    search_market_news,
    query_nifty50_historical_dataset,
)

BUFFETT_SYSTEM_PROMPT = (
    "You are an AI modeled after Warren Buffett. You are a strict value investor. "
    "You evaluate stocks based on intrinsic value, economic moats, and fundamentals. "
    "You never give day-trading advice. You must ALWAYS use your tools to fetch actual "
    "financial data and news before answering a user's question. Do not hallucinate numbers."
)

def build_stock_agent(model_name: str = "llama3.1"):
    """
    Initializes and returns a tool-calling LangChain AgentExecutor powered locally
    by Ollama with a Warren Buffett value-investing persona and conversation memory.
    """
    tools = [
        get_stock_price_and_fundamentals,
        get_financial_statements,
        search_market_news,
        analyze_portfolio,
        query_nifty50_historical_dataset,
    ]

    # Temperature set to 0 to ensure deterministic, reliable tool calling
    llm = ChatOllama(model=model_name, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", BUFFETT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Persistent conversation memory across multi-turn user dialogs
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )
    return executor
