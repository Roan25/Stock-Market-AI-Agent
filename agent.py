try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

try:
    from langchain.memory import ConversationBufferWindowMemory
except ImportError:
    from langchain_classic.memory import ConversationBufferWindowMemory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from tools import (
    analyze_portfolio,
    get_financial_statements,
    get_stock_price_and_fundamentals,
    search_market_news,
)

BUFFETT_SYSTEM_PROMPT = (
    "You are an AI modeled after Warren Buffett. You are a strict value investor. "
    "You evaluate stocks based on intrinsic value, economic moats, and fundamentals. "
    "You never give day-trading advice. You must ALWAYS use your tools to fetch actual "
    "financial data and news before answering a user's question. Do not hallucinate numbers."
)

def build_stock_agent(model_name: str = "llama3.1", k: int = 5):
    """
    Initializes and returns a tool-calling LangChain AgentExecutor powered locally
    by Ollama with a Warren Buffett value-investing persona and bounded window memory.

    :param model_name: Local Ollama model tag (e.g., 'llama3.1', 'qwen2.5:7b')
    :param k: Number of conversation turns to retain in windowed memory (default: 5)
    """
    tools = [
        get_stock_price_and_fundamentals,
        get_financial_statements,
        search_market_news,
        analyze_portfolio,
    ]

    # Temperature set to 0 for deterministic tool calling; num_ctx=8192 bounds context safely
    llm = ChatOllama(model=model_name, temperature=0, num_ctx=8192)

    prompt = ChatPromptTemplate.from_messages([
        ("system", BUFFETT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Bounded window memory prevents context window overflow across multi-turn dialogs
    memory = ConversationBufferWindowMemory(k=k, memory_key="chat_history", return_messages=True)

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )
    return executor
