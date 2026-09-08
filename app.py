import streamlit as st
from agent import build_stock_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

def sync_agent_memory(agent_executor, messages):
    """
    Synchronizes the agent's internal window memory with the Streamlit chat history,
    ensuring switching models or restoring sessions keeps memory in 100% lockstep.
    """
    if hasattr(agent_executor, "memory") and agent_executor.memory is not None:
        agent_executor.memory.clear()
        user_input = None
        for msg in messages:
            if msg["role"] == "user":
                user_input = msg["content"]
            elif msg["role"] == "assistant" and user_input is not None:
                agent_executor.memory.save_context({"input": user_input}, {"output": msg["content"]})
                user_input = None

# 1. Page Configuration
st.set_page_config(
    page_title="Value Investing Market Agent",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Value Investing Market Agent (Buffett Persona)")
st.caption("Powered 100% locally on your GPU via Ollama & LangChain — Zero API Keys Required")

# 2. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Agent Configuration")
    model_choice = st.selectbox(
        "Select Local LLM",
        ["llama3.1", "qwen2.5:7b", "mistral"],
        index=0,
        help="Ensure you have pulled this model in Ollama (e.g., `ollama pull llama3.1`)."
    )

    st.markdown("---")
    st.markdown("### 🔒 Total Privacy & Local Execution")
    st.info(
        "This agent runs entirely on your local machine / GPU using Ollama. "
        "No financial data, portfolio allocations, or queries are transmitted to external cloud LLM providers."
    )

    st.markdown("---")
    st.markdown("### 💡 Example Prompts")
    st.markdown("- *'Analyze AAPL fundamentals and recent news from a value investor perspective.'*")
    st.markdown("- *'Give me a breakdown of RELIANCE or TCS based on current financial metrics.'*")
    st.markdown("- *'Analyze my portfolio risk: {\"AAPL\": 40, \"MSFT\": 35, \"TSLA\": 25}'*")
    st.markdown("- *'What are the owner earnings and free cash flow for Microsoft?'*")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        if "agent_executor" in st.session_state and hasattr(st.session_state.agent_executor, "memory"):
            st.session_state.agent_executor.memory.clear()
        st.rerun()

# 3. Session State Initialization & Memory Synchronization
if "messages" not in st.session_state:
    st.session_state.messages = []

# If model was switched or agent not initialized, re-create agent and sync existing memory
if ("agent_executor" not in st.session_state or
    "current_model" not in st.session_state or
    st.session_state.current_model != model_choice):
    
    st.session_state.current_model = model_choice
    try:
        new_agent = build_stock_agent(model_choice)
        # Re-hydrate the new agent's memory with previous dialog history
        sync_agent_memory(new_agent, st.session_state.messages)
        st.session_state.agent_executor = new_agent
    except Exception as e:
        st.error(f"Failed to initialize Ollama agent with '{model_choice}'. Is Ollama running? Error: {e}")
        st.stop()

# 4. Display Existing Message History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Handle User Input & Live Tool Execution
if prompt := st.chat_input("Ask about a stock, market trend, or portfolio allocation..."):
    # Display user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response with real-time tool thoughts
    with st.chat_message("assistant"):
        thought_container = st.container()
        st_callback = StreamlitCallbackHandler(
            thought_container,
            expand_new_thoughts=True,
            collapse_completed_thoughts=True
        )
        try:
            response = st.session_state.agent_executor.invoke(
                {"input": prompt},
                {"callbacks": [st_callback]}
            )
            output_text = response["output"]
            st.markdown(output_text)
            st.session_state.messages.append({"role": "assistant", "content": output_text})

        except Exception as e:
            error_message = f"Agent Execution Error: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
