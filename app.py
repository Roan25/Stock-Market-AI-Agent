import streamlit as st
from agent import build_stock_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

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
    st.markdown("- *'Check historical archive data for RELIANCE from the Kaggle dataset.'*")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_executor = build_stock_agent(model_choice)
        st.rerun()

# 3. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_model" not in st.session_state or st.session_state.current_model != model_choice:
    st.session_state.current_model = model_choice
    try:
        st.session_state.agent_executor = build_stock_agent(model_choice)
    except Exception as e:
        st.error(f"Failed to initialize Ollama agent with '{model_choice}'. Is Ollama running? Error: {e}")
        st.stop()

# 4. Display Existing Message History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Handle User Input & Streaming Tool Thoughts
if prompt := st.chat_input("Ask about a stock, market trend, or portfolio allocation..."):
    # Display user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response with visual tool execution status
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=True,
            collapse_completed_thoughts=True
        )
        try:
            with st.spinner("Analyzing fundamentals and consulting tools..."):
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
