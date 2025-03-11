import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize core components once
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    
if "workflow" not in st.session_state:
    # Initialize Google Gemini model
    st.session_state.llm = ChatGoogleGenerativeAI(
        api_key="AIzaSyDOGNoA-G1ceO6rW0S_ujw6Y0opowIQGf8",
        model="gemini-2.0-flash-exp",
        temperature=1
    )
    
    # Your original workflow setup
    workflow = StateGraph(state_schema=MessagesState)
    
    def call_model(state: MessagesState):
        system_prompt = """You are a Data Science Expert Tutor. Rules:
1. Only answer data science questions (ML, stats, programming, analytics)
2. For non-DS queries: "I specialize in data science topics"
3. Maintain conversation context
4. Explain concepts simply with examples"""
        
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = st.session_state.llm.invoke(messages)
        return {"messages": response}
    
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")
    st.session_state.workflow = workflow.compile(
        checkpointer=st.session_state.memory
    )

# Initialize session-specific components
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit UI setup
st.set_page_config(page_title="Data Science Tutor", page_icon="🧪")
st.title(f"AI Tutor (Session: {st.session_state.thread_id[:8]})")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input handling
if prompt := st.chat_input("Ask data science question..."):
    # Store user message
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        try:
            # Correctly format input for workflow invocation
            input_messages = [HumanMessage(content=prompt)]
            
            # Invoke workflow with formatted input
            response = st.session_state.workflow.invoke(
                {"messages": input_messages},
                st.session_state.config
            )
            
            # Check if response is a dictionary with 'messages' key
            if isinstance(response, dict) and "messages" in response:
                # Extract and display response
                if isinstance(response["messages"], list) and response["messages"]:
                    assistant_response = response["messages"][-1].content
                    st.write(assistant_response)
                    
                    # Store assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response
                    })
                else:
                    st.error("Error processing request: Invalid response format")
            else:
                st.error("Error processing request: Invalid response format")
                
        except Exception as e:
            st.error(f"Error processing request: {str(e)}")
