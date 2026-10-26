import uuid
import html
import streamlit as st
import streamlit.components.v1 as components
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command

# =====================================================================
# 1. STREAMLIT PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Agentic LinkedIn Studio (HITL) | LangGraph",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================================
# 2. YOUR ORIGINAL BACKEND CODE (100% UNCHANGED)
# =====================================================================
load_dotenv()

search_tool = TavilySearch(max_results=3)
tools = [search_tool]

writer_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.8)
writer_llm_with_tools = writer_llm.bind_tools(tools)

MAX_ATTEMPTS = 3


class State(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int


WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim (short paragraphs), around "
    "150-200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags. When you are done researching, "
    "reply with ONLY the final post text."
)


# ---------- nodes ----------

def prepare_node(state: State) -> dict:
    """Runs once per attempt: bumps the counter and adds the human instruction."""
    attempt = state.get("attempt", 0) + 1
    topic = state["topic"]

    if attempt == 1:
        user_message = (
            f"Write a LinkedIn post on this topic: {topic}. "
            "If you need current info, search the web first."
        )
    else:
        feedback = state.get("review_feedback", "")
        user_message = (
            f"Your previous draft on '{topic}' was rejected. "
            f"Here is the reviewer's feedback:\n\n{feedback}\n\n"
            "Write a new, improved draft that fixes every issue mentioned. "
            "Do not repeat the same mistake."
        )

    return {"messages": [("human", user_message)], "attempt": attempt}


def writer_node(state: State) -> dict:
    """Calls the LLM with the full history (so it can see tool results)."""
    messages = [("system", WRITER_SYSTEM_PROMPT)] + state["messages"]
    response = writer_llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)


def extract_draft_node(state: State) -> dict:
    """Once the writer stops calling tools, store its final text as the draft."""
    content = state["messages"][-1].content
    if isinstance(content, list):  # some models return content blocks
        content = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return {"draft": content.strip()}


def human_review_node(state: State) -> dict:
    """Pauses the graph and waits for the human to approve or give feedback."""
    human_response = interrupt({
        "draft": state["draft"],
        "attempt": state["attempt"],
        "instruction": "Type 'approved' to accept, or type your feedback to request a rewrite.",
    })

    response = human_response.strip()

    if response.lower() in ["approved", "approve", "yes", "ok", "good"]:
        return {"is_approved": True, "review_feedback": "Approved by human."}
    return {"is_approved": False, "review_feedback": response}


def should_stop_looping(state: State):
    if state["is_approved"]:
        print("\n[Post approved by human. Ending workflow.]")
        return END
    if state["attempt"] >= MAX_ATTEMPTS:
        print(f"\n[Reached max {MAX_ATTEMPTS} attempts. Ending with last draft.]")
        return END
    print(f"\n[Rejected. Looping back to writer for attempt {state['attempt'] + 1}...]")
    return "prepare"


# ---------- graph ----------

graph = StateGraph(State)

graph.add_node("prepare", prepare_node)
graph.add_node("writer", writer_node)
graph.add_node("tools", tool_node)
graph.add_node("extract_draft", extract_draft_node)
graph.add_node("human_review", human_review_node)

graph.add_edge(START, "prepare")
graph.add_edge("prepare", "writer")

graph.add_conditional_edges(
    "writer",
    tools_condition,
    {"tools": "tools", END: "extract_draft"},
)
graph.add_edge("tools", "writer")

graph.add_edge("extract_draft", "human_review")

graph.add_conditional_edges(
    "human_review",
    should_stop_looping,
    {"prepare": "prepare", END: END},
)

app = graph.compile(checkpointer=MemorySaver())

if "hitl_compiled_app" not in st.session_state:
    st.session_state.hitl_compiled_app = app
app = st.session_state.hitl_compiled_app


# =====================================================================
# 3. TAILWIND / REACT-STYLE CUSTOM CSS & HITL FRONTEND
# =====================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

.stApp {
    background-color: #09090b;
    background-image: 
        radial-gradient(at 12% 8%, rgba(99, 102, 241, 0.15) 0px, transparent 45%),
        radial-gradient(at 88% 12%, rgba(245, 158, 11, 0.10) 0px, transparent 45%),
        radial-gradient(at 50% 92%, rgba(16, 185, 129, 0.08) 0px, transparent 50%);
    color: #f8fafc;
}

.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 4rem !important;
    max-width: 1280px !important;
}

.tw-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.85rem 1.5rem;
    background: rgba(24, 24, 27, 0.7);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    margin-bottom: 1.8rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
}

.tw-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    font-weight: 800;
    font-size: 1.12rem;
    color: #ffffff;
    letter-spacing: -0.02em;
}

.tw-logo-box {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.45);
}

.tw-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 9999px;
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #fbbf24;
    font-size: 0.76rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace !important;
}

.tw-pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #f59e0b;
    box-shadow: 0 0 8px #f59e0b;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(0.95); opacity: 0.8; }
    50% { transform: scale(1.2); opacity: 1; }
    100% { transform: scale(0.95); opacity: 0.8; }
}

.tw-hero {
    text-align: center;
    margin-bottom: 1.8rem;
}

.tw-hero-badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 9999px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #818cf8;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

.tw-hero-title {
    font-size: 2.55rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.15;
    background: linear-gradient(180deg, #ffffff 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.6rem;
}

.tw-hero-sub {
    color: #94a3b8;
    font-size: 0.98rem;
    max-width: 680px;
    margin: 0 auto;
    line-height: 1.6;
}

.tw-agent-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 1.8rem;
}

.tw-agent-card {
    background: rgba(24, 24, 27, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 12px 15px;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: all 0.2s ease;
}

.tw-agent-card:hover {
    border-color: rgba(99, 102, 241, 0.35);
    transform: translateY(-2px);
    background: rgba(30, 30, 36, 0.75);
}

.tw-agent-icon {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}

.tw-agent-meta h4 {
    margin: 0;
    font-size: 0.84rem;
    font-weight: 700;
    color: #f1f5f9;
}

.tw-agent-meta p {
    margin: 2px 0 0 0;
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace !important;
    color: #64748b;
}

div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {
    background-color: rgba(24, 24, 27, 0.9) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stTextInput"] input:focus, div[data-testid="stTextArea"] textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.22) !important;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 50%, #0ea5e9 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.72rem 1.4rem !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.5) !important;
    transition: all 0.2s ease !important;
}

div.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 15px 30px -5px rgba(99, 102, 241, 0.7) !important;
}

div.stButton > button[kind="secondary"] {
    background: rgba(39, 39, 42, 0.65) !important;
    color: #cbd5e1 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 9999px !important;
    padding: 0.35rem 0.85rem !important;
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

div.stButton > button[kind="secondary"]:hover {
    background: rgba(99, 102, 241, 0.18) !important;
    border-color: rgba(99, 102, 241, 0.45) !important;
    color: #ffffff !important;
}

.tw-section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.tw-section-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #e2e8f0;
    display: flex;
    align-items: center;
    gap: 8px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.tw-step-card {
    background: rgba(24, 24, 27, 0.72);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
}

.tw-step-card::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
}

.tw-step-prepare::before { background: #a855f7; }
.tw-step-writer::before { background: #6366f1; }
.tw-step-tools::before { background: #0ea5e9; }
.tw-step-interrupt::before { background: #f59e0b; }
.tw-step-approved::before { background: #10b981; }
.tw-step-rejected::before { background: #f43f5e; }

.tw-step-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.tw-step-name {
    font-weight: 700;
    font-size: 0.88rem;
    color: #f8fafc;
}

.tw-badge {
    padding: 3px 9px;
    border-radius: 9999px;
    font-size: 0.68rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: uppercase;
}

.badge-purple { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
.badge-indigo { background: rgba(99, 102, 241, 0.15); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.3); }
.badge-sky { background: rgba(14, 165, 233, 0.15); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.3); }
.badge-amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }
.badge-emerald { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
.badge-rose { background: rgba(244, 63, 94, 0.15); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.3); }

.tw-step-body {
    font-size: 0.83rem;
    color: #94a3b8;
    line-height: 1.5;
}

.hitl-banner {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.14) 0%, rgba(99, 102, 241, 0.12) 100%);
    border: 1px solid rgba(245, 158, 11, 0.4);
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 16px;
    box-shadow: 0 10px 30px -10px rgba(245, 158, 11, 0.2);
}

.hitl-banner-title {
    font-weight: 800;
    font-size: 0.96rem;
    color: #fbbf24;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 4px;
}

.hitl-banner-sub {
    font-size: 0.82rem;
    color: #e2e8f0;
}

.li-card {
    background: #18181b;
    border: 1px solid rgba(255, 255, 255, 0.11);
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.65);
    margin-bottom: 16px;
}

.li-header {
    padding: 16px 20px 10px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.li-avatar {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 1.1rem;
    color: white;
    flex-shrink: 0;
}

.li-user-info h5 {
    margin: 0;
    font-size: 0.94rem;
    font-weight: 700;
    color: #f8fafc;
}

.li-user-info p {
    margin: 2px 0 0 0;
    font-size: 0.75rem;
    color: #94a3b8;
}

.li-content {
    padding: 8px 20px 18px 20px;
    font-size: 0.92rem;
    line-height: 1.65;
    color: #e2e8f0;
    white-space: pre-wrap;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
}

.li-stats {
    padding: 9px 20px;
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: #64748b;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.li-actions {
    padding: 6px 12px;
    display: flex;
    justify-content: space-around;
}

.li-action-btn {
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 6px 10px;
}

.tw-metrics-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 14px;
}

.tw-metric-box {
    background: rgba(24, 24, 27, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 10px 12px;
    text-align: center;
}

.tw-metric-val {
    font-size: 1.15rem;
    font-weight: 800;
    color: #ffffff;
}

.tw-metric-lbl {
    font-size: 0.68rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 2px;
}

.tw-empty-state {
    background: rgba(24, 24, 27, 0.4);
    border: 1px dashed rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 3.2rem 1.5rem;
    text-align: center;
    color: #64748b;
}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 4. SESSION STATE & GRAPH RUNNER HELPERS
# =====================================================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"linkedin_session_{uuid.uuid4().hex[:6]}"
if "topic_input" not in st.session_state:
    st.session_state.topic_input = ""
if "execution_logs" not in st.session_state:
    st.session_state.execution_logs = []
if "interrupt_data" not in st.session_state:
    st.session_state.interrupt_data = None
if "final_result" not in st.session_state:
    st.session_state.final_result = None
if "workflow_status" not in st.session_state:
    st.session_state.workflow_status = "idle"
if "feedback_clear_flag" not in st.session_state:
    st.session_state.feedback_clear_flag = False

# FIX: Clear feedback draft BEFORE the text_area widget is instantiated
if st.session_state.feedback_clear_flag:
    st.session_state.feedback_draft = ""
    st.session_state.feedback_clear_flag = False
elif "feedback_draft" not in st.session_state:
    st.session_state.feedback_draft = ""


def set_quick_topic(t: str):
    st.session_state.topic_input = t


def set_quick_feedback(fb: str):
    st.session_state.feedback_draft = fb


def reset_session():
    st.session_state.thread_id = f"linkedin_session_{uuid.uuid4().hex[:6]}"
    st.session_state.execution_logs = []
    st.session_state.interrupt_data = None
    st.session_state.final_result = None
    st.session_state.workflow_status = "idle"
    st.session_state.feedback_clear_flag = True


def run_graph_step(payload, status_container):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    st.session_state.interrupt_data = None

    with status_container:
        status_box = st.status("⚡ Executing LangGraph Nodes...", expanded=True)

        for event in app.stream(payload, config=config):
            for node_name, node_update in event.items():

                if node_name == "prepare":
                    att = node_update.get("attempt", 1)
                    card_html = f"""
                    <div class="tw-step-card tw-step-prepare">
                        <div class="tw-step-top">
                            <div class="tw-step-name">📋 Prepare Node</div>
                            <span class="tw-badge badge-purple">Attempt #{att}</span>
                        </div>
                        <div class="tw-step-body">Initialized instructions for Attempt #{att} and queued Writer Agent.</div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
                    st.session_state.execution_logs.append(card_html)

                elif node_name == "writer":
                    msgs = node_update.get("messages", [])
                    last_msg = msgs[-1] if msgs else None
                    called_tool = bool(getattr(last_msg, "tool_calls", None))

                    badge = "Tool Call -> Tavily" if called_tool else "Draft Generated"
                    body = "Writer requested fresh context from Tavily Web Search." if called_tool else "Writer synthesized research & feedback into a new LinkedIn post draft."

                    card_html = f"""
                    <div class="tw-step-card tw-step-writer">
                        <div class="tw-step-top">
                            <div class="tw-step-name">✍️ Writer Agent (Groq)</div>
                            <span class="tw-badge badge-indigo">{badge}</span>
                        </div>
                        <div class="tw-step-body">{html.escape(body)}</div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
                    st.session_state.execution_logs.append(card_html)

                elif node_name == "tools":
                    card_html = """
                    <div class="tw-step-card tw-step-tools">
                        <div class="tw-step-top">
                            <div class="tw-step-name">🌐 Tavily Search ToolNode</div>
                            <span class="tw-badge badge-sky">3 Results Fed Back</span>
                        </div>
                        <div class="tw-step-body">Fetched live web snippets and routed back to Writer Agent.</div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
                    st.session_state.execution_logs.append(card_html)

                elif node_name == "extract_draft":
                    draft_txt = node_update.get("draft", "")
                    words = len(draft_txt.split())
                    card_html = f"""
                    <div class="tw-step-card tw-step-writer">
                        <div class="tw-step-top">
                            <div class="tw-step-name">📄 Extract Draft Node</div>
                            <span class="tw-badge badge-indigo">{words} Words</span>
                        </div>
                        <div class="tw-step-body">Cleaned draft text and routed to <code>human_review</code> node.</div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
                    st.session_state.execution_logs.append(card_html)

                elif node_name == "__interrupt__":
                    int_Info = node_update[0].value
                    st.session_state.interrupt_data = int_Info
                    att = int_Info.get("attempt", 1)
                    card_html = f"""
                    <div class="tw-step-card tw-step-interrupt">
                        <div class="tw-step-top">
                            <div class="tw-step-name">⏸️ Graph Interrupted (HITL)</div>
                            <span class="tw-badge badge-amber">Waiting for You • Attempt {att}</span>
                        </div>
                        <div class="tw-step-body">Execution paused at <code>human_review_node</code> via <code>interrupt()</code>. Approve or send feedback on the right panel.</div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
                    st.session_state.execution_logs.append(card_html)

                elif node_name == "human_review":
                    approved = node_update.get("is_approved", False)
                    fb = node_update.get("review_feedback", "")
                    if approved:
                        card_html = """
                        <div class="tw-step-card tw-step-approved">
                            <div class="tw-step-top">
                                <div class="tw-step-name">👤 Human Review Decision</div>
                                <span class="tw-badge badge-emerald">APPROVED BY YOU ✓</span>
                            </div>
                            <div class="tw-step-body">You approved the draft. Ending workflow.</div>
                        </div>
                        """
                    else:
                        card_html = f"""
                        <div class="tw-step-card tw-step-rejected">
                            <div class="tw-step-top">
                                <div class="tw-step-name">👤 Human Review Decision</div>
                                <span class="tw-badge badge-rose">REWRITE REQUESTED</span>
                            </div>
                            <div class="tw-step-body"><b>Your Feedback:</b> {html.escape(fb)}</div>
                        </div>
                        """
                    st.markdown(card_html, unsafe_allow_html=True)
                    st.session_state.execution_logs.append(card_html)

        snapshot = app.get_state(config)
        current_values = snapshot.values

        if st.session_state.interrupt_data is not None:
            st.session_state.workflow_status = "awaiting_human"
            status_box.update(label="⏸️ Paused at Human-in-the-Loop Checkpoint — Waiting for your review!", state="complete", expanded=True)
        else:
            st.session_state.workflow_status = "completed"
            st.session_state.final_result = current_values
            status_box.update(label="✅ Workflow Completed!", state="complete", expanded=True)


# =====================================================================
# 5. UI LAYOUT & INTERACTIVE WORKSPACE
# =====================================================================

st.markdown(f"""
<div class="tw-navbar">
    <div class="tw-brand">
        <div class="tw-logo-box">⚡</div>
        <span>AgenticPost<span style="color:#818cf8;">.hitl</span></span>
    </div>
    <div class="tw-status-pill">
        <span class="tw-pulse-dot"></span>
        MemorySaver Thread: {st.session_state.thread_id}
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="tw-hero">
    <div class="tw-hero-badge">Human-in-the-Loop (HITL) Edition</div>
    <div class="tw-hero-title">Interactive AI LinkedIn Post Studio</div>
    <div class="tw-hero-sub">
        Drafts high-impact LinkedIn posts with live Tavily web research, pauses execution using <code>langgraph.types.interrupt</code>, and rewrites directly from <b>your</b> real-time feedback.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="tw-agent-grid">
    <div class="tw-agent-card">
        <div class="tw-agent-icon" style="background:rgba(168,85,247,0.15); color:#c084fc;">📋</div>
        <div class="tw-agent-meta">
            <h4>Prepare Node</h4>
            <p>Attempt & Feedback Builder</p>
        </div>
    </div>
    <div class="tw-agent-card">
        <div class="tw-agent-icon" style="background:rgba(99,102,241,0.15); color:#818cf8;">✍️</div>
        <div class="tw-agent-meta">
            <h4>Writer + Tavily</h4>
            <p>Groq • gpt-oss-120b</p>
        </div>
    </div>
    <div class="tw-agent-card">
        <div class="tw-agent-icon" style="background:rgba(245,158,11,0.15); color:#fbbf24;">⏸️</div>
        <div class="tw-agent-meta">
            <h4>Human Interrupt</h4>
            <p>interrupt() & Command()</p>
        </div>
    </div>
    <div class="tw-agent-card">
        <div class="tw-agent-icon" style="background:rgba(16,185,129,0.15); color:#34d399;">💾</div>
        <div class="tw-agent-meta">
            <h4>Checkpointer</h4>
            <p>MemorySaver State</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='font-size:0.76rem; color:#94a3b8; margin-bottom:6px; font-weight:700;'>🔥 QUICK TOPIC IDEAS:</div>", unsafe_allow_html=True)
q1, q2, q3, q4 = st.columns(4)
with q1:
    st.button("🤖 Human-in-the-Loop AI Agents", on_click=set_quick_topic, args=("Why Human-in-the-Loop (HITL) is essential for production AI agents in 2026",), type="secondary", use_container_width=True)
with q2:
    st.button("⚡ System Design vs Coding Speed", on_click=set_quick_topic, args=("Why System Design matters more than typing speed in the AI era",), type="secondary", use_container_width=True)
with q3:
    st.button("🚀 Building with LangGraph", on_click=set_quick_topic, args=("How stateful graphs and checkpoints solve reliability in multi-agent systems",), type="secondary", use_container_width=True)
with q4:
    st.button("🔄 Reset / New Session", on_click=reset_session, type="secondary", use_container_width=True)

in_col1, in_col2 = st.columns([3.6, 1.2])
with in_col1:
    topic_val = st.text_input(
        "Topic",
        key="topic_input",
        placeholder="What topic do you want a LinkedIn post about?",
        label_visibility="collapsed"
    )
with in_col2:
    start_btn = st.button("🚀 Start Generation", type="primary", use_container_width=True)

st.markdown("<div style='margin-top: 1.4rem;'></div>", unsafe_allow_html=True)

left_col, right_col = st.columns([1.02, 1.28], gap="large")

with left_col:
    st.markdown(f"""
    <div class="tw-section-header">
        <div class="tw-section-title">📡 Live Graph Execution Trace</div>
        <span style="font-size:0.74rem; color:#64748b;">Max {MAX_ATTEMPTS} Attempts</span>
    </div>
    """, unsafe_allow_html=True)
    telemetry_container = st.container()

with right_col:
    st.markdown("""
    <div class="tw-section-header">
        <div class="tw-section-title">💼 Human-in-the-Loop Review & Preview</div>
        <span style="font-size:0.74rem; color:#64748b;">Interactive Checkpoint</span>
    </div>
    """, unsafe_allow_html=True)
    review_container = st.container()

if start_btn:
    clean_topic = topic_val.strip()
    if not clean_topic:
        st.warning("⚠️ Please enter a topic first.")
    else:
        reset_session()
        initial_state = {
            "topic": clean_topic,
            "messages": [],
            "draft": "",
            "review_feedback": "",
            "is_approved": False,
            "attempt": 0,
        }
        run_graph_step(initial_state, telemetry_container)
        st.rerun()

with telemetry_container:
    if st.session_state.execution_logs:
        for log_item in st.session_state.execution_logs:
            st.markdown(log_item, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="tw-empty-state">
            <div style="font-size:2rem; margin-bottom:8px;">🛰️</div>
            <div style="font-weight:700; color:#cbd5e1; margin-bottom:4px;">Waiting to Start Graph</div>
            <div style="font-size:0.82rem;">Enter a topic above and click <b>Start Generation</b>. The graph will research, draft, and pause at <code>human_review_node</code> for your input.</div>
        </div>
        """, unsafe_allow_html=True)

with review_container:
    if st.session_state.workflow_status == "awaiting_human" and st.session_state.interrupt_data:
        data = st.session_state.interrupt_data
        current_draft = data.get("draft", "")
        current_attempt = data.get("attempt", 1)
        instruction_txt = data.get("instruction", "")
        word_count = len(current_draft.split()) if current_draft else 0

        st.markdown(f"""
        <div class="hitl-banner">
            <div class="hitl-banner-title">
                <span>⏸️ DRAFT FOR YOUR REVIEW (Attempt {current_attempt} of {MAX_ATTEMPTS})</span>
                <span class="tw-badge badge-amber">{word_count} Words</span>
            </div>
            <div class="hitl-banner-sub">{html.escape(instruction_txt)}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="li-card">
            <div class="li-header">
                <div class="li-avatar">in</div>
                <div class="li-user-info">
                    <h5>Draft Preview • Attempt #{current_attempt}</h5>
                    <p>Awaiting Human-in-the-Loop Approval • Just now • 🌐</p>
                </div>
            </div>
            <div class="li-content">{html.escape(current_draft)}</div>
            <div class="li-stats">
                <span>👍 ❤️ 💡 Preview Mode</span>
                <span>{word_count} words</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#cbd5e1; margin-bottom:6px;'>⚡ QUICK FEEDBACK TEMPLATES (Optional):</div>", unsafe_allow_html=True)
        fb_c1, fb_c2, fb_c3 = st.columns(3)
        with fb_c1:
            st.button("🔥 Stronger Hook", on_click=set_quick_feedback, args=("Make the first line hook much punchier and more curiosity-driven.",), type="secondary", use_container_width=True)
        with fb_c2:
            st.button("✂️ More Concise", on_click=set_quick_feedback, args=("Make the paragraphs shorter, easier to skim, and cut any generic fluff.",), type="secondary", use_container_width=True)
        with fb_c3:
            st.button("📊 Add Concrete Stats", on_click=set_quick_feedback, args=("Include a concrete real-world example or statistic and end with a sharper question.",), type="secondary", use_container_width=True)

        feedback_text = st.text_area(
            "Your Feedback for the Writer Agent",
            key="feedback_draft",
            placeholder="Type your feedback here to request a rewrite (or click Approve below to accept this draft)...",
            height=95
        )

        act_col1, act_col2 = st.columns([1, 1.3])
        with act_col1:
            approve_clicked = st.button("✅ Approve & Finish", type="primary", use_container_width=True)
        with act_col2:
            rewrite_clicked = st.button("🔄 Submit Feedback & Rewrite", type="secondary", use_container_width=True)

        if approve_clicked:
            st.session_state.feedback_clear_flag = True
            run_graph_step(Command(resume="approved"), telemetry_container)
            st.rerun()

        if rewrite_clicked:
            clean_fb = feedback_text.strip()
            if not clean_fb:
                st.warning("⚠️ Please type some feedback or pick a quick template before requesting a rewrite.")
            else:
                st.session_state.feedback_clear_flag = True
                run_graph_step(Command(resume=clean_fb), telemetry_container)
                st.rerun()

    elif st.session_state.workflow_status == "completed" and st.session_state.final_result:
        res = st.session_state.final_result
        final_draft = res.get("draft", "").strip()
        total_attempts = res.get("attempt", 1)
        approved_by_human = res.get("is_approved", False)
        words = len(final_draft.split()) if final_draft else 0

        verdict_label = "APPROVED BY YOU" if approved_by_human else "MAX ATTEMPTS REACHED"
        verdict_color = "#10b981" if approved_by_human else "#f59e0b"

        st.markdown(f"""
        <div class="tw-metrics-row">
            <div class="tw-metric-box">
                <div class="tw-metric-val" style="color:{verdict_color};">{verdict_label}</div>
                <div class="tw-metric-lbl">Final Status</div>
            </div>
            <div class="tw-metric-box">
                <div class="tw-metric-val">{total_attempts} / {MAX_ATTEMPTS}</div>
                <div class="tw-metric-lbl">Total Attempts</div>
            </div>
            <div class="tw-metric-box">
                <div class="tw-metric-val">{words}</div>
                <div class="tw-metric-lbl">Word Count</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="li-card">
            <div class="li-header">
                <div class="li-avatar">in</div>
                <div class="li-user-info">
                    <h5>Final LinkedIn Post <span style="font-weight:400; color:#64748b; font-size:0.8rem;">• Publish Ready</span></h5>
                    <p>Human-in-the-Loop Verified • LangGraph & Groq</p>
                </div>
            </div>
            <div class="li-content">{html.escape(final_draft)}</div>
            <div class="li-stats">
                <span>👍 ❤️ 💡 214</span>
                <span>46 comments • 19 reposts</span>
            </div>
            <div class="li-actions">
                <div class="li-action-btn">👍 Like</div>
                <div class="li-action-btn">💬 Comment</div>
                <div class="li-action-btn">🔁 Repost</div>
                <div class="li-action-btn">✈️ Send</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if final_draft:
            escaped_js = final_draft.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
            components.html(f"""
            <div style="display: flex; justify-content: flex-end;">
                <button id="copyBtn" onclick="copyPost()" style="
                    background: rgba(16, 185, 129, 0.15);
                    border: 1px solid rgba(16, 185, 129, 0.45);
                    color: #34d399;
                    font-family: 'Plus Jakarta Sans', sans-serif;
                    font-weight: 700;
                    font-size: 13px;
                    padding: 9px 18px;
                    border-radius: 10px;
                    cursor: pointer;
                ">📋 Copy Final Post to Clipboard</button>
            </div>
            <script>
            function copyPost() {{
                navigator.clipboard.writeText(`{escaped_js}`).then(() => {{
                    const btn = document.getElementById('copyBtn');
                    btn.innerText = '✅ Copied to Clipboard!';
                    setTimeout(() => {{ btn.innerText = '📋 Copy Final Post to Clipboard'; }}, 2000);
                }});
            }}
            </script>
            """, height=50)

    else:
        st.markdown("""
        <div class="tw-empty-state">
            <div style="font-size:2rem; margin-bottom:8px;">🧑‍💻</div>
            <div style="font-weight:700; color:#cbd5e1; margin-bottom:4px;">Human-in-the-Loop Review Dock</div>
            <div style="font-size:0.82rem;">Once the Writer Agent generates a draft, the graph will pause via <code>interrupt()</code> and display an interactive <b>Approve / Request Rewrite</b> console right here.</div>
        </div>
        """, unsafe_allow_html=True)
