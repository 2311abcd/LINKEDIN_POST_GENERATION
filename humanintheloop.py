from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command

load_dotenv()

search_tool = TavilySearch(max_results=3)
tools = [search_tool]

writer_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
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

# writer -> tools (if it asked for a search) or -> extract_draft (if it wrote the post)
graph.add_conditional_edges(
    "writer",
    tools_condition,
    {"tools": "tools", END: "extract_draft"},
)
graph.add_edge("tools", "writer")  # feed search results back to the writer

graph.add_edge("extract_draft", "human_review")

graph.add_conditional_edges(
    "human_review",
    should_stop_looping,
    {"prepare": "prepare", END: END},
)

app = graph.compile(checkpointer=MemorySaver())


# ---------- CLI ----------

def main():
    print("=" * 55)
    print("Welcome to the LinkedIn Post Generator (HITL Edition)")
    print("=" * 55)
    print("\nThis tool will draft a LinkedIn post for you, show it to")
    print("YOU for review, and rewrite based on your feedback.")
    print("=" * 55)

    topic = input("\nWhat topic do you want a LinkedIn post about?\n> ").strip()
    if not topic:
        print("\nNo topic given. Exiting.")
        return

    print("\nStarting generation...\n")

    config = {"configurable": {"thread_id": "linkedin_session_1"}}
    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0,
    }

    result = app.invoke(initial_state, config=config)

    while "__interrupt__" in result:
        data = result["__interrupt__"][0].value

        print("\n" + "=" * 55)
        print(f"DRAFT FOR YOUR REVIEW (Attempt {data['attempt']})")
        print("=" * 55)
        print(data["draft"])
        print("=" * 55)
        print(f"\n{data['instruction']}")

        human_input = input("\nYour response: ").strip()
        result = app.invoke(Command(resume=human_input), config=config)

    print("\n" + "=" * 55)
    print("FINAL LINKEDIN POST")
    print("=" * 55)
    print(result["draft"])
    print("=" * 55)
    print(f"Total attempts: {result['attempt']}")
    print(f"Approved by human: {result['is_approved']}")


if __name__ == "__main__":
    main()