from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import List, Any, Optional, Dict
from pydantic import BaseModel, Field
from sidekick_tools import playwright_tools, other_tools
import uuid
import asyncio
from datetime import datetime

load_dotenv(override=True)


class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool


class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(
        description="True if more input is needed from the user, or clarifications, or the assistant is stuck"
    )


class Sidekick:
    def __init__(self):
        self.worker_llm_with_tools = None
        self.evaluator_llm_with_output = None
        self.tools = None
        self.llm_with_tools = None
        self.graph = None
        self.sidekick_id = str(uuid.uuid4())
        self.memory = MemorySaver()
        self.browser = None
        self.playwright = None

    async def setup(self):
        self.tools, self.browser, self.playwright = await playwright_tools()
        self.tools += await other_tools()

        worker_llm = ChatOllama(model="qwen2.5:7b")
        self.worker_llm_with_tools = worker_llm.bind_tools(self.tools)
        evaluator_llm = ChatOllama(model="qwen2.5:7b")
        self.evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)
        await self.build_graph()

    def worker(self, state: State) -> Dict[str, Any]:
        system_message = f"""You are a helpful autonomous assistant that can use tools to complete tasks.

Your goal is to continue working until the user's request is fully completed and all success criteria have been met.

Important rules:
- Do not stop early.
- Do not say "I will do this next", "Let's proceed", "Next I will", or describe future actions.
- If another step is needed and a tool is available, perform the step immediately.
- Continue using tools until the task is complete.
- Only provide a final answer when all requested work has been completed.

Tool usage rules:
- You have tools for web browsing, web search, file management, Python execution, and other tasks.
- If the user asks to create, save, write, export, generate, or store a file, ALWAYS use the file management tools.
- NEVER use Python to create, write, save, or modify files.
- All files must be saved inside the sandbox directory using the file management tools.
- If a file is created, clearly state the exact filename in your final answer.
- Do not claim a file was created unless you actually used a file management tool.

Research rules:
- If the user requests current information, exact addresses, Google Maps links, websites, locations, prices, or other factual information, use the search or browser tools to obtain the information before answering.
- Do not guess or invent details.
- Verify information using available tools whenever possible.
- For recommendations like hotels, restaurants, colleges, products, jobs, or travel places, do not give a weak list from one search snippet. Gather enough details to answer the user's constraints.
- If the user gives a budget, location, address requirement, rating requirement, or Google Maps requirement, include those details in the final answer.
- If a browser URL fails with 404 or another error, recover by using search/browser tools again instead of stopping.

Python rules:
- Use Python only for calculations, data processing, analysis, formatting, or transformations.
- If you want Python output, remember to use print().
- Never use Python for file creation or file storage.

The current date and time is:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Success criteria:
{state["success_criteria"]}

You must do exactly one of the following:

1. Ask a clarification question ONLY if you genuinely cannot continue without additional user information.
   Format:
   Question: <your question>

2. Continue working using tools.

3. Provide a final answer only after all requested work is complete and all success criteria have been satisfied.

Do not ask unnecessary questions.
Do not stop when more work can still be done.
"""

        if state.get("feedback_on_work"):
            system_message += f"""
    Previously you thought you completed the assignment, but your reply was rejected because the success criteria was not met.
    Here is the feedback on why this was rejected:
    {state["feedback_on_work"]}
    With this feedback, please continue the assignment, ensuring that you meet the success criteria or have a question for the user."""

        # Add in the system message

        found_system_message = False
        messages = state["messages"]
        for message in messages:
            if isinstance(message, SystemMessage):
                message.content = system_message
                found_system_message = True

        if not found_system_message:
            messages = [SystemMessage(content=system_message)] + messages

        # Invoke the LLM with tools
        response = self.worker_llm_with_tools.invoke(messages)

        # Return updated state
        return {
            "messages": [response],
        }

    def worker_router(self, state: State) -> str:
        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        else:
            return "evaluator"

    def format_conversation(self, messages: List[Any]) -> str:
        conversation = "Conversation history:\n\n"
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                text = message.content or "[Tools use]"
                conversation += f"Assistant: {text}\n"
        return conversation

    def evaluator(self, state: State) -> State:
        last_response = state["messages"][-1].content

        system_message = """You are a strict evaluator that decides whether an Assistant has fully completed the user's task.

Your job is to decide:
1. Whether the success criteria have been met.
2. Whether the Assistant should continue working with tools.
3. Whether the user must provide more input.

Important routing rules:
- Set success_criteria_met=True only if the task is fully completed.
- Set user_input_needed=True ONLY if the Assistant asks the user a direct clarification question using the format "Question: ...".
- If the Assistant can continue using tools, set user_input_needed=False.
- If the Assistant says "I will proceed", "next I will", "let's proceed", "I will now", "I need to find", or describes a future action without doing it, set success_criteria_met=False and user_input_needed=False.
- If more work is needed, set success_criteria_met=False and user_input_needed=False so the graph returns to the worker.
- Do not mark user_input_needed=True just because more work is required.

Quality rules:
- If the user requested a file, success requires the Assistant to clearly confirm the exact filename that was saved.
- If the user requested exact addresses, Google Maps links, prices, current data, or location-specific recommendations, success requires those details to be actually included.
- If the Assistant gives vague items like "location not specified", "not specified in detail", or "some options include" when exact details were requested, mark success_criteria_met=False.
- If the Assistant only copied incomplete search snippets without validating the user's constraints, mark success_criteria_met=False.
- Do not give the Assistant benefit of the doubt if it only says it will do something later.
"""

        user_message = f"""You are evaluating a conversation between the User and Assistant.

The entire conversation is:
{self.format_conversation(state["messages"])}

The success criteria for this assignment is:
{state["success_criteria"]}

The Assistant's latest response is:
{last_response}

Evaluate the latest response carefully.

Return:
- feedback: Explain what is complete and what is missing.
- success_criteria_met: True only if the task is fully complete.
- user_input_needed: True only if the Assistant explicitly asked a direct clarification question using "Question:".

Important:
If the Assistant has not completed the task but can continue working with tools, set:
success_criteria_met=False
user_input_needed=False

Examples:
- If the user asked for hotels under a budget and the Assistant gave hotel names without verified prices/addresses, success_criteria_met=False and user_input_needed=False.
- If the user asked for Google Maps links and the Assistant did not include exact links or locations, success_criteria_met=False and user_input_needed=False.
- If the user asked to save a file and the Assistant did not clearly confirm the saved filename, success_criteria_met=False and user_input_needed=False.
- If the Assistant says "I will now find..." or "Let's proceed..." but has not completed the work, success_criteria_met=False and user_input_needed=False.

This allows the graph to send the task back to the worker instead of ending early.
"""
        if state["feedback_on_work"]:
            user_message += f"Also, note that in a prior attempt from the Assistant, you provided this feedback: {state['feedback_on_work']}\n"
            user_message += "If you're seeing the Assistant repeating the same mistakes, then consider responding that user input is required."

        evaluator_messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ]

        eval_result = self.evaluator_llm_with_output.invoke(evaluator_messages)
        new_state = {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Evaluator Feedback on this answer: {eval_result.feedback}",
                }
            ],
            "feedback_on_work": eval_result.feedback,
            "success_criteria_met": eval_result.success_criteria_met,
            "user_input_needed": eval_result.user_input_needed,
        }
        return new_state

    def _latest_worker_response(self, state: State) -> str:
        """Return the latest non-evaluator AI response text."""
        for message in reversed(state["messages"]):
            if isinstance(message, AIMessage):
                content = message.content or ""
                if not content.startswith("Evaluator Feedback on this answer:"):
                    return content
            elif isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content", "")
                if not content.startswith("Evaluator Feedback on this answer:"):
                    return content
        return ""

    def route_based_on_evaluation(self, state: State) -> str:
        print("SUCCESS:", state["success_criteria_met"])
        print("USER INPUT NEEDED:", state["user_input_needed"])
        print("FEEDBACK:", state["feedback_on_work"])

        if state["success_criteria_met"]:
            return "END"

        # Only stop for user input if the worker actually asked a direct question.
        latest_worker_response = self._latest_worker_response(state)
        if state["user_input_needed"] and "Question:" in latest_worker_response:
            return "END"

        # If the evaluator says more work is needed, send it back to the worker.
        return "worker"

    async def build_graph(self):
        # Set up Graph Builder with State
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("evaluator", self.evaluator)

        # Add edges
        graph_builder.add_conditional_edges(
            "worker", self.worker_router, {"tools": "tools", "evaluator": "evaluator"}
        )
        graph_builder.add_edge("tools", "worker")
        graph_builder.add_conditional_edges(
            "evaluator", self.route_based_on_evaluation, {"worker": "worker", "END": END}
        )
        graph_builder.add_edge(START, "worker")

        # Compile the graph
        self.graph = graph_builder.compile(checkpointer=self.memory)

    async def run_superstep(self, message, success_criteria, history):
        config = {
            "configurable": {"thread_id": self.sidekick_id},
            "recursion_limit": 30,
        }

        state = {
            "messages": [HumanMessage(content=message)],
            "success_criteria": success_criteria or "The answer should be clear and accurate",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
        }
        result = await self.graph.ainvoke(state, config=config)

        # Find the final worker response and final evaluator feedback safely.
        final_reply = ""
        final_feedback = ""

        for msg in reversed(result["messages"]):
            content = ""
            if isinstance(msg, AIMessage):
                content = msg.content or ""
            elif isinstance(msg, dict):
                content = msg.get("content", "")

            if content.startswith("Evaluator Feedback on this answer:") and not final_feedback:
                final_feedback = content
            elif content and not content.startswith("Evaluator Feedback on this answer:") and not final_reply:
                final_reply = content

            if final_reply and final_feedback:
                break

        user = {"role": "user", "content": message}
        reply = {"role": "assistant", "content": final_reply or "No final response was produced."}
        feedback = {"role": "assistant", "content": final_feedback or "No evaluator feedback was produced."}
        return history + [user, reply, feedback]

    def cleanup(self):
        if self.browser:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.browser.close())
                if self.playwright:
                    loop.create_task(self.playwright.stop())
            except RuntimeError:
                # If no loop is running, do a direct run
                asyncio.run(self.browser.close())
                if self.playwright:
                    asyncio.run(self.playwright.stop())
