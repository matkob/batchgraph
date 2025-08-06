import asyncio
import typing as t
from operator import add

from langgraph.graph import END, StateGraph

from batchgraph import GraphStateRepository


class InMemoryStateRepository(GraphStateRepository):
    """Simple in-memory implementation of GraphStateRepository for testing.

    Stores states in a dictionary, useful for development and testing.
    All data is lost when the process ends.
    """

    def __init__(self) -> None:
        """Initialize the in-memory storage."""
        self.storage: dict[str, dict[str, t.Any]] = {}

    def save_state(self, graph_id: str, state: dict[str, t.Any]) -> None:
        self.storage[graph_id] = state.copy()

    def load_state(self, graph_id: str) -> dict[str, t.Any] | None:
        return self.storage.get(graph_id, None)

    def delete_state(self, graph_id: str) -> None:
        if graph_id in self.storage:
            del self.storage[graph_id]

    def exists(self, graph_id: str) -> bool:
        return graph_id in self.storage


class GraphState(t.TypedDict):
    """State that gets passed between nodes in the graph."""

    input_data: str
    messages: t.Annotated[list, add]
    paused: bool
    result: str


class SimpleGraph:
    """Simple implementation of a Langgraph for testing.

    Creates a three-node graph where each node prints to console.
    """

    def __init__(self, input_data: str) -> None:
        """Initialize the graph with input data.

        Args:
            input_data: Initial data to process through the graph.
        """
        self.input_data = input_data
        self.state = {"input_data": input_data, "messages": [], "paused": False, "result": ""}
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile()

    def _build_graph(self) -> StateGraph:
        """Build the langgraph StateGraph.

        Returns:
            A configured StateGraph with three nodes.
        """
        workflow = StateGraph(GraphState)
        workflow.add_node("node1", self._node1_func)
        workflow.add_node("node2", self._node2_func)
        workflow.add_node("node3", self._node3_func)
        workflow.set_entry_point("node1")
        workflow.add_edge("node1", "node2")
        workflow.add_edge("node2", "node3")
        workflow.add_edge("node3", END)

        return workflow

    def _node1_func(self, state: GraphState) -> GraphState:
        """First node - processes initial input.

        Args:
            state: Current graph state.

        Returns:
            Updated graph state.
        """
        state["messages"].append(f"node1_output({state['input_data']})")
        return state

    async def _node2_func(self, state: GraphState) -> GraphState:
        """Second node - simulates batch API pause.

        Args:
            state: Current graph state.

        Returns:
            Updated graph state.
        """
        last_message = state["messages"][-1] if state["messages"] else state["input_data"]
        state["paused"] = True
        state["messages"].append(f"node2_output({last_message})")
        return state

    def _node3_func(self, state: GraphState) -> GraphState:
        """Third node - finalizes processing.

        Args:
            state: Current graph state.

        Returns:
            Updated graph state with final result.
        """
        last_message = state["messages"][-1] if state["messages"] else ""
        state["result"] = f"Final result: {last_message}"
        state["messages"].append(state["result"])
        return state

    def node1(self, input_data: str) -> str:
        """Compatibility method for AwaitableGraph - executes node1.

        Args:
            input_data: Data to process.

        Returns:
            Output from node1.
        """
        state = {"input_data": input_data, "messages": [], "paused": False, "result": ""}
        result_state = self._node1_func(state)
        return result_state["messages"][-1] if result_state["messages"] else input_data

    async def node2(self, input_data: str) -> str:
        """Compatibility method for AwaitableGraph - executes node2.

        Args:
            input_data: Data from node1.

        Returns:
            Output from node2.
        """
        state = self.state
        state["messages"].append(input_data)
        result_state = await self._node2_func(state)
        self.state = result_state
        return result_state["messages"][-1]

    def node3(self, result: str) -> str:
        """Compatibility method for AwaitableGraph - executes node3.

        Args:
            result: Data from node2 or batch API.

        Returns:
            Final result.
        """
        state = self.state
        state["messages"].append(result)
        result_state = self._node3_func(state)
        return result_state["result"]


async def custom_batch_api(input_data: str) -> str:
    """Custom batch API function for testing.

    Can be replaced with any async API call:
    - await litellm.acompletion(...)
    - await openai.ChatCompletion.acreate(...)
    - await anthropic.AsyncAnthropic().messages.create(...)
    - Custom REST API call with httpx/aiohttp

    Args:
        input_data: Data to process.

    Returns:
        Processed result.
    """
    await asyncio.sleep(0.1)
    return f"BatchAPI_Result({input_data})"
