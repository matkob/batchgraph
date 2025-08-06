import abc
import logging
import typing as t

logger = logging.getLogger(__name__)


__all__ = [
    "GraphStateRepository",
    "AwaitableGraph",
    "awaitable",
]


class GraphStateRepository(abc.ABC):
    """Abstract base class for graph state storage implementations.

    This defines the interface for storing and retrieving graph execution states.
    Implementations can use any storage backend (database, file system, memory, etc.).
    """

    @abc.abstractmethod
    def save_state(self, graph_id: str, state: dict[str, t.Any]) -> None:
        """
        Save the graph state to storage

        Args:
            graph_id: Unique identifier for the graph instance
            state: The state dictionary to save
        """
        pass

    @abc.abstractmethod
    def load_state(self, graph_id: str) -> dict[str, t.Any] | None:
        """
        Load the graph state from storage

        Args:
            graph_id: Unique identifier for the graph instance

        Returns:
            The saved state dictionary, or None if no state exists
        """
        pass

    @abc.abstractmethod
    def delete_state(self, graph_id: str) -> None:
        """
        Delete the graph state from storage

        Args:
            graph_id: Unique identifier for the graph instance
        """
        pass

    @abc.abstractmethod
    def exists(self, graph_id: str) -> bool:
        """
        Check if a state exists for the given graph_id

        Args:
            graph_id: Unique identifier for the graph instance

        Returns:
            True if state exists, False otherwise
        """
        pass


class AwaitableGraph:
    """Base class that adds batch processing capabilities to Langgraph classes.

    This class can be used directly or through the @awaitable decorator.
    """

    def __init__(
        self,
        state_repository: GraphStateRepository,
        batch_api_func: t.Callable[..., t.Awaitable[t.Any]],
        graph_id: str | None = None,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> None:
        """Initialize the AwaitableGraph.

        Args:
            state_repository: Repository for storing graph state.
            batch_api_func: Function to call for batch API processing.
            graph_id: Optional unique identifier for the graph.
            *args: Additional positional arguments passed to parent class.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(*args, **kwargs)
        if not hasattr(self, "state"):
            self.state = {}
        if not hasattr(self, "graph"):
            self.graph = None

        self.graph_id = graph_id or f"{self.__class__.__name__}_{id(self)}"
        self.state_repository = state_repository
        self.batch_api_func = batch_api_func
        self.batch_api_args = ()
        self.batch_api_kwargs = {}

    def set_batch_api_params(self, *args: t.Any, **kwargs: t.Any) -> None:
        """Set additional parameters to pass to the batch API function.

        Args:
            *args: Positional arguments for the batch API.
            **kwargs: Keyword arguments for the batch API.
        """
        self.batch_api_args = args
        self.batch_api_kwargs = kwargs

    async def execute(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Execute the Langgraph while pausing and resuming as needed.

        Args:
            *args: Positional arguments for graph execution.
            **kwargs: Keyword arguments for graph execution.

        Returns:
            The result of the graph execution.
        """
        saved_state = self.state_repository.load_state(self.graph_id)

        if saved_state:
            self.state = saved_state
        if self.state and self.state.get("paused", False):
            logger.info("Resuming the graph execution...")
            result = await self._resume_state()
        else:
            logger.info("Starting the graph execution from scratch...")
            result = await self._run_graph(*args, **kwargs)

        return result

    async def _run_graph(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Run the Langgraph and handle pause/resume on batch API calls.

        Args:
            *args: Positional arguments for graph execution.
            **kwargs: Keyword arguments for graph execution.

        Returns:
            The result from the final node.
        """
        node1_output = self.node1(*args, **kwargs)

        self.state["paused"] = True
        self.state_repository.save_state(self.graph_id, self.state)

        batch_api_result = await self.node2(node1_output)
        result = self.node3(batch_api_result)

        return result

    async def _resume_state(self) -> t.Any:
        """Resume execution of the graph from the saved state.

        Returns:
            The result from the final node.
        """
        graph_state = self.state
        result = await self.batch_api_func(graph_state["input_data"], *self.batch_api_args, **self.batch_api_kwargs)

        self.state["result"] = result
        self.state["paused"] = False
        self.state_repository.save_state(self.graph_id, self.state)

        return self.node3(result)


def awaitable(
    state_repository: GraphStateRepository,
    batch_api_func: t.Callable[..., t.Awaitable[t.Any]],
    graph_id: str | None = None,
) -> t.Callable[[type], type]:
    """Decorator that adds batch processing capabilities to a Langgraph class.

    Args:
        state_repository: Repository for storing graph state.
        batch_api_func: Function to call for batch API processing.
        graph_id: Optional unique identifier for the graph.

    Returns:
        The decorated class with AwaitableGraph functionality.

    Example:
        async def my_batch_api(data: str, model: str = "gpt-4", **kwargs) -> str:
            response = await litellm.acompletion(
                model=model,
                messages=[{"content": data}],
                **kwargs
            )
            return response.choices[0].message.content

        @awaitable(repository, my_batch_api)
        class MyGraph(SimpleLanggraph):
            pass
    """

    def decorator(graph_class: type) -> type:
        class AwaitableGraphWrapper(AwaitableGraph, graph_class):
            def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
                # Initialize the user's graph class first
                graph_class.__init__(self, *args, **kwargs)
                # Then initialize AwaitableGraph without calling super()
                if not hasattr(self, "state"):
                    self.state = {}
                if not hasattr(self, "graph"):
                    self.graph = None

                self.graph_id = graph_id or f"{self.__class__.__name__}_{id(self)}"
                self.state_repository = state_repository
                self.batch_api_func = batch_api_func
                self.batch_api_args = ()
                self.batch_api_kwargs = {}

        AwaitableGraphWrapper.__name__ = graph_class.__name__
        AwaitableGraphWrapper.__qualname__ = graph_class.__qualname__
        AwaitableGraphWrapper.__module__ = graph_class.__module__

        return AwaitableGraphWrapper

    return decorator
