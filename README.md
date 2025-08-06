# [WIP] BatchGraph

> This repository is under rapid development and isn't ready yet. Leave a star if you're excited about the project and want to be notified about future updates. Feel free to open issues or contribute if you're interested in helping shape the development!

A seamless adapter for LangGraph users to integrate asynchronous LLM batch APIs without major code changes. BatchGraph pauses your graph while waiting for results and resumes automatically when the batch API delivers the output.

## Features

- **Zero-refactor integration**: Add batch processing to existing LangGraph workflows with minimal changes
- **Automatic state persistence**: Save and restore graph state during batch processing
- **Flexible storage backends**: Abstract repository pattern for any storage implementation
- **Async/await support**: Built for modern Python asynchronous workflows

## Installation

```bash
uv add batchgraph
```

## Quick Start

### Basic Usage

```python
import asyncio
from batchgraph import AwaitableGraph, GraphStateRepository

# Implement your state repository
class InMemoryRepository(GraphStateRepository):
    def __init__(self):
        self.storage = {}
    
    def save_state(self, graph_id: str, state: dict) -> None:
        self.storage[graph_id] = state.copy()
    
    def load_state(self, graph_id: str) -> dict | None:
        return self.storage.get(graph_id)
    
    def delete_state(self, graph_id: str) -> None:
        self.storage.pop(graph_id, None)
    
    def exists(self, graph_id: str) -> bool:
        return graph_id in self.storage

# Define your batch API function
async def batch_llm_api(prompt: str, model: str = "gpt-4") -> str:
    """Simulate a batch API call that takes time to process."""
    await asyncio.sleep(0.1)  # Simulate processing time
    return f"Response to: {prompt} (using {model})"

# Create and use your graph
async def main():
    repository = InMemoryRepository()
    
    class MyGraph(AwaitableGraph):
        def node1(self, user_input: str) -> str:
            """Process initial input."""
            processed = f"Processed: {user_input}"
            self.state['input_data'] = processed
            return processed
        
        def node2(self, data: str) -> str:
            """This will be replaced by batch API."""
            return self.batch_api_func(data)
        
        def node3(self, api_result: str) -> str:
            """Final processing step."""
            return f"Final: {api_result}"
    
    graph = MyGraph(repository, batch_llm_api, "my-graph-1")
    result = await graph.execute("Hello, world!")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Using the Decorator

```python
from batchgraph import awaitable

@awaitable(repository, batch_llm_api, "chatbot-graph")
class ChatBotGraph:
    """A simple chatbot using batch processing."""
    
    def __init__(self):
        self.conversation_history = []
    
    def node1(self, user_message: str) -> str:
        """Prepare the conversation context."""
        self.conversation_history.append(f"User: {user_message}")
        context = "\n".join(self.conversation_history)
        self.state['input_data'] = context
        return context
    
    def node2(self, context: str) -> str:
        """Will be handled by batch API."""
        return self.batch_api_func(context)
    
    def node3(self, response: str) -> str:
        """Process the API response."""
        self.conversation_history.append(f"Assistant: {response}")
        return response

# Usage
chatbot = ChatBotGraph()
response = await chatbot.execute("What's the weather like?")
```

## How It Works

1. **Normal execution**: Graph runs through `node1` → `node2` → `node3`
2. **Batch API detection**: When `node2` requires batch processing, the graph pauses
3. **State persistence**: Current state is saved to the repository
4. **Batch processing**: Your batch API function is called asynchronously
5. **Automatic resume**: Graph resumes from the saved state when results are ready

## Requirements

- Python 3.13+
- Support for `asyncio`

## Development

```bash
# Clone the repository
git clone https://github.com/your-org/batchgraph.git
cd batchgraph

# Install development dependencies
uv sync --dev

# Run tests
uv run pytest
```

## License

MIT License - see LICENSE file for details.
