import pytest
from asera.memory.state import AgentStateManager, SessionState
from asera.memory.in_memory import InMemoryCheckpointSaver
from asera.memory.checkpointer import Checkpoint
from asera.exceptions import SessionNotFoundError


def test_agent_state_manager():
    manager = AgentStateManager()
    thread_id = "test_thread_001"

    # Initialize
    state = manager.initialize_session(
        thread_id=thread_id,
        summary="Prior summary",
        history=[{"role": "user", "content": "Hi"}]
    )
    assert state.summary == "Prior summary"
    assert len(state.messages) == 1

    # Update
    manager.update_session(
        thread_id=thread_id,
        new_messages=[{"role": "model", "content": "Hello!"}]
    )
    fetched = manager.get(thread_id)
    assert len(fetched.messages) == 2

    # Clear
    manager.clear_session(thread_id)
    assert manager.get(thread_id) is None


@pytest.mark.asyncio
async def test_in_memory_checkpoint_saver():
    saver = InMemoryCheckpointSaver()
    thread_id = "thread_abc"

    cp1 = Checkpoint(
        thread_id=thread_id,
        messages=[{"role": "user", "content": "msg 1"}]
    )
    await saver.save(cp1)

    loaded = await saver.load(thread_id)
    assert len(loaded) == 1
    assert loaded[0]["content"] == "msg 1"

    await saver.clear(thread_id)
    assert await saver.load(thread_id) is None
