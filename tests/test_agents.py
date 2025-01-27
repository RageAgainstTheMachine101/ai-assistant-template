import pytest
from src.agents.base_agent import ConversationManager

@pytest.fixture
def conversation_manager():
    return ConversationManager()

def test_process_query(conversation_manager):
    result = conversation_manager.process_query("What is AI?")
    assert "answer" in result
