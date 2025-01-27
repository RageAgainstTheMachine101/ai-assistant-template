from typing import Dict, Optional, List
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseChatMessageHistory
from ..database.supabase_client import SupabaseManager

class PersistentConversationMemory(ConversationBufferMemory):
    def __init__(
        self,
        user_id: str,
        memory_key: str = "chat_history",
        return_messages: bool = True
    ):
        super().__init__(
            memory_key=memory_key,
            return_messages=return_messages
        )
        self.user_id = user_id
        self.supabase = SupabaseManager()

    def save_context(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str]
    ) -> None:
        """Save context from this conversation to buffer and long-term storage"""
        super().save_context(inputs, outputs)

        # Save to Supabase for long-term storage
        memory_data = {
            "messages": self.chat_memory.messages,
            "moving_summary_buffer": self.moving_summary_buffer
        }
        self.supabase.save_conversation_memory(
            self.user_id,
            self.memory_key,
            memory_data
        )

    def load_memory_variables(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Load memory variables from both buffer and long-term storage"""
        # Try to load from long-term storage first
        stored_memory = self.supabase.load_conversation_memory(
            self.user_id,
            self.memory_key
        )

        if stored_memory:
            self.chat_memory.messages = stored_memory["messages"]
            self.moving_summary_buffer = stored_memory.get("moving_summary_buffer", "")

        return super().load_memory_variables(inputs)

    def clear(self) -> None:
        """Clear memory contents"""
        super().clear()
        # Also clear from long-term storage
        self.supabase.save_conversation_memory(
            self.user_id,
            self.memory_key,
            {"messages": [], "moving_summary_buffer": ""}
        )