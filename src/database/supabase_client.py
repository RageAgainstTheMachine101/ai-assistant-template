from typing import Dict, List, Optional
from supabase import create_client, Client
from langchain.vectorstores import SupabaseVectorStore
from langchain.embeddings import OpenAIEmbeddings
from .config import settings

class SupabaseManager:
    def __init__(self):
        self.client: Optional[Client] = None
        self.vector_store: Optional[SupabaseVectorStore] = None
        self.embeddings = OpenAIEmbeddings()

    def initialize(self) -> None:
        """Initialize Supabase client and vector store"""
        if not self.client:
            self.client = create_client(
                settings.supabase_url,
                settings.supabase_key
            )
            self.vector_store = SupabaseVectorStore(
                self.client,
                self.embeddings,
                table_name="documents",
                query_name="match_documents"
            )

    def add_texts(self, texts: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Add texts to the vector store"""
        if not self.vector_store:
            self.initialize()
        self.vector_store.add_texts(texts, metadata)

    def similarity_search(self, query: str, k: int = 3) -> List[str]:
        """Search for similar documents in vector store"""
        if not self.vector_store:
            self.initialize()
        return self.vector_store.similarity_search(query, k=k)

    def save_conversation_memory(self, user_id: str, memory_key: str, memory_data: Dict) -> None:
        """Save conversation memory to Supabase"""
        if not self.client:
            self.initialize()
        self.client.table("conversation_memory").upsert({
            "user_id": user_id,
            "memory_key": memory_key,
            "memory_data": memory_data
        }).execute()

    def load_conversation_memory(self, user_id: str, memory_key: str) -> Optional[Dict]:
        """Load conversation memory from Supabase"""
        if not self.client:
            self.initialize()
        result = self.client.table("conversation_memory").select("memory_data").eq(
            "user_id", user_id
        ).eq("memory_key", memory_key).execute()
        return result.data[0]["memory_data"] if result.data else None

    async def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """Validate API key and return user role if valid"""
        if not self.client:
            self.initialize()
        result = self.client.table("api_keys").select("role").eq(
            "key", api_key
        ).execute()
        return result.data[0] if result.data else None
