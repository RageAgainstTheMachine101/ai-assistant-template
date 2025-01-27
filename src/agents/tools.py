from typing import List, Dict, Any
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from ..database.supabase_client import SupabaseManager

class DocumentSearchTool(BaseTool):
    name = "document_search"
    description = "Search through documents using vector similarity"
    supabase: SupabaseManager

    def __init__(self):
        super().__init__()
        self.supabase = SupabaseManager()

    def _run(self, query: str, run_manager: CallbackManagerForToolRun = None) -> List[Dict[str, Any]]:
        """Search for relevant documents using vector similarity"""
        return self.supabase.similarity_search(query)

class DocumentIngestionTool(BaseTool):
    name = "ingest_document"
    description = "Ingest and process new documents for RAG"
    supabase: SupabaseManager

    def __init__(self):
        super().__init__()
        self.supabase = SupabaseManager()

    def _run(self, texts: List[str], run_manager: CallbackManagerForToolRun = None) -> bool:
        """Ingest new documents into the vector store"""
        try:
            self.supabase.add_texts(texts)
            return True
        except Exception as e:
            print(f"Error ingesting documents: {str(e)}")
            return False