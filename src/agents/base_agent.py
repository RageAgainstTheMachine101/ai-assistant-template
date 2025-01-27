from typing import List, Dict, Optional
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from ..config import settings
from ..services.rag_service import DocumentProcessor

class ConversationManager:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=settings.temperature,
            model_name=settings.model_name,
            max_tokens=settings.max_tokens
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.doc_processor = DocumentProcessor()
        self.qa_chain = None

    def initialize_chain(self) -> None:
        """Initialize the QA chain with vector store"""
        if not self.doc_processor.vector_store:
            self.doc_processor.load_vector_store()
        
        if self.doc_processor.vector_store:
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.doc_processor.vector_store.as_retriever(),
                memory=self.memory,
                return_source_documents=True
            )

    async def process_query(self, question: str, context: Optional[List[str]] = None) -> Dict:
        """Process a query using the RAG system"""
        if context:
            # Process new context documents
            texts = self.doc_processor.process_documents(context)
            self.doc_processor.create_vector_store(texts)
            self.doc_processor.save_vector_store()

        if not self.qa_chain:
            self.initialize_chain()

        if not self.qa_chain:
            raise ValueError("QA chain not initialized and no context provided")

        # Get response from the chain
        response = await self.qa_chain.acall(
            {"question": question}
        )

        # Extract sources from the response
        sources = [doc.page_content for doc in response.get("source_documents", [])]

        return {
            "answer": response["answer"],
            "sources": sources
        }