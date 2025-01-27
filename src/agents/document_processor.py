from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    TextLoader,
    PDFLoader,
    Docx2txtLoader,
    UnstructuredURLLoader
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from .config import settings

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None

    def load_document(self, file_path: str) -> List[str]:
        """Load document based on file extension"""
        if file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        elif file_path.endswith('.pdf'):
            loader = PDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        return loader.load()

    def process_documents(self, documents: List[str]) -> List[str]:
        """Split documents into chunks"""
        texts = self.text_splitter.split_documents(documents)
        return texts

    def create_vector_store(self, texts: List[str]) -> None:
        """Create or update vector store with document chunks"""
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(
                texts,
                self.embeddings
            )
        else:
            self.vector_store.add_texts(texts)

    def save_vector_store(self) -> None:
        """Save vector store to disk"""
        if self.vector_store:
            self.vector_store.save_local(settings.vector_store_path)

    def load_vector_store(self) -> None:
        """Load vector store from disk"""
        if os.path.exists(settings.vector_store_path):
            self.vector_store = FAISS.load_local(
                settings.vector_store_path,
                self.embeddings
            )

    def similarity_search(self, query: str, k: int = 3) -> List[str]:
        """Search for similar documents in vector store"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        return self.vector_store.similarity_search(query, k=k)