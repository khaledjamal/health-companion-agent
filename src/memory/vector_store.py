#vector_store.py

import os
from typing import List, Optional, Dict, Any
import numpy as np
from sqlalchemy import create_engine, text, Column, String, JSON, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
from loguru import logger

from ..models import KnowledgeDocument, RetrievalQuery
from ..config import settings

Base = declarative_base()


class KnowledgeDocumentModel(Base):
    """SQLAlchemy model for knowledge documents with vector embeddings."""
    
    __tablename__ = "knowledge_documents"
    
    id = Column(String, primary_key=True)
    content = Column(String, nullable=False)
    topic = Column(String, nullable=False)
    doc_metadata = Column(JSON, default={})
    embedding = Column(Vector(384))  # 384 dimensions for all-MiniLM-L6-v2


class VectorStoreManager:
    """Manages PostgreSQL vector store with pgvector for RAG knowledge base."""
    
    def __init__(self, connection_url: Optional[str] = None):
        """Initialize vector store manager.
        
        Args:
            connection_url: PostgreSQL connection URL
        """
        self.connection_url = connection_url or settings.postgres_vector_url
        
        # Initialize database connection
        self.engine = create_engine(
            self.connection_url,
            pool_pre_ping=True,
            echo=settings.environment == "development"
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize database
        self._initialize_database()
        
        count = self._get_document_count()
        logger.info(f"Vector store initialized with {count} documents")
    
    def _initialize_database(self) -> None:
        """Initialize database schema and pgvector extension."""
        with self.engine.connect() as conn:
            # Enable pgvector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        
        # Create tables
        Base.metadata.create_all(self.engine)
        logger.info("Vector store tables initialized")
    
    def _get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    def _get_document_count(self) -> int:
        """Get total number of documents in the store."""
        session = self._get_session()
        try:
            count = session.query(KnowledgeDocumentModel).count()
            return count
        finally:
            session.close()
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def add_document(
        self,
        content: str,
        topic: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a document to the knowledge base.
        
        Args:
            content: Document content
            topic: Main topic/category
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        if metadata is None:
            metadata = {}
        
        metadata["topic"] = topic
        
        # Generate document ID
        current_count = self._get_document_count()
        doc_id = f"{topic}_{current_count}"
        
        # Generate embedding
        embedding = self._generate_embedding(content)
        
        # Save to database
        session = self._get_session()
        try:
            doc = KnowledgeDocumentModel(
                id=doc_id,
                content=content,
                topic=topic,
                doc_metadata=metadata,
                embedding=embedding
            )
            session.add(doc)
            session.commit()
            logger.info(f"Added document to knowledge base: {doc_id} (topic: {topic})")
            return doc_id
        finally:
            session.close()
    
    def add_documents_bulk(self, documents: List[KnowledgeDocument]) -> List[str]:
        """Add multiple documents at once.
        
        Args:
            documents: List of knowledge documents
            
        Returns:
            List of document IDs
        """
        session = self._get_session()
        doc_ids = []
        
        try:
            current_count = self._get_document_count()
            
            for i, doc in enumerate(documents):
                doc_id = f"{doc.topic}_{current_count + i}"
                doc_ids.append(doc_id)
                
                metadata = doc.metadata.copy()
                metadata["topic"] = doc.topic
                
                # Generate embedding
                embedding = self._generate_embedding(doc.content)
                
                # Create document model
                doc_model = KnowledgeDocumentModel(
                    id=doc_id,
                    content=doc.content,
                    topic=doc.topic,
                    doc_metadata=metadata,
                    embedding=embedding
                )
                session.add(doc_model)
            
            session.commit()
            logger.info(f"Added {len(documents)} documents to knowledge base")
            return doc_ids
        finally:
            session.close()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filter_topic: Optional[str] = None
    ) -> List[KnowledgeDocument]:
        """Retrieve relevant documents from the knowledge base.
        
        Args:
            query: Search query
            top_k: Number of documents to return
            filter_topic: Filter by specific topic
            
        Returns:
            List of relevant knowledge documents with relevance scores
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        session = self._get_session()
        try:
            # Build query with vector similarity search
            query_obj = session.query(
                KnowledgeDocumentModel,
                KnowledgeDocumentModel.embedding.cosine_distance(query_embedding).label("distance")
            )
            
            # Apply topic filter if specified
            if filter_topic:
                query_obj = query_obj.filter(KnowledgeDocumentModel.topic == filter_topic)
            
            # Order by similarity and limit results
            results = query_obj.order_by(text("distance")).limit(top_k).all()
            
            # Convert to KnowledgeDocument objects
            documents = []
            for doc_model, distance in results:
                # Convert distance to relevance score (0-1, higher is better)
                # Cosine distance ranges from 0 (identical) to 2 (opposite)
                relevance_score = 1.0 - (distance / 2.0)
                
                doc = KnowledgeDocument(
                    content=doc_model.content,
                    metadata=doc_model.doc_metadata,
                    topic=doc_model.topic,
                    relevance_score=relevance_score
                )
                documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} documents for query: '{query[:50]}...'")
            return documents
        finally:
            session.close()
    
    def retrieve_by_query_model(self, query: RetrievalQuery) -> List[KnowledgeDocument]:
        """Retrieve documents using a query model.
        
        Args:
            query: Retrieval query model
            
        Returns:
            List of relevant knowledge documents
        """
        return self.retrieve(
            query=query.query,
            top_k=query.top_k,
            filter_topic=query.filter_topic
        )
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        session = self._get_session()
        try:
            session.query(KnowledgeDocumentModel).delete()
            session.commit()
            logger.warning("Cleared all documents from knowledge base")
        finally:
            session.close()
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        session = self._get_session()
        try:
            count = session.query(KnowledgeDocumentModel).count()
            
            # Get unique topics
            topics = set()
            if count > 0:
                docs = session.query(KnowledgeDocumentModel.topic).distinct().all()
                topics = {doc.topic for doc in docs}
            
            return {
                "total_documents": count,
                "unique_topics": len(topics),
                "topics": list(topics),
                "embedding_model": settings.embedding_model,
                "embedding_dimension": self.embedding_dimension
            }
        finally:
            session.close()
    
    def load_from_directory(self, directory: str) -> int:
        """Load knowledge base documents from a directory.
        
        Args:
            directory: Path to directory containing text files
            
        Returns:
            Number of documents loaded
        """
        loaded_count = 0
        
        if not os.path.exists(directory):
            logger.warning(f"Knowledge base directory not found: {directory}")
            return 0
        
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                topic = filename.replace(".txt", "").replace("_", " ")
                
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                
                if content:
                    self.add_document(
                        content=content,
                        topic=topic,
                        metadata={"source": filename}
                    )
                    loaded_count += 1
        
        logger.info(f"Loaded {loaded_count} documents from {directory}")
        return loaded_count