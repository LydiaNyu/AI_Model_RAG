#!/usr/bin/env python3
"""
rag_engine.py - Retrieval-Augmented Generation Engine for Personal Research Portal

This module implements a production-grade RAG system with:
1. MMR (Maximal Marginal Relevance) retrieval for diverse results
2. Cross-Encoder reranking for improved accuracy
3. Strict grounding prompts to prevent hallucination
4. Automated logging of all queries and responses

Research-Grade Requirements Met:
- Citation enforcement in every response
- Grounded answers with source attribution
- Comprehensive audit logging for reproducibility
- Enhanced retrieval with MMR + reranking

Author: Research Portal Team
Date: February 2026
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Cross-encoder for reranking (Research-Grade enhancement)
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logging.warning(
        "sentence-transformers not installed. Cross-encoder reranking disabled. "
        "Install with: pip install sentence-transformers"
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# =============================================================================
# SYSTEM PROMPT - CRITICAL FOR RESEARCH-GRADE GROUNDING
# =============================================================================

SYSTEM_PROMPT = """You are an expert Research Assistant specializing in Software Engineering, specifically analyzing the impact of AI on Developer Productivity and Skill Decay.

Your goal is to answer research questions based **ONLY** on the provided context chunks.

### CRITICAL RULES (From Phase 1 Protocols):

1.  **Zero-Inference Rule:**
    * You must answer strictly based on the retrieved documents.
    * Do NOT use outside knowledge (e.g., do not explain what GitHub Copilot is unless the text explains it).
    * If the answer is not in the context, state: "I cannot find evidence in the provided corpus to answer this."

2.  **Strict Citation Format:**
    * Every major claim must be immediately followed by a citation.
    * Format: `(SourceID)`. Example: "Copilot increases speed by 55% (Peng2023)."
    * Do NOT use footnotes. Citations must be inline.

3.  **Synthesis & Conflict Resolution:**
    * If multiple sources discuss the same topic (e.g., Productivity), compare them.
    * If sources disagree (e.g., one says productivity goes up, another says security goes down), explicitly mention this tension.
    * Highlight the trade-off between "Speed" (Productivity) and "Quality/Learning" (Skill Decay).

4.  **Tone:**
    * Maintain a rigorous, objective academic tone.
    * Avoid marketing buzzwords.

### CONTEXT:
{context}

### QUESTION:
{question}

Remember: Every claim needs a citation. No exceptions."""


HUMAN_PROMPT = """Based on the research papers in the context above, please answer the following question:

Question: {question}

Provide a thorough, well-cited answer following the grounding and citation rules."""


class ResearchRAGEngine:
    """
    Production-grade RAG engine with enhanced retrieval and strict citation enforcement.
    
    Features:
    - MMR (Maximal Marginal Relevance) for diverse retrieval
    - Cross-Encoder reranking for accuracy improvement
    - Strict grounding prompts to prevent hallucination
    - Comprehensive query logging for audit trails
    
    Attributes:
        vectorstore: ChromaDB vector store instance
        llm: OpenAI language model
        cross_encoder: Optional cross-encoder for reranking
    """
    
    def __init__(
        self,
        base_path: Optional[str] = None,
        model_name: Optional[str] = None,
        use_reranker: bool = True,
        collection_name: str = "research_papers"
    ) -> None:
        """
        Initialize the RAG engine.
        
        Args:
            base_path: Root directory (auto-detected if None)
            model_name: OpenAI model to use (defaults to env var or gpt-4o)
            use_reranker: Whether to use cross-encoder reranking
            collection_name: ChromaDB collection name
        """
        # Path setup
        if base_path is None:
            self.base_path = Path(__file__).parent.parent.resolve()
        else:
            self.base_path = Path(base_path).resolve()
        
        self.chroma_persist_dir = self.base_path / "data" / "chroma_db"
        self.logs_dir = self.base_path / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment.")
        
        # Initialize embedding model (must match ingestion)
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Load persistent vector store
        if not self.chroma_persist_dir.exists():
            raise FileNotFoundError(
                f"Vector store not found at {self.chroma_persist_dir}. "
                "Run ingest.py first to create the vector store."
            )
        
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.chroma_persist_dir)
        )
        
        doc_count = self.vectorstore._collection.count()
        logger.info(f"Loaded vector store with {doc_count} documents")
        
        # Initialize LLM
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,  # Research-Grade: Deterministic outputs for reproducibility
        )
        logger.info(f"Initialized LLM: {self.model_name}")
        
        # Initialize cross-encoder reranker (Research-Grade enhancement)
        self.cross_encoder = None
        if use_reranker and CROSS_ENCODER_AVAILABLE:
            try:
                # ms-marco-MiniLM is efficient and effective for reranking
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                logger.info("Cross-encoder reranker initialized")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}")
        
        # Build prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(HUMAN_PROMPT)
        ])
        
        logger.info("RAG Engine initialized successfully")
    
    def retrieve_with_mmr(
        self,
        query: str,
        k: int = 6,
        fetch_k: int = 20,
        lambda_mult: float = 0.7
    ) -> List[Document]:
        """
        Retrieve documents using Maximal Marginal Relevance (MMR).
        
        Research-Grade Enhancement: MMR balances relevance with diversity,
        ensuring we get varied perspectives from different papers rather
        than redundant similar passages.
        
        Args:
            query: User's question
            k: Number of documents to return
            fetch_k: Number of candidates to fetch before MMR selection
            lambda_mult: Balance between relevance (1.0) and diversity (0.0)
            
        Returns:
            List of diverse, relevant documents
        """
        # MMR retrieval - Research-Grade: This prevents echo-chamber responses
        # by ensuring diversity in retrieved passages
        docs = self.vectorstore.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )
        
        logger.info(f"MMR retrieval: {len(docs)} documents for query: {query[:50]}...")
        return docs
    
    def rerank_with_cross_encoder(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 4
    ) -> List[Document]:
        """
        Rerank documents using a cross-encoder model.
        
        Research-Grade Enhancement: Cross-encoders provide more accurate
        relevance scores than embedding similarity because they jointly
        encode the query and document together.
        
        Args:
            query: User's question
            documents: Initial retrieved documents
            top_k: Number of top documents to keep after reranking
            
        Returns:
            Reranked documents (most relevant first)
        """
        if not self.cross_encoder or not documents:
            return documents[:top_k]
        
        # Create query-document pairs for cross-encoder
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Sort documents by score (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        reranked = [doc for doc, score in scored_docs[:top_k]]
        
        logger.info(
            f"Reranked {len(documents)} docs -> top {len(reranked)} "
            f"(scores: {[f'{s:.3f}' for _, s in scored_docs[:top_k]]})"
        )
        
        return reranked
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a context string with metadata.
        
        Research-Grade: Each chunk is clearly labeled with its source_id
        so the LLM can properly cite sources in its response.
        
        Args:
            documents: Retrieved documents with metadata
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source_id = doc.metadata.get('source_id', 'Unknown')
            title = doc.metadata.get('title', 'Unknown Title')
            year = doc.metadata.get('year', 'N/A')
            page = doc.metadata.get('page_number', 'N/A')
            
            header = (
                f"[Source {i}] {source_id} - \"{title}\" ({year}) - Page {page}\n"
                f"{'-' * 60}"
            )
            context_parts.append(f"{header}\n{doc.page_content}\n")
        
        return "\n\n".join(context_parts)
    
    def log_query(
        self,
        query: str,
        retrieved_docs: List[Document],
        response: str
    ) -> None:
        """
        Log query details to JSON file for audit and analysis.
        
        Research-Grade Requirement: Every query is logged with full details
        for reproducibility, debugging, and research analysis.
        
        Auditability Enhancement: Now captures detailed chunk metadata including
        source_id, year, page, and a content snippet for verification.
        
        Args:
            query: Original user query
            retrieved_docs: Documents used to generate response
            response: Generated response text
        """
        log_file = self.logs_dir / "query_log.json"
        
        # Build detailed chunk information for auditability
        # Each chunk includes metadata + content snippet (first 300 chars)
        retrieved_chunks = []
        for doc in retrieved_docs:
            chunk_info = {
                "source_id": doc.metadata.get('source_id', 'Unknown'),
                "title": doc.metadata.get('title', 'Unknown'),
                "year": doc.metadata.get('year', None),
                "page": doc.metadata.get('page_number', None),
                "authors": doc.metadata.get('authors', 'Unknown'),
                "tags": doc.metadata.get('tags', ''),
                # Content snippet: first 300 chars to prove retrieval without bloating
                "content_snippet": doc.page_content[:300].strip() + "..." if len(doc.page_content) > 300 else doc.page_content.strip()
            }
            retrieved_chunks.append(chunk_info)
        
        # Create log entry with comprehensive details
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            # Keep simple list for backward compatibility
            "retrieved_source_ids": list(set(
                doc.metadata.get('source_id', 'Unknown') 
                for doc in retrieved_docs
            )),
            # NEW: Detailed chunk information for auditability
            "retrieved_chunks": retrieved_chunks,
            "retrieved_count": len(retrieved_docs),
            "full_response": response,
            "model_name": self.model_name,
            "retrieval_method": "MMR + CrossEncoder" if self.cross_encoder else "MMR"
        }
        
        # Append to log file (create if doesn't exist)
        logs = []
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Corrupted log file, starting fresh")
                logs = []
        
        logs.append(entry)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Query logged to {log_file}")
    
    def query(
        self,
        question: str,
        k: int = 6,
        top_k_rerank: int = 4
    ) -> Dict[str, Any]:
        """
        Execute a full RAG query with retrieval, reranking, and generation.
        
        Args:
            question: User's research question
            k: Number of documents for initial MMR retrieval
            top_k_rerank: Number of documents to keep after reranking
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        logger.info(f"Processing query: {question}")
        
        # Step 1: MMR Retrieval (Research-Grade: ensures diversity)
        initial_docs = self.retrieve_with_mmr(question, k=k)
        
        if not initial_docs:
            no_docs_response = {
                "answer": "I cannot find any relevant documents in the corpus to answer this question.",
                "sources": [],
                "retrieved_count": 0,
                "model": self.model_name
            }
            self.log_query(question, [], no_docs_response["answer"])
            return no_docs_response
        
        # Step 2: Cross-Encoder Reranking (Research-Grade: improves accuracy)
        reranked_docs = self.rerank_with_cross_encoder(
            question, initial_docs, top_k=top_k_rerank
        )
        
        # Step 3: Format context with source attribution
        context = self.format_context(reranked_docs)
        
        # Step 4: Generate response with strict grounding
        chain = self.prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "context": context,
            "question": question
        })
        
        # Step 5: Extract detailed source information for auditability
        # Each source includes metadata + content snippet for verification
        sources = []
        for doc in reranked_docs:
            source_info = {
                "source_id": doc.metadata.get('source_id'),
                "title": doc.metadata.get('title'),
                "year": doc.metadata.get('year'),
                "page": doc.metadata.get('page_number'),
                "authors": doc.metadata.get('authors', 'Unknown'),
                "tags": doc.metadata.get('tags', ''),
                # Content snippet: first 300 chars for auditability
                "content_snippet": doc.page_content[:300].strip() + "..." if len(doc.page_content) > 300 else doc.page_content.strip()
            }
            sources.append(source_info)
        
        # Step 6: Log the query (Research-Grade requirement)
        self.log_query(question, reranked_docs, response)
        
        result = {
            "answer": response,
            "sources": sources,
            # Convenience field: list of unique source IDs for quick reference
            "source_ids": list(set(s["source_id"] for s in sources)),
            "retrieved_count": len(reranked_docs),
            "model": self.model_name,
            "retrieval_method": "MMR + CrossEncoder" if self.cross_encoder else "MMR"
        }
        
        logger.info(f"Generated response using {len(sources)} sources")
        return result


def main() -> None:
    """
    Interactive main function for testing the RAG engine.
    """
    print("\n" + "=" * 60)
    print("Personal Research Portal - RAG Query Engine")
    print("=" * 60 + "\n")
    
    try:
        engine = ResearchRAGEngine(use_reranker=True)
        
        print("✅ RAG Engine initialized successfully!")
        print(f"   - Model: {engine.model_name}")
        print(f"   - Reranker: {'Enabled' if engine.cross_encoder else 'Disabled'}")
        print(f"   - Documents: {engine.vectorstore._collection.count()}")
        print("\nEnter your research questions (type 'quit' to exit):\n")
        
        while True:
            question = input("📚 Question: ").strip()
            
            if question.lower() in ('quit', 'exit', 'q'):
                print("\nGoodbye!")
                break
            
            if not question:
                continue
            
            print("\n🔍 Searching and analyzing...\n")
            result = engine.query(question)
            
            print("-" * 60)
            print("📝 ANSWER:")
            print("-" * 60)
            print(result['answer'])
            print("-" * 60)
            print(f"\n📎 Sources used ({len(result['sources'])}):")
            for src in result['sources']:
                print(f"   - {src['source_id']}: {src['title']} ({src['year']})")
            print("\n")
            
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please run ingest.py first to create the vector store.")
        sys.exit(1)
    except Exception as e:
        logger.exception("RAG Engine error!")
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
