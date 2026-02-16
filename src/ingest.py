#!/usr/bin/env python3
"""
ingest.py - Data Pipeline for Personal Research Portal

This module handles the complete ingestion pipeline:
1. Read metadata from data_manifest.csv
2. Load PDF documents using PyPDFLoader
3. Inject critical metadata (source_id, title, year, tags) into every chunk
4. Chunk documents using RecursiveCharacterTextSplitter
5. Embed and store in persistent ChromaDB

Research-Grade Requirements Met:
- Full metadata provenance for citation tracking
- Graceful error handling for missing files
- Persistent vector store for reproducibility
- Configurable chunking parameters

Author: Research Portal Team
Date: February 2026
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Configure logging for production-grade monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/ingest.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ResearchDataIngestor:
    """
    Production-grade data ingestion pipeline for academic research documents.
    
    This class ensures that every document chunk maintains full provenance
    metadata, which is CRITICAL for academic citation requirements.
    
    Attributes:
        base_path: Root directory of the project
        manifest_path: Path to the CSV manifest file
        chroma_persist_dir: Directory for ChromaDB persistence
        chunk_size: Size of text chunks (default: 1000 characters)
        chunk_overlap: Overlap between chunks (default: 100 characters)
    """
    
    def __init__(
        self,
        base_path: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ) -> None:
        """
        Initialize the ingestor with configurable parameters.
        
        Args:
            base_path: Root directory (auto-detected if None)
            chunk_size: Target size for text chunks
            chunk_overlap: Overlap between consecutive chunks for context preservation
        """
        # Auto-detect base path if not provided
        if base_path is None:
            # Navigate from src/ to project root
            self.base_path = Path(__file__).parent.parent.resolve()
        else:
            self.base_path = Path(base_path).resolve()
        
        self.manifest_path = self.base_path / "data" / "data_manifest.csv"
        self.raw_data_dir = self.base_path / "data" / "raw"
        self.chroma_persist_dir = self.base_path / "data" / "chroma_db"
        
        # Chunking parameters - Research-Grade: These values are optimized for
        # academic papers which typically have dense, interconnected paragraphs
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Validate OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY not found in environment. "
                "Please set it in your .env file."
            )
        
        # Initialize embeddings model
        # Research-Grade: Using text-embedding-3-small for cost-efficiency
        # while maintaining quality. Upgrade to text-embedding-3-large for
        # higher dimensional representations if needed.
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        logger.info(f"Initialized ingestor with base_path: {self.base_path}")
        logger.info(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        logger.info(f"Embedding model: {embedding_model}")
    
    def load_manifest(self) -> pd.DataFrame:
        """
        Load and validate the data manifest CSV.
        
        Research-Grade: The manifest provides structured metadata that enables
        proper academic citations. Each row represents one source document.
        
        Returns:
            DataFrame with manifest data
            
        Raises:
            FileNotFoundError: If manifest file doesn't exist
        """
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest file not found at {self.manifest_path}. "
                "Please ensure data_manifest.csv exists in data/ directory."
            )
        
        df = pd.read_csv(self.manifest_path)
        
        # Validate required columns for citation support
        required_columns = {'source_id', 'title', 'year', 'raw_path', 'tags'}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")
        
        logger.info(f"Loaded manifest with {len(df)} entries")
        return df
    
    def load_pdf_with_metadata(
        self, 
        row: pd.Series
    ) -> List[Document]:
        """
        Load a single PDF and inject critical metadata into every page.
        
        CRITICAL FOR CITATIONS: Each document chunk MUST carry its source_id
        so that the RAG engine can properly attribute answers.
        
        Args:
            row: A row from the manifest DataFrame
            
        Returns:
            List of Document objects with metadata, or empty list if file missing
        """
        source_id = row['source_id']
        raw_path = row['raw_path']
        
        # Construct full path - handle both absolute and relative paths
        if Path(raw_path).is_absolute():
            pdf_path = Path(raw_path)
        else:
            pdf_path = self.base_path / raw_path
        
        # Graceful handling of missing files (Research-Grade requirement)
        if not pdf_path.exists():
            logger.warning(
                f"[SKIP] PDF not found for {source_id}: {pdf_path}"
            )
            return []
        
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            
            # CRITICAL: Inject metadata into every page/document
            # This ensures citation traceability through the entire pipeline
            for page in pages:
                page.metadata.update({
                    'source_id': source_id,  # e.g., "Perry2023" - PRIMARY citation key
                    'title': row['title'],
                    'year': int(row['year']) if pd.notna(row['year']) else None,
                    'tags': row['tags'] if pd.notna(row['tags']) else '',
                    'authors': row.get('authors', 'Unknown'),
                    'original_file': str(pdf_path.name),
                    # Preserve page number for precise citations
                    'page_number': page.metadata.get('page', 0) + 1
                })
            
            logger.info(
                f"[OK] Loaded {source_id}: {len(pages)} pages from {pdf_path.name}"
            )
            return pages
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to load {source_id}: {e}")
            return []
    
    def chunk_documents(
        self, 
        documents: List[Document]
    ) -> List[Document]:
        """
        Split documents into chunks while preserving metadata.
        
        Research-Grade: Uses RecursiveCharacterTextSplitter which respects
        natural language boundaries (paragraphs, sentences) better than
        simple character splitting.
        
        Args:
            documents: List of full documents to chunk
            
        Returns:
            List of chunked documents with metadata preserved
        """
        # RecursiveCharacterTextSplitter tries these separators in order,
        # preferring to split at paragraph boundaries first
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            # Keep metadata attached to chunks
            add_start_index=True  # Useful for debugging chunk positions
        )
        
        chunks = splitter.split_documents(documents)
        
        # Verify metadata preservation (Research-Grade: sanity check)
        if chunks and 'source_id' not in chunks[0].metadata:
            raise RuntimeError(
                "Metadata lost during chunking! This breaks citation tracking."
            )
        
        logger.info(
            f"Created {len(chunks)} chunks from {len(documents)} documents"
        )
        return chunks
    
    def create_vector_store(
        self, 
        chunks: List[Document],
        collection_name: str = "research_papers"
    ) -> Chroma:
        """
        Create or update the persistent ChromaDB vector store.
        
        Research-Grade: Using persistent storage ensures reproducibility
        and allows incremental updates without re-processing all documents.
        
        Args:
            chunks: Document chunks with metadata
            collection_name: Name for the ChromaDB collection
            
        Returns:
            Chroma vector store instance
        """
        # Ensure persistence directory exists
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating vector store at {self.chroma_persist_dir}")
        logger.info(f"Embedding {len(chunks)} chunks...")
        
        # Create persistent vector store
        # Research-Grade: Persistence allows the RAG engine to load without
        # re-embedding, saving API costs and time
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=str(self.chroma_persist_dir)
        )
        
        logger.info(
            f"Vector store created with {vectorstore._collection.count()} vectors"
        )
        return vectorstore
    
    def run_full_ingestion(self) -> Dict[str, Any]:
        """
        Execute the complete ingestion pipeline.
        
        Returns:
            Summary statistics of the ingestion process
        """
        logger.info("=" * 60)
        logger.info("STARTING FULL INGESTION PIPELINE")
        logger.info("=" * 60)
        
        stats = {
            'total_sources': 0,
            'loaded_sources': 0,
            'skipped_sources': 0,
            'total_pages': 0,
            'total_chunks': 0,
            'errors': []
        }
        
        # Step 1: Load manifest
        try:
            manifest_df = self.load_manifest()
            stats['total_sources'] = len(manifest_df)
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            raise
        
        # Step 2: Load all PDFs with metadata
        all_documents: List[Document] = []
        for idx, row in manifest_df.iterrows():
            docs = self.load_pdf_with_metadata(row)
            if docs:
                all_documents.extend(docs)
                stats['loaded_sources'] += 1
            else:
                stats['skipped_sources'] += 1
                stats['errors'].append(f"Missing: {row['source_id']}")
        
        stats['total_pages'] = len(all_documents)
        
        if not all_documents:
            logger.error("No documents loaded! Check your raw_path values in manifest.")
            return stats
        
        # Step 3: Chunk documents
        chunks = self.chunk_documents(all_documents)
        stats['total_chunks'] = len(chunks)
        
        # Step 4: Create vector store
        self.create_vector_store(chunks)
        
        # Log summary
        logger.info("=" * 60)
        logger.info("INGESTION COMPLETE")
        logger.info(f"Sources: {stats['loaded_sources']}/{stats['total_sources']} loaded")
        logger.info(f"Pages: {stats['total_pages']}")
        logger.info(f"Chunks: {stats['total_chunks']}")
        if stats['errors']:
            logger.warning(f"Errors: {stats['errors']}")
        logger.info("=" * 60)
        
        return stats


def main() -> None:
    """
    Main entry point for running the ingestion pipeline.
    """
    print("\n" + "=" * 60)
    print("Personal Research Portal - Data Ingestion Pipeline")
    print("=" * 60 + "\n")
    
    try:
        ingestor = ResearchDataIngestor(
            chunk_size=1000,
            chunk_overlap=100
        )
        stats = ingestor.run_full_ingestion()
        
        print("\n✅ Ingestion completed successfully!")
        print(f"   - Loaded: {stats['loaded_sources']} sources")
        print(f"   - Created: {stats['total_chunks']} searchable chunks")
        print(f"   - Vector store saved to: data/chroma_db/")
        
    except Exception as e:
        logger.exception("Ingestion failed!")
        print(f"\n❌ Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
