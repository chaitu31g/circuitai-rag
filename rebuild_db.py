import json
import logging
from pathlib import Path
from dataclasses import asdict
from ingestion.datasheet_chunker import chunk_document
from rag_pipeline.embeddings.bge_embedder import BGEM3Embedder
from rag_pipeline.embeddings.embed_pipeline import EmbeddingPipeline
from rag_pipeline.vectordb.chroma_store import ChromaStore
from backend.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rebuild_from_docling():
    docling_dir = Path("docling_output")
    knowledge_dir = Path("knowledge_json")
    db_dir = Path("data/vectordb")
    
    knowledge_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Chunking
    all_chunks = []
    
    files = list(docling_dir.glob("*.json"))
    total_files = len(files)
    logger.info(f"Found {total_files} parsed JSONs.")
    
    for idx, f in enumerate(files):
        logger.info(f"({idx+1}/{total_files}) Chunking {f.name}...")
        try:
            with open(f, "r", encoding="utf-8") as jf:
                docling_data = json.load(jf)
                
            part_number = f.stem
            chunks = chunk_document(docling_data, part_number=part_number)
            
            # Save to knowledge JSON 
            knowledge_data = [asdict(c) for c in chunks]
            dest_knowledge = knowledge_dir / f"{part_number}_knowledge.json"
            with open(dest_knowledge, "w", encoding="utf-8") as out_f:
                json.dump(knowledge_data, out_f, indent=2, ensure_ascii=False)
                
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Error chunking {f.name}: {e}")
            
    if not all_chunks:
        logger.error("No chunks produced.")
        return
        
    logger.info(f"Created {len(all_chunks)} total chunks.")
    
    # Check if parameter rows were extracted
    param_rows = sum(1 for c in all_chunks if c.chunk_type == "parameter_row")
    logger.info(f"Extracted {param_rows} parameter rows!")
            
    # 2. Embedding
    logger.info("Initializing Embedder...")
    embedder = BGEM3Embedder()
    pipeline = EmbeddingPipeline(embedder=embedder)
    
    raw_chunks_dicts = [asdict(c) for c in all_chunks]
    logger.info("Embedding...")
    embedded = pipeline.run(raw_chunks_dicts)
    
    # Flatten metadata 
    for chunk in embedded:
        flat = {}
        for k, v in chunk.get("metadata", {}).items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                flat[k] = v
            else:
                flat[k] = str(v)
        chunk["metadata"] = flat
        
    # 3. Upsert
    logger.info("Upserting to ChromaDB...")
    store = ChromaStore(
        persist_dir=db_dir,
        collection_name=config.chroma_collection,
        expected_dim=len(embedded[0]["embedding"]) if embedded else 1024
    )
    store.upsert_chunks(embedded)
    store.persist()
    
    logger.info(f"Rebuild completes! db now has {store.count()} chunks.")
    
    # Run a test query 
    logger.info("Running a test query for 'Dynamic characteristics Ciss'")
    results = store.search("Dynamic characteristics Ciss", top_k=3)
    for i, res in enumerate(results):
        logger.info(f"[{i+1}]: {res.get('text')}")

if __name__ == "__main__":
    rebuild_from_docling()
