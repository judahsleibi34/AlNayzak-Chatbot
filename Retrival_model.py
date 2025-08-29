#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Interactive Semantic Search for Arabic PDF QA
=====================================================

An improved version focusing on precision, structured data extraction,
and targeted retrieval for complex documents.
Assumes chunks.json is correctly generated from the PDF extraction process.
"""

import os
import json
import numpy as np
import time
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

# --- Configuration ---
CHUNKS_FILE_PATH = 'out/chunks.json' # Path to your chunks file

# --- Embedding Model ---
# Using a strong multilingual model suitable for Arabic
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large' # Or 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' for speed/less resource intensive

# --- Reranker Model ---
RERANKER_MODEL_NAME = 'BAAI/bge-reranker-v2-m3' # Strong reranker

# --- RETRIEVAL PARAMETERS (ENHANCED & TARGETED) ---
# Increased candidate pool for better coverage, especially for complex structures
TOP_K_BM25_ENHANCED = 50
TOP_K_VECTOR_ENHANCED = 50
TOP_K_HYBRID_FUSED_ENHANCED = 75  # Larger pool for reranking
TOP_K_RERANKED_ENHANCED = 25      # More results for aggregation/structured output
RRF_K_ENHANCED = 60               # Standard RRF parameter

# --- TARGETED RETRIEVAL CONFIG ---
# Pages identified as containing critical information (e.g., authority matrix)
TARGET_PAGES = [122, 123, 124]

# --- TEXT NORMALIZATION (DUPLICATED HERE FOR CONSISTENCY) ---
import re
_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]")
_NOISE_CHARS = re.compile(r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFFa-zA-Z0-9\s\.\,\!\?\:\;\-\(\)\[\]\{\}\"\'\/\\@#\$%\^&\*\+\=\_\|\~\`]")

def strip_diacritics(s: str) -> str:
    """Remove Arabic diacritics (tashkeel)."""
    return _ARABIC_DIACRITICS.sub("", s)

def strip_tatweel(s: str) -> str:
    """Remove Arabic Tatweel (elongation) character."""
    return s.replace("ـ", "")

def clean_ocr_noise(s: str) -> str:
    """Remove common OCR artifacts and noise."""
    if not s:
        return s
    s = re.sub(r'[|\\/_\-]{3,}', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'^\s*[|\\/_\-]+\s*$', '', s)
    s = _NOISE_CHARS.sub('', s)
    return s.strip()

def normalize_arabic_for_query(s: str) -> str:
    """Normalize Arabic text for querying, similar to chunk text_norm."""
    if not s:
        return s
    s = clean_ocr_noise(s)
    s = strip_diacritics(strip_tatweel(s))
    s = (s.replace("أ", "ا")
           .replace("إ", "ا")
           .replace("آ", "ا")
           .replace("ٱ", "ا")
           .replace("ى", "ي"))
    # Consider if mapping Ta Marbuta is needed for queries too
    # s = s.replace("ة", "ه")
    return s

# --- LOGGING ---
logging.basicConfig(level=logging.INFO) # Change to logging.WARNING to reduce noise during search
logger = logging.getLogger(__name__)

class EnhancedSemanticSearchPipeline:
    def __init__(self, chunks_file_path, embedding_model_name, reranker_model_name):
        self.chunks_file_path = chunks_file_path
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name

        self.chunks_data = []
        self.texts_for_embedding = [] # List of text_norm strings
        self.chunk_embeddings = None
        self.faiss_index = None

        self.tokenized_corpus = [] # For BM25
        self.bm25_model = None

        self.embedding_model = None
        self.reranker_model = None

        logger.info("Initializing Enhanced Semantic Search Pipeline...")
        self._load_data()
        self._setup_embedding_model()
        self._setup_reranker_model()
        self._generate_embeddings()
        self._build_faiss_index()
        self._setup_bm25()

    def _load_data(self):
        """Loads chunks from the JSON file."""
        logger.info(f"Loading chunks from {self.chunks_file_path}...")
        try:
            with open(self.chunks_file_path, 'r', encoding='utf-8') as f:
                self.chunks_data = json.load(f)
            logger.info(f"Loaded {len(self.chunks_data)} chunks.")
            self.texts_for_embedding = [chunk.get('text_norm', '') for chunk in self.chunks_data]
            if not self.texts_for_embedding or not all(self.texts_for_embedding):
                 logger.warning("Some chunks might be missing 'text_norm'. Check data integrity.")
        except FileNotFoundError:
            logger.error(f"File {self.chunks_file_path} not found.")
            raise
        except Exception as e:
            logger.error(f"Error loading chunks: {e}")
            raise

    def _setup_embedding_model(self):
        """Initializes the sentence transformer model for embeddings."""
        logger.info(f"Loading embedding model: {self.embedding_model_name}...")
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded.")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def _setup_reranker_model(self):
        """Initializes the cross-encoder model for reranking."""
        logger.info(f"Loading reranker model: {self.reranker_model_name}...")
        try:
            from sentence_transformers import CrossEncoder
            self.reranker_model = CrossEncoder(self.reranker_model_name)
            logger.info("Reranker model loaded.")
        except Exception as e:
             logger.error(f"Error loading reranker model: {e}")
             raise

    def _generate_embeddings(self):
        """Generates embeddings for all chunk texts."""
        logger.info("Generating embeddings for chunks...")
        try:
            self.chunk_embeddings = self.embedding_model.encode(
                self.texts_for_embedding,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True # Explicitly normalize during encoding if supported
            )
            logger.info(f"Generated {self.chunk_embeddings.shape[0]} embeddings of dimension {self.chunk_embeddings.shape[1]}.")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def _build_faiss_index(self):
        """Builds the FAISS index for vector similarity search using Cosine Similarity."""
        logger.info("Building FAISS index for Cosine Similarity...")
        try:
            import faiss
            # Ensure embeddings are a float32 NumPy array
            self.chunk_embeddings = np.array(self.chunk_embeddings).astype('float32')

            # Normalize embeddings for Cosine Similarity (if not done during encoding)
            # faiss.normalize_L2(self.chunk_embeddings) # Usually not needed if normalized_embeddings=True

            # Create FAISS IndexFlatIP (Inner Product) for normalized vectors = Cosine Similarity
            dimension = self.chunk_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            logger.info(f"FAISS IndexFlatIP created with dimension {dimension}.")

            # Add embeddings to the index
            self.faiss_index.add(self.chunk_embeddings)
            logger.info(f"Added {self.chunk_embeddings.shape[0]} vectors to the FAISS index.")
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            raise

    def _setup_bm25(self):
        """Sets up the BM25 model for keyword-based retrieval."""
        logger.info("Setting up BM25 model...")
        try:
            from rank_bm25 import BM25Okapi
            # Tokenize the corpus for BM25 (using text_norm)
            self.tokenized_corpus = [doc.split(" ") for doc in self.texts_for_embedding]
            self.bm25_model = BM25Okapi(self.tokenized_corpus)
            logger.info("BM25 model initialized.")
        except Exception as e:
            logger.error(f"Error setting up BM25: {e}")
            raise

    def dense_retrieve(self, query_text_norm: str, top_k: int = 10):
        """Performs dense vector retrieval using FAISS."""
        logger.debug(f"Dense retrieval for query (top {top_k})")
        start_time = time.time()
        try:
            # 1. Encode the NORMALIZED query
            query_embedding = self.embedding_model.encode([query_text_norm], convert_to_numpy=True, normalize_embeddings=True).astype('float32')

            # 2. Search the FAISS index
            scores, indices = self.faiss_index.search(query_embedding, top_k)

            # 3. Format results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks_data):
                    chunk = self.chunks_data[idx]
                    results.append({
                        'idx': int(idx),
                        'score': float(scores[0][i]), # Cosine similarity score
                        'text_raw': chunk.get('text_raw', ''),
                        'text_norm': chunk.get('text_norm', ''),
                        'page': chunk.get('page', -1),
                        'source_ref': chunk.get('source_ref', ''),
                        'type': 'vector'
                    })
            logger.debug(f"Dense retrieval completed in {time.time() - start_time:.2f} seconds.")
            return results
        except Exception as e:
            logger.error(f"Error during dense retrieval: {e}")
            return []

    def bm25_retrieve(self, query_text_norm: str, top_k: int = 10):
        """Performs BM25 keyword-based retrieval."""
        logger.debug(f"BM25 retrieval for query (top {top_k})")
        start_time = time.time()
        try:
            # 1. Tokenize the NORMALIZED query
            tokenized_query = query_text_norm.split(" ")

            # 2. Get BM25 scores for all documents
            bm25_scores = self.bm25_model.get_scores(tokenized_query)

            # 3. Get top K indices
            top_indices = np.argsort(bm25_scores)[::-1][:top_k]

            # 4. Format results
            results = []
            for idx in top_indices:
                if idx < len(self.chunks_data):
                    chunk = self.chunks_data[idx]
                    results.append({
                        'idx': int(idx),
                        'score': float(bm25_scores[idx]), # BM25 score
                        'text_raw': chunk.get('text_raw', ''),
                        'text_norm': chunk.get('text_norm', ''),
                        'page': chunk.get('page', -1),
                        'source_ref': chunk.get('source_ref', ''),
                        'type': 'bm25'
                    })
            logger.debug(f"BM25 retrieval completed in {time.time() - start_time:.2f} seconds.")
            return results
        except Exception as e:
            logger.error(f"Error during BM25 retrieval: {e}")
            return []

    def hybrid_retrieve(self, query_text_norm: str, top_k_bm25: Optional[int] = None, top_k_vector: Optional[int] = None, top_k_final: Optional[int] = None, rrf_k: Optional[int] = None):
        """Combines BM25 and Vector results using Reciprocal Rank Fusion (RRF)."""
        top_k_bm25 = top_k_bm25 or TOP_K_BM25_ENHANCED
        top_k_vector = top_k_vector or TOP_K_VECTOR_ENHANCED
        top_k_final = top_k_final or TOP_K_HYBRID_FUSED_ENHANCED
        rrf_k = rrf_k or RRF_K_ENHANCED

        logger.debug(f"Hybrid (RRF) retrieval...")
        start_time = time.time()
        try:
            # 1. Get results from both methods using NORMALIZED query
            bm25_results = self.bm25_retrieve(query_text_norm, top_k=top_k_bm25)
            vector_results = self.dense_retrieve(query_text_norm, top_k=top_k_vector)

            # 2. Create a set of unique indices from both results
            all_indices = set()
            bm25_ranks = {} # {index: rank}
            vector_ranks = {} # {index: rank}

            for i, res in enumerate(bm25_results):
                idx = res['idx']
                all_indices.add(idx)
                bm25_ranks[idx] = i + 1 # Rank starts at 1

            for i, res in enumerate(vector_results):
                idx = res['idx']
                all_indices.add(idx)
                vector_ranks[idx] = i + 1 # Rank starts at 1

            # 3. Calculate RRF scores
            fused_scores = {}
            for idx in all_indices:
                bm25_rank = bm25_ranks.get(idx, top_k_bm25 + 1) # If not in top K, use K+1
                vector_rank = vector_ranks.get(idx, top_k_vector + 1)
                rrf_score = 1 / (rrf_k + bm25_rank) + 1 / (rrf_k + vector_rank)
                fused_scores[idx] = rrf_score

            # 4. Sort by fused score and get top K
            sorted_indices = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
            final_indices = sorted_indices[:top_k_final]

            # 5. Retrieve full chunk data for final results
            results = []
            for idx in final_indices:
                if idx < len(self.chunks_data):
                    chunk = self.chunks_data[idx]
                    results.append({
                        'idx': int(idx),
                        'fused_score': float(fused_scores[idx]),
                        'bm25_score': float(bm25_ranks.get(idx, 0)),
                        'vector_score': float(vector_ranks.get(idx, 0)),
                        'text_raw': chunk.get('text_raw', ''),
                        'text_norm': chunk.get('text_norm', ''),
                        'page': chunk.get('page', -1),
                        'source_ref': chunk.get('source_ref', ''),
                        'type': 'hybrid'
                    })
            logger.debug(f"Hybrid retrieval completed in {time.time() - start_time:.2f} seconds.")
            return results
        except Exception as e:
            logger.error(f"Error during hybrid retrieval: {e}")
            return []

    def rerank(self, query_text_norm: str, retrieved_chunks: List[Dict], top_k_reranked: Optional[int] = None):
        """Reranks a list of retrieved chunks using the Cross-Encoder."""
        top_k_reranked = top_k_reranked or TOP_K_RERANKED_ENHANCED
        logger.debug(f"Reranking {len(retrieved_chunks)} chunks (top {top_k_reranked})")
        start_time = time.time()
        if not retrieved_chunks:
            logger.warning("No chunks provided for reranking.")
            return []

        try:
            # 1. Prepare pairs for the reranker: (NORMALIZED query, passage text_norm)
            # Using text_norm for reranking consistency
            pairs = [[query_text_norm, chunk.get('text_norm', '')] for chunk in retrieved_chunks]

            # 2. Predict relevance scores
            reranker_scores = self.reranker_model.predict(pairs, convert_to_numpy=True)

            # 3. Add scores to chunks and sort
            for i, chunk in enumerate(retrieved_chunks):
                chunk['rerank_score'] = float(reranker_scores[i])

            # Sort by reranker score descending
            reranked_chunks = sorted(retrieved_chunks, key=lambda x: x.get('rerank_score', -1), reverse=True)

            logger.debug(f"Reranking completed in {time.time() - start_time:.2f} seconds.")
            # Return top K reranked results
            return reranked_chunks[:top_k_reranked]
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # If reranking fails, return the input chunks sorted by fused score
            return sorted(retrieved_chunks, key=lambda x: x.get('fused_score', x.get('score', 0)), reverse=True)[:top_k_reranked]


    def search(self, query_text_raw: str, use_hybrid: bool = True, use_reranker: bool = True, top_k_final: Optional[int] = None, top_k_reranked: Optional[int] = None):
        """
        Main search function orchestrating the pipeline.
        1. Normalize the query.
        2. Retrieve (Dense, BM25, or Hybrid).
        3. Rerank (Optional).
        """
        # --- 1. QUERY NORMALIZATION ---
        query_text_norm = normalize_arabic_for_query(query_text_raw)
        logger.info(f"Searching for (normalized): '{query_text_norm[:50]}...'")

        top_k_final = top_k_final or TOP_K_HYBRID_FUSED_ENHANCED
        top_k_reranked = top_k_reranked or TOP_K_RERANKED_ENHANCED

        logger.debug(f"Pipeline: Hybrid={use_hybrid}, Reranker={use_reranker}")

        # --- 2. INITIAL RETRIEVAL (ENHANCED) ---
        if use_hybrid:
            initial_results = self.hybrid_retrieve(
                query_text_norm,
                top_k_bm25=TOP_K_BM25_ENHANCED,
                top_k_vector=TOP_K_VECTOR_ENHANCED,
                top_k_final=top_k_final, # Use the potentially overridden final count
                rrf_k=RRF_K_ENHANCED
            )
        else:
            # Fallback to just dense retrieval if not hybrid (still use enhanced top_k)
            initial_results = self.dense_retrieve(query_text_norm, top_k=top_k_final)

        if not initial_results:
            logger.info("No results found from initial retrieval.")
            return []

        # --- 3. RERANKING (ENHANCED) ---
        if use_reranker:
            # Potentially feed slightly more to reranker than final desired, then trim
            reranker_pool_size = min(len(initial_results), top_k_reranked + 10) # Slightly larger pool
            reranker_input = initial_results[:reranker_pool_size]
            final_results = self.rerank(query_text_norm, reranker_input, top_k_reranked=top_k_reranked)
        else:
            # If no reranking, sort by the best available score and take top K
            if use_hybrid:
                final_results = sorted(initial_results, key=lambda x: x.get('fused_score', 0), reverse=True)[:top_k_reranked]
            else:
                final_results = sorted(initial_results, key=lambda x: x.get('score', 0), reverse=True)[:top_k_reranked]

        logger.info(f"Search completed. Returning {len(final_results)} final results.")
        return final_results

    # --- ENHANCEMENT: MULTI-QUERY APPROACH ---
    def search_multi_query(self, base_query: str, additional_queries: List[str], top_k_reranked: int = TOP_K_RERANKED_ENHANCED) -> List[Dict]:
        """Performs search with multiple related queries and aggregates results."""
        logger.info(f"Performing multi-query search for base query: '{base_query}'")
        all_results = []
        queries = [base_query] + additional_queries

        for i, q in enumerate(queries):
            logger.debug(f"  - Executing query {i+1}/{len(queries)}: '{q}'")
            results = self.search(q, use_hybrid=True, use_reranker=True, top_k_reranked=top_k_reranked)
            all_results.extend(results)
        
        # Deduplicate results based on chunk index to avoid redundancy
        unique_results = {res['idx']: res for res in all_results}.values()
        # Sort aggregated results by rerank score if available, otherwise fused score
        sorted_results = sorted(unique_results, key=lambda x: x.get('rerank_score', x.get('fused_score', 0)), reverse=True)
        logger.info(f"Multi-query search aggregated {len(sorted_results)} unique results.")
        return sorted_results[:top_k_reranked]

    # --- ENHANCEMENT: TARGETED PAGE RETRIEVAL ---
    def search_targeted_pages(self, query: str, target_pages: List[int], top_k_reranked: int = TOP_K_RERANKED_ENHANCED) -> List[Dict]:
        """Filters search results to only include chunks from specific pages."""
        logger.info(f"Performing targeted search on pages {target_pages} for query: '{query}'")
        # 1. Get results from normal search
        initial_results = self.search(query, use_hybrid=True, use_reranker=True, top_k_reranked=top_k_reranked * 2) # Get more candidates
        
        # 2. Filter results by page number
        targeted_results = [res for res in initial_results if res.get('page') in target_pages]
        
        # 3. Re-sort by score and limit
        sorted_targeted = sorted(targeted_results, key=lambda x: x.get('rerank_score', x.get('fused_score', 0)), reverse=True)
        logger.info(f"Targeted search found {len(sorted_targeted)} results on specified pages.")
        return sorted_targeted[:top_k_reranked]

    # --- ENHANCEMENT: SPECIALIZED AUTHORITY MATRIX SEARCH ---
    def search_complete_authority_matrix(self, base_query: str = "مصفوفة الصلاحيات") -> List[Dict]:
        """Specialized search for complete authority matrix procedures."""
        logger.info("Performing specialized search for complete authority matrix procedures...")
        
        # 1. Multi-query approach for matrix content
        matrix_queries = [
            "الاجراءات المدرجة مصفوفة الصلاحيات",
            "الصلاحيات الادارية المالية التقنية",
            "قائمة الاجراءات والسلطات",
            "جدول الصلاحيات والمسؤوليات",
            "هيكل الصلاحيات المؤسسة"
        ]
        
        # 2. Perform multi-query search
        logger.debug("  - Running multi-query search...")
        all_results = self.search_multi_query(base_query, matrix_queries, top_k_reranked=TOP_K_RERANKED_ENHANCED)
        
        # 3. Filter results to target specific pages (122-124)
        logger.debug("  - Filtering results to target pages...")
        targeted_results = [res for res in all_results if res.get('page') in TARGET_PAGES]
        
        # 4. Sort final results by score
        final_results = sorted(targeted_results, key=lambda x: x.get('rerank_score', x.get('fused_score', 0)), reverse=True)
        
        logger.info(f"Specialized authority matrix search completed. Found {len(final_results)} targeted results.")
        # Note: Full reconstruction logic (`reconstruct_complete_matrix`) would go here
        # if we had a specific algorithm for it. For now, we return the best targeted chunks.
        return final_results 

    # --- ENHANCEMENT: MULTI-CHUNK AGGREGATION (SIMPLIFIED) ---
    def aggregate_related_chunks(self, anchor_chunks: List[Dict], context_window: int = 2) -> List[Dict]:
        """
        Combines related chunks from the same pages/sections to provide more context.
        This is a simplified version. A more complex one could look at `route_tokens`.
        """
        logger.info(f"Aggregating related chunks for {len(anchor_chunks)} anchors...")
        aggregated_results = []
        
        # Sort anchors by page and then by index (assuming order reflects position)
        sorted_anchors = sorted(anchor_chunks, key=lambda x: (x.get('page', 0), x.get('idx', 0)))
        
        for anchor in sorted_anchors:
            anchor_idx = anchor['idx']
            anchor_page = anchor['page']
            
            # Find preceding and succeeding chunks on the same page within the window
            related_indices = []
            # Look before
            for i in range(1, context_window + 1):
                prev_idx = anchor_idx - i
                if prev_idx >= 0 and self.chunks_data[prev_idx].get('page') == anchor_page:
                    related_indices.append(prev_idx)
            # Add anchor itself
            related_indices.append(anchor_idx)
            # Look after
            for i in range(1, context_window + 1):
                next_idx = anchor_idx + i
                if next_idx < len(self.chunks_data) and self.chunks_data[next_idx].get('page') == anchor_page:
                    related_indices.append(next_idx)
            
            # Get unique, sorted indices
            related_indices = sorted(list(set(related_indices)))
            
            # Aggregate text from related chunks
            aggregated_text_raw = " ".join([self.chunks_data[i].get('text_raw', '') for i in related_indices])
            aggregated_text_norm = " ".join([self.chunks_data[i].get('text_norm', '') for i in related_indices])
            
            # Create a new aggregated result object, keeping metadata from the anchor
            aggregated_result = anchor.copy()
            aggregated_result['text_raw'] = aggregated_text_raw
            aggregated_result['text_norm'] = aggregated_text_norm
            aggregated_result['aggregated_from_indices'] = related_indices # Meta-data
            aggregated_result['aggregated_from_count'] = len(related_indices)
            
            aggregated_results.append(aggregated_result)
            
        logger.info(f"Aggregation completed, resulting in {len(aggregated_results)} aggregated chunks.")
        return aggregated_results

    def display_results(self, results: List[Dict], query_text_raw: str, show_aggregated_info: bool = False):
        """Nicely formats and prints the search results."""
        print(f"\n--- Top {len(results)} Results for: '{query_text_raw}' ---")
        if not results:
            print("No results found.")
            return

        for i, res in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            # Print the most relevant score
            if 'rerank_score' in res:
                print(f"Rerank Score: {res['rerank_score']:.4f}")
            elif 'fused_score' in res:
                print(f"Hybrid (RRF) Score: {res['fused_score']:.4f}")
            else:
                print(f"{res['type'].title()} Score: {res.get('score', 'N/A'):.4f}")

            print(f"Text (Raw): {res.get('text_raw', 'N/A')}")
            # Optional: print text_norm if raw is not helpful for debugging
            # print(f"Text (Norm): {res.get('text_norm', 'N/A')}")
            print(f"Page: {res.get('page', 'N/A')}")
            print(f"Source Ref: {res.get('source_ref', 'N/A')}")
            
            # Show info about aggregation if requested and available
            if show_aggregated_info and 'aggregated_from_count' in res:
                 print(f"Aggregated from {res['aggregated_from_count']} chunks (indices: {res.get('aggregated_from_indices', [])})")

            print("-" * 40) # Longer separator for clarity

def main():
    """Main execution loop for interactive search."""
    try:
        # Initialize the pipeline (models, index, etc.)
        pipeline = EnhancedSemanticSearchPipeline(
            chunks_file_path=CHUNKS_FILE_PATH,
            embedding_model_name=EMBEDDING_MODEL_NAME,
            reranker_model_name=RERANKER_MODEL_NAME
        )
        print("\n" + "="*70)
        print("      ENHANCED Semantic Search is Ready!")
        print("      Features: Multi-Query, Targeted Pages, Authority Matrix Search.")
        print("="*70)
        print("Commands:")
        print("  - Type your question normally.")
        print("  - Type 'matrix' for specialized authority matrix search.")
        print("  - Type 'targeted <query>' to search only pages 122-124.")
        print("  - Type 'aggregate' after a search to combine context.")
        print("  - Type 'exit' or 'quit' to quit.")
        print("-" * 70)

        last_results = [] # Store last search results for potential aggregation

        while True:
            try:
                user_input = input("\nEnter command/query: ").strip()
                if not user_input:
                    print("Please enter a command or query.")
                    continue
                if user_input.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break

                query_raw = user_input
                use_aggregation = False

                # --- Handle Special Commands ---
                if user_input.lower() == 'matrix':
                    results = pipeline.search_complete_authority_matrix()
                    query_raw = "Specialized Authority Matrix Search"
                elif user_input.lower().startswith('targeted '):
                    actual_query = user_input[9:] # Remove 'targeted '
                    results = pipeline.search_targeted_pages(actual_query, TARGET_PAGES)
                    query_raw = f"Targeted Search: {actual_query}"
                elif user_input.lower() == 'aggregate':
                    if last_results:
                        print("Aggregating context for last search results...")
                        results = pipeline.aggregate_related_chunks(last_results)
                        query_raw = f"Aggregated results from last search"
                        use_aggregation = True # Flag for display
                    else:
                         print("No previous search results to aggregate. Please perform a search first.")
                         continue
                else:
                    # --- Standard Search ---
                    results = pipeline.search(
                        query_text_raw=query_raw,
                        use_hybrid=True,
                        use_reranker=True,
                        top_k_final=TOP_K_HYBRID_FUSED_ENHANCED,
                        top_k_reranked=TOP_K_RERANKED_ENHANCED
                    )
                
                # Store results for potential future aggregation
                last_results = results

                # Display results
                pipeline.display_results(results, query_raw, show_aggregated_info=use_aggregation)

            except KeyboardInterrupt:
                print("\n\nReceived interrupt signal. Exiting...")
                break
            except Exception as e:
                logger.error(f"An error occurred during search: {e}", exc_info=True) # Log full traceback
                print(f"\nAn error occurred: {e}. Please try again.")

    except Exception as e:
        logger.critical(f"Failed to initialize the search pipeline: {e}", exc_info=True)
        print(f"\nCritical Error: Could not start the search system. Reason: {e}")

if __name__ == "__main__":
    main()