import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
import warnings
import os
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor
import time

warnings.filterwarnings("ignore")

@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata"""
    id: str
    text_display: str
    text_embed: str
    language: str
    source: str
    page: int
    chunk_no: int
    embedding: Optional[np.ndarray] = None

class OptimizedArabicSemanticRetriever:
    """
    High-performance semantic retrieval system optimized for Arabic text
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """Initialize with a faster multilingual model"""
        print("🔄 Loading embedding model...")
        start_time = time.time()
        
        # Use a faster multilingual model that works well with Arabic
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 256  # Reduce sequence length for speed
        
        print(f"✅ Model loaded in {time.time() - start_time:.2f} seconds")
        
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.cache_dir = Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def preprocess_arabic_text(self, text: str) -> str:
        """Optimized Arabic text preprocessing"""
        if not text:
            return ""
        
        # Basic cleaning - simplified for speed
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)  # Remove diacritics
        return text.strip()
    
    def load_documents(self, json_data: List[Dict]) -> None:
        """Load documents from JSON data with progress tracking"""
        print(f"📚 Loading {len(json_data)} documents...")
        self.chunks = []
        
        for i, item in enumerate(json_data):
            if i % 1000 == 0:
                print(f"  Processing document {i}/{len(json_data)}")
                
            embed_text = self.preprocess_arabic_text(item.get('text_embed', ''))
            display_text = item.get('text_display', embed_text)
            
            if embed_text.strip():
                chunk = DocumentChunk(
                    id=item['id'],
                    text_display=display_text,
                    text_embed=embed_text,
                    language=item['language'],
                    source=item['source'],
                    page=item['page'],
                    chunk_no=item['chunk_no']
                )
                self.chunks.append(chunk)
        
        print(f"✅ Loaded {len(self.chunks)} valid chunks")
    
    def _get_cache_path(self, data_hash: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"embeddings_{data_hash}.pkl"
    
    def _get_data_hash(self) -> str:
        """Generate hash for current data"""
        texts = [chunk.text_embed for chunk in self.chunks]
        data_str = ''.join(texts[:100])  # Use first 100 for hash
        return str(hash(data_str))
    
    def build_embeddings(self, use_cache: bool = True, batch_size: int = 64) -> None:
        """Build embeddings with caching and batch processing"""
        if not self.chunks:
            raise ValueError("No documents loaded.")
        
        data_hash = self._get_data_hash()
        cache_path = self._get_cache_path(data_hash)
        
        # Try to load from cache
        if use_cache and cache_path.exists():
            print("📦 Loading embeddings from cache...")
            try:
                with open(cache_path, 'rb') as f:
                    self.embeddings = pickle.load(f)
                
                for i, chunk in enumerate(self.chunks):
                    if i < len(self.embeddings):
                        chunk.embedding = self.embeddings[i]
                
                self.build_faiss_index()
                print(f"✅ Loaded cached embeddings for {len(self.chunks)} chunks")
                return
            except Exception as e:
                print(f"⚠️ Cache loading failed: {e}, rebuilding...")
        
        # Build embeddings with progress tracking
        print("🔄 Building embeddings...")
        start_time = time.time()
        
        texts = [chunk.text_embed for chunk in self.chunks]
        
        # Process in batches to show progress and manage memory
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch_texts, 
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            all_embeddings.append(batch_embeddings)
            
            current_batch = (i // batch_size) + 1
            print(f"  Batch {current_batch}/{total_batches} processed")
        
        self.embeddings = np.vstack(all_embeddings)
        
        # Assign embeddings to chunks
        for i, chunk in enumerate(self.chunks):
            chunk.embedding = self.embeddings[i]
        
        # Cache the embeddings
        if use_cache:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.embeddings, f)
                print("💾 Embeddings cached for future use")
            except Exception as e:
                print(f"⚠️ Failed to cache embeddings: {e}")
        
        self.build_faiss_index()
        elapsed = time.time() - start_time
        print(f"✅ Built embeddings for {len(self.chunks)} chunks in {elapsed:.2f} seconds")
    
    def build_faiss_index(self) -> None:
        """Build optimized FAISS index"""
        dimension = self.embeddings.shape[1]
        
        # Use IndexFlatIP for smaller datasets, or IndexIVFFlat for larger ones
        if len(self.embeddings) < 10000:
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(self.embeddings.astype('float32'))
        else:
            # For larger datasets, use IVF index for faster search
            nlist = min(100, len(self.embeddings) // 100)  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.faiss_index.train(self.embeddings.astype('float32'))
            self.faiss_index.add(self.embeddings.astype('float32'))
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Optimized semantic search"""
        processed_query = self.preprocess_arabic_text(query)
        query_embedding = self.model.encode([processed_query], normalize_embeddings=True)
        
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def get_relevant_context(self, query: str, max_context_length: int = 1500) -> str:
        """Get relevant context for RAG with better formatting"""
        results = self.semantic_search(query, top_k=3)
        
        if not results:
            return ""
        
        context_parts = []
        total_length = 0
        
        for i, (chunk, score) in enumerate(results):
            chunk_text = chunk.text_display
            source_info = f"[المصدر {i+1}: {chunk.source} - ص{chunk.page}]"
            
            if total_length + len(chunk_text) + len(source_info) <= max_context_length:
                context_parts.append(f"{source_info}\n{chunk_text}")
                total_length += len(chunk_text) + len(source_info)
            else:
                remaining_space = max_context_length - total_length - len(source_info)
                if remaining_space > 100:
                    truncated = chunk_text[:remaining_space-20] + "..."
                    context_parts.append(f"{source_info}\n{truncated}")
                break
        
        return "\n\n".join(context_parts)

class OptimizedSambaLingoRAG:
    """
    Optimized RAG system with faster initialization and processing
    """
    
    def __init__(self, jsonl_filepath: str, use_cache: bool = True):
        """
        Initialize RAG system with optimization flags
        
        Args:
            jsonl_filepath: Path to JSONL knowledge file
            use_cache: Whether to use embedding cache
        """
        print("🚀 Initializing Optimized Arabic RAG System...")
        start_time = time.time()
        
        # Initialize retriever
        print("📚 Setting up semantic retriever...")
        self.retriever = self._setup_retriever(jsonl_filepath, use_cache)
        
        # Initialize SambaLingo model with optimizations
        print("🧠 Loading SambaLingo Arabic model...")
        self.tokenizer, self.model = self._setup_sambanova_model()
        
        total_time = time.time() - start_time
        print(f"✅ RAG System initialized in {total_time:.2f} seconds!")
    
    def _load_jsonl_fast(self, filepath: str) -> List[Dict]:
        """Optimized JSONL loading with better error handling"""
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                
            print(f"📖 Processing {len(lines)} lines...")
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        if line_num <= 10:  # Only show first 10 errors
                            print(f"⚠️ Error parsing line {line_num}: {e}")
                
                if line_num % 5000 == 0:
                    print(f"  Processed {line_num}/{len(lines)} lines")
            
            print(f"✅ Loaded {len(data)} valid chunks from {filepath}")
            return data
        except FileNotFoundError:
            print(f"❌ File {filepath} not found!")
            return []
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return []
    
    def _setup_retriever(self, jsonl_filepath: str, use_cache: bool) -> OptimizedArabicSemanticRetriever:
        """Setup optimized semantic retriever"""
        data = self._load_jsonl_fast(jsonl_filepath)
        if not data:
            raise ValueError("No data loaded from JSONL file")
        
        retriever = OptimizedArabicSemanticRetriever()
        retriever.load_documents(data)
        retriever.build_embeddings(use_cache=use_cache, batch_size=128)
        return retriever
    
    def _setup_sambanova_model(self):
        """Setup SambaLingo model with aggressive optimization"""
        model_name = "sambanovasystems/SambaLingo-Arabic-Chat"
        
        # More aggressive quantization for RTX 4060
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with memory optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_cache=True
        )
        
        # Enable evaluation mode and optimizations
        model.eval()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Compile model for faster inference (PyTorch 2.0+)
        try:
            if torch.__version__ >= "2.0":
                model = torch.compile(model, mode="reduce-overhead")
                print("🚀 Model compiled for faster inference")
        except Exception as e:
            print(f"⚠️ Model compilation failed: {e}")
        
        print(f"📊 Model loaded on: {model.device}")
        memory_gb = model.get_memory_footprint() / 1024**3 if hasattr(model, 'get_memory_footprint') else 0
        print(f"💾 Model memory footprint: ~{memory_gb:.1f}GB")
        
        return tokenizer, model
    
    def _create_optimized_prompt(self, question: str, context: str) -> str:
        """Create optimized prompt that's shorter and more focused"""
        return f"""<|start_header_id|>system<|end_header_id|>
أجب على السؤال باللغة العربية بناءً على السياق المعطى. كن مختصراً ودقيقاً.<|end_header_id|>

<|start_header_id|>user<|end_header_id|>
السياق: {context}

السؤال: {question}<|end_header_id|>

<|start_header_id|>assistant<|end_header_id|>"""
    
    def generate_answer(self, question: str, max_length: int = 256, temperature: float = 0.3) -> Dict:
        """
        Optimized answer generation
        
        Args:
            question: Question in Arabic
            max_length: Maximum generation length (reduced default)
            temperature: Generation temperature (reduced for consistency)
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        # Step 1: Retrieve relevant context (faster)
        print("🔍 Retrieving context...")
        context = self.retriever.get_relevant_context(question, max_context_length=1200)
        
        if not context.strip():
            return {
                "answer": "عذراً، لم أجد معلومات ذات صلة بسؤالك في قاعدة المعرفة المتاحة.",
                "context": "",
                "confidence": 0.0,
                "time_taken": time.time() - start_time
            }
        
        # Step 2: Create optimized prompt
        prompt = self._create_optimized_prompt(question, context)
        
        # Step 3: Generate answer with optimizations
        print("🧠 Generating answer...")
        try:
            # Tokenize with length limits
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1500,  # Reduced for speed
                padding=False
            ).to(self.model.device)
            
            # Generate with optimized parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.8,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True,
                    use_cache=True
                )
            
            # Quick decode
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer efficiently
            if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
                answer = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            else:
                answer = full_response[len(prompt):].strip()
            
            # Clean up
            answer = answer.replace("<|end_header_id|>", "").strip()
            
            time_taken = time.time() - start_time
            print(f"⚡ Answer generated in {time_taken:.2f} seconds")
            
            return {
                "answer": answer,
                "context": context,
                "confidence": 0.8,
                "time_taken": time_taken
            }
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return {
                "answer": "حدث خطأ أثناء معالجة سؤالك. يرجى المحاولة مرة أخرى.",
                "context": context,
                "confidence": 0.0,
                "time_taken": time.time() - start_time
            }
    
    def ask(self, question: str) -> str:
        """Simple interface to ask questions"""
        result = self.generate_answer(question)
        return result["answer"]
    
    def benchmark(self, questions: List[str]) -> None:
        """Run benchmark tests"""
        print("\n🧪 Running benchmark tests...")
        print("="*60)
        
        total_time = 0
        for i, question in enumerate(questions, 1):
            print(f"\n{i}. سؤال: {question}")
            result = self.generate_answer(question)
            print(f"الوقت المستغرق: {result['time_taken']:.2f} ثانية")
            print(f"الإجابة: {result['answer'][:100]}...")
            total_time += result['time_taken']
            print("-"*40)
        
        avg_time = total_time / len(questions)
        print(f"\n📊 متوسط الوقت لكل سؤال: {avg_time:.2f} ثانية")
        print(f"📊 إجمالي الوقت: {total_time:.2f} ثانية")
    
    def chat_loop(self):
        """Optimized interactive chat loop"""
        print("\n" + "="*60)
        print("🤖 مرحباً! أنا مساعدك الذكي المحسّن")
        print("💡 اكتب 'خروج' للانتهاء")
        print("="*60 + "\n")
        
        while True:
            try:
                question = input("🙋 سؤالك: ").strip()
                
                if question.lower() in ['خروج', 'exit', 'quit']:
                    print("🙏 شكراً!")
                    break
                
                if not question:
                    continue
                
                print()
                answer = self.ask(question)
                print(f"🤖 {answer}")
                print()
                
            except KeyboardInterrupt:
                print("\n👋 تم الإيقاف")
                break
            except Exception as e:
                print(f"❌ خطأ: {e}")
    
    def cleanup(self):
        """Enhanced cleanup"""
        print("🧹 تنظيف الذاكرة...")
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'retriever'):
            del self.retriever
        
        torch.cuda.empty_cache()
        print("✅ تم تنظيف الذاكرة")

# Optimized main execution
if __name__ == "__main__":
    try:
        # Initialize with caching enabled
        rag_system = OptimizedSambaLingoRAG("arabic_chatbot_knowledge.jsonl", use_cache=True)
        
        # Quick benchmark
        test_questions = [
            "ما هي سياسات التوظيف في مؤسسة النيزك؟",
            "كيف يمكن التقدم للوظائف؟",
            "ما هي الخدمات المتاحة؟"
        ]
        
        rag_system.benchmark(test_questions)
        
        # Debug: Test a single question with detailed output
        print("\n🔧 Debug Mode - Single Question Test:")
        print("="*60)
        
        # First, let's see what's in the knowledge base
        print("📚 Testing knowledge base search...")
        test_results = rag_system.retriever.semantic_search("التوظيف", top_k=3)
        print(f"Found {len(test_results)} results for 'التوظيف':")
        for i, (chunk, score) in enumerate(test_results):
            print(f"  {i+1}. Score: {score:.3f} - Source: {chunk.source} - Text: {chunk.text_display[:100]}...")
        
        print("\n🔧 Full Answer Generation Test:")
        test_result = rag_system.generate_answer("ما هي سياسات التوظيف؟")
        print(f"Context found: {len(test_result['context'])} characters")
        print(f"Answer: '{test_result['answer']}'")
        print(f"Confidence: {test_result['confidence']}")
        print("="*60)
        # Start chat
        rag_system.chat_loop()
        
    except KeyboardInterrupt:
        print("\n👋 تم إيقاف النظام")
    except Exception as e:
        print(f"❌ خطأ في النظام: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'rag_system' in locals():
            rag_system.cleanup()