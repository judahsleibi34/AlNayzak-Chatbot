# ==================== PATCH FOR BITSANDBYTES METADATA ISSUE ====================
import importlib.metadata

_original_version = importlib.metadata.version

def patched_version(package_name: str) -> str:
    if package_name == "bitsandbytes":
        return "0.47.0"  # or whatever version you installed
    return _original_version(package_name)

importlib.metadata.version = patched_version
# ==============================================================================

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
    BitsAndBytesConfig
)
import warnings
import os
from pathlib import Path
import pickle
import time
from collections import defaultdict

warnings.filterwarnings("ignore")

@dataclass
class DocumentChunk:
    id: str
    text_display: str
    text_embed: str
    language: str
    source: str
    page: int
    chunk_no: int
    embedding: Optional[np.ndarray] = None

class OptimizedArabicSemanticRetriever:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        print("🔄 Loading embedding model...")
        start_time = time.time()
        
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 256
        
        print(f"✅ Model loaded in {time.time() - start_time:.2f} seconds")
        
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.cache_dir = Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def preprocess_arabic_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)
        return text.strip()
    
    def sliding_window_chunk(self, text: str, window_size: int = 700, overlap: int = 180) -> List[str]:
        if len(text) <= window_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + window_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start += (window_size - overlap)
        return chunks

    def load_documents(self, json_data: List[Dict]) -> None:  # ✅ FIXED SIGNATURE
        print(f"📚 Loading {len(json_data)} documents with sliding window chunking...")
        self.chunks = []
        
        for i, item in enumerate(json_data):
            if i % 1000 == 0:
                print(f"  Processing document {i}/{len(json_data)}")
                
            full_text = item.get('text_display', '')
            if not full_text.strip():
                continue
                
            chunks = self.sliding_window_chunk(full_text, window_size=700, overlap=180)
            
            for j, chunk_text in enumerate(chunks):
                embed_text = self.preprocess_arabic_text(chunk_text)
                if embed_text.strip():
                    chunk = DocumentChunk(
                        id=f"{item['id']}_chunk{j}",
                        text_display=chunk_text,
                        text_embed=embed_text,
                        language=item.get('language', 'ar'),
                        source=item.get('source', 'unknown'),
                        page=item.get('page', 0),
                        chunk_no=j
                    )
                    self.chunks.append(chunk)
        
        print(f"✅ Loaded {len(self.chunks)} valid chunks (after sliding window)")
    
    def _get_cache_path(self, data_hash: str) -> Path:
        return self.cache_dir / f"embeddings_{data_hash}.pkl"
    
    def _get_data_hash(self) -> str:
        texts = [chunk.text_embed for chunk in self.chunks[:100]]
        data_str = ''.join(texts)
        return str(hash(data_str))
    
    def build_embeddings(self, use_cache: bool = True, batch_size: int = 128) -> None:
        if not self.chunks:
            raise ValueError("No documents loaded.")
        
        data_hash = self._get_data_hash()
        cache_path = self._get_cache_path(data_hash)
        
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
        
        print("🔄 Building embeddings...")
        start_time = time.time()
        
        texts = [chunk.text_embed for chunk in self.chunks]
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
        
        for i, chunk in enumerate(self.chunks):
            chunk.embedding = self.embeddings[i]
        
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
        dimension = self.embeddings.shape[1]
        
        if len(self.embeddings) < 10000:
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(self.embeddings.astype('float32'))
        else:
            nlist = min(100, len(self.embeddings) // 100)
            quantizer = faiss.IndexFlatIP(dimension)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.faiss_index.train(self.embeddings.astype('float32'))
            self.faiss_index.add(self.embeddings.astype('float32'))
    
    def semantic_search(self, query: str, top_k: int = 20) -> List[Tuple[DocumentChunk, float]]:
        processed_query = self.preprocess_arabic_text(query)
        query_embedding = self.model.encode([processed_query], normalize_embeddings=True)
        
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results

    def rerank_with_cross_encoder(self, query: str, candidates: List[Tuple[DocumentChunk, float]], top_k_final: int = 5) -> List[Tuple[DocumentChunk, float]]:
        print("🔄 Reranking top 20 → 5 with cross-encoder (stub)...")
        return sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k_final]
    
    def get_relevant_context(self, query: str, max_context_length: int = 1500) -> Tuple[str, List[Dict], float]:
        faiss_results = self.semantic_search(query, top_k=20)
        reranked_results = self.rerank_with_cross_encoder(query, faiss_results, top_k_final=5)
        
        context_parts = []
        metadata = []
        total_length = 0
        total_score = 0.0
        used_chunks = 0
        
        for i, (chunk, score) in enumerate(reranked_results):
            chunk_text = chunk.text_display.strip()
            source_info = f"[المصدر: {chunk.source} - ص{chunk.page}]"
            
            proposed_length = total_length + len(chunk_text) + len(source_info) + 2
            
            if proposed_length <= max_context_length:
                context_parts.append(f"{source_info}\n{chunk_text}")
                metadata.append({
                    "source": chunk.source,
                    "page": chunk.page,
                    "score": score,
                    "chunk_id": chunk.id
                })
                total_length = proposed_length
                total_score += score
                used_chunks += 1
            else:
                remaining = max_context_length - total_length - len(source_info) - 2
                if remaining > 200:
                    truncated_text = chunk_text[:remaining-50] + " ... [مقطع مختصر]"
                    context_parts.append(f"{source_info}\n{truncated_text}")
                    metadata.append({
                        "source": chunk.source,
                        "page": chunk.page,
                        "score": score,
                        "chunk_id": chunk.id,
                        "truncated": True
                    })
                    total_length += len(truncated_text) + len(source_info) + 2
                    total_score += score * 0.9
                    used_chunks += 1
                break
        
        avg_score = total_score / used_chunks if used_chunks > 0 else 0.0
        context = "\n\n".join(context_parts)
        
        return context, metadata, avg_score

class OptimizedSambaLingoRAG:
    def __init__(self, jsonl_filepath: str, use_cache: bool = True):
        print("🚀 Initializing Enhanced Arabic RAG System...")
        start_time = time.time()
        
        print("📚 Setting up semantic retriever...")
        self.retriever = self._setup_retriever(jsonl_filepath, use_cache)
        
        print("🧠 Loading SambaLingo Arabic model...")
        self.tokenizer, self.model = self._setup_sambanova_model()
        
        total_time = time.time() - start_time
        print(f"✅ RAG System initialized in {total_time:.2f} seconds!")
    
    def _load_jsonl_fast(self, filepath: str) -> List[Dict]:
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
                        if line_num <= 10:
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
        data = self._load_jsonl_fast(jsonl_filepath)
        if not data:  # ✅ FIXED CONDITION
            raise ValueError("No data loaded from JSONL file")
        
        retriever = OptimizedArabicSemanticRetriever()
        retriever.load_documents(data)
        retriever.build_embeddings(use_cache=use_cache, batch_size=128)
        return retriever
    
    def _setup_sambanova_model(self):
        model_name = "sambanovasystems/SambaLingo-Arabic-Chat"
        
        # ✅ SAFETY CHECK FOR GPU
        if not torch.cuda.is_available():
            print("⚠️ No GPU detected. Loading model on CPU may be very slow.")
        
        # ✅ TRY BNB CONFIG WITH FALLBACK
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True
            )
            use_bnb = True
        except Exception:
            print("⚠️ bitsandbytes not available, falling back to non-quantized load.")
            bnb_config, use_bnb = None, False
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "use_cache": True,
        }
        
        if use_bnb:
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.bfloat16
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        model.eval()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        print(f"📊 Model loaded on: {model.device}")
        memory_gb = model.get_memory_footprint() / 1024**3 if hasattr(model, 'get_memory_footprint') else 0
        print(f"💾 Model memory footprint: ~{memory_gb:.1f}GB")
        
        return tokenizer, model
    
    def _create_structured_prompt(self, question: str, context: str) -> str:
        """SIMPLE, CLEAR PROMPT THAT WORKS"""
        prompt = f"""الإجابة على السؤال التالي بناءً على السياق المتوفر:

السياق:
{context}

السؤال: {question}

أجب بتنسيق عربي منظم يحتوي على:
- النطاق:
- الأهلية:
- الخطوات:
- المستندات:
- الاستثناءات:
- ملخص تنفيذي:
- ثغرات: (إن وجدت)

استخدم النقاط واضحة. اذكر المصدر [صX] عند نهاية كل قسم.
"""
        return prompt

    def _compute_confidence(self, similarity_score: float, context_metadata: List[Dict], generated_answer: str) -> Dict:  # ✅ FIXED SIGNATURE
        signals = {}
        signals['similarity'] = min(max(similarity_score, 0.0), 1.0)
        
        required_sections = ["النطاق", "الأهلية", "الخطوات", "المستندات", "الاستثناءات", "ملخص تنفيذي"]
        found_sections = sum(1 for s in required_sections if s in generated_answer)  # ✅ CLEANER
        signals['coverage'] = found_sections / len(required_sections) if required_sections else 1.0
        
        mentioned_pages = re.findall(r'\[ص(\d+)\]', generated_answer)
        valid_pages = {str(meta['page']) for meta in context_metadata}
        if mentioned_pages:
            valid_mentions = sum(1 for p in mentioned_pages if p in valid_pages)
            signals['faithfulness'] = valid_mentions / len(mentioned_pages)
        else:
            signals['faithfulness'] = 1.0
        
        weights = {'similarity': 0.4, 'coverage': 0.3, 'faithfulness': 0.3}
        confidence = sum(weights[k] * signals[k] for k in weights)  # ✅ CLEANER
        
        return {
            "confidence": confidence,
            "signals": signals,
            "coverage_score": signals['coverage']
        }
    
    def _post_process_answer(self, answer: str, prompt: str) -> str:
        """Remove prompt echo and clean up"""
        # Remove the prompt from the beginning
        if answer.startswith(prompt):
            answer = answer[len(prompt):].strip()
        
        # Remove any remaining instruction echoes
        answer = re.sub(r'^.*?أنت مساعد ذكي.*?\n', '', answer, flags=re.DOTALL)
        answer = re.sub(r'^.*?الإجابة على.*?:', '', answer, flags=re.DOTALL)
        
        # Clean artifacts
        answer = answer.replace("<|end_header_id|>", "").strip()
        
        # Ensure structure
        if "ملخص تنفيذي" not in answer:
            answer += "\n\nملخص تنفيذي: لم يتم العثور على معلومات كافية."
        
        return answer.strip()
    
    def generate_answer(self, question: str, max_length: int = 500, temperature: float = 0.3, 
                       confidence_threshold: float = 0.5) -> Dict:
        start_time = time.time()
        
        print("🔍 Retrieving context...")
        context, metadata, avg_similarity = self.retriever.get_relevant_context(question, max_context_length=1500)
        
        if not context.strip():
            return {
                "answer": "لا توجد معلومات كافية في السياق المتوفر للإجابة عن هذا السؤال.",
                "context": "",
                "metadata": [],
                "confidence": 0.0,
                "signals": {},
                "coverage_score": 0.0,
                "time_taken": time.time() - start_time,
                "was_truncated": False
            }
        
        prompt = self._create_structured_prompt(question, context)
        
        print("🧠 Generating answer...")
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1500,
                padding=False
            ).to(self.model.device)
            
            generate_kwargs = {
                "max_new_tokens": max_length,
                "do_sample": True,
                "temperature": temperature,
                "top_p": 0.92,
                "repetition_penalty": 1.15,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generate_kwargs)
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract and clean answer
            answer = self._post_process_answer(full_response, prompt)
            
            # Compute confidence
            confidence_data = self._compute_confidence(avg_similarity, metadata, answer)
            
            # Confidence-based fallback
            if confidence_data["confidence"] < confidence_threshold or confidence_data["coverage_score"] < 0.6:
                fallback = (
                    "لا يمكن تقديم إجابة دقيقة حاليًا بسبب محدودية السياق.\n\n"
                    "ملخص تنفيذي: غير متوفر بسبب نقص المعلومات.\n"
                    "ثغرات: لم يتم العثور على معلومات كافية حول هذا الموضوع."
                )
                answer = fallback
                confidence_data["confidence"] = 0.3
            
            time_taken = time.time() - start_time
            print(f"⚡ Answer generated in {time_taken:.2f} seconds | Confidence: {confidence_data['confidence']:.2f}")
            
            return {
                "answer": answer,
                "context": context,
                "metadata": metadata,
                "confidence": confidence_data["confidence"],
                "signals": confidence_data["signals"],
                "coverage_score": confidence_data["coverage_score"],
                "time_taken": time_taken,
                "was_truncated": any(m.get('truncated', False) for m in metadata)
            }
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return {
                "answer": "حدث خطأ أثناء معالجة سؤالك. يرجى المحاولة مرة أخرى.",
                "context": context,
                "metadata": metadata,
                "confidence": 0.0,
                "signals": {},
                "coverage_score": 0.0,
                "time_taken": time.time() - start_time,
                "was_truncated": any(m.get('truncated', False) for m in metadata)
            }
    
    def ask(self, question: str) -> str:
        result = self.generate_answer(question)
        return result["answer"]
    
    def benchmark(self, questions: List[str]) -> None:
        print(f"\n🧪 Running benchmark with your questions...")
        print("="*80)
        
        total_time = 0
        total_confidence = 0
        total_coverage = 0
        
        for i, question in enumerate(questions, 1):
            print(f"\n{i}. سؤال: {question}")
            result = self.generate_answer(question)
            print(f"⏱️ الوقت: {result['time_taken']:.2f}s | 🎯 الثقة: {result['confidence']:.2f} | 📊 التغطية: {result['coverage_score']:.2f}")
            
            # Show first 400 chars of answer
            answer_preview = result['answer'][:400]
            if len(result['answer']) > 400:
                answer_preview += "..."
            print(f"📄 الإجابة: {answer_preview}")
            
            if result['was_truncated']:
                print("⚠️  ملاحظة: تم اقتطاع جزء من السياق")
            total_time += result['time_taken']
            total_confidence += result['confidence']
            total_coverage += result['coverage_score']
            print("-"*80)
        
        n = len(questions)
        avg_time = total_time / n
        avg_conf = total_confidence / n
        avg_cov = total_coverage / n
        
        print(f"\n📊 متوسط الوقت: {avg_time:.2f}s | متوسط الثقة: {avg_conf:.2f} | متوسط التغطية: {avg_cov:.2f}")
        print(f"📊 الإجمالي: {total_time:.2f}s")
    
    def chat_loop(self):
        print("\n" + "="*80)
        print("🤖 مرحباً! أنا مساعدك الذكي")
        print("   - 'خروج' للانتهاء")
        print("="*80 + "\n")
        
        while True:
            try:
                question = input("🙋 سؤالك: ").strip()
                
                if question.lower() in ['خروج', 'exit', 'quit']:
                    print("🙏 شكراً!")
                    break
                
                if not question:
                    continue
                
                print()
                result = self.generate_answer(question)
                print(f"🤖 {result['answer']}")
                print(f"\n📈 الثقة: {result['confidence']:.2f}")
                print()
                
            except KeyboardInterrupt:
                print("\n👋 تم الإيقاف")
                break
            except Exception as e:
                print(f"❌ خطأ: {e}")
    
    def cleanup(self):
        print("🧹 تنظيف الذاكرة...")
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'retriever'):
            del self.retriever
        
        torch.cuda.empty_cache()
        print("✅ تم تنظيف الذاكرة")

# Main execution with your questions
if __name__ == "__main__":
    try:
        rag_system = OptimizedSambaLingoRAG("arabic_chatbot_knowledge.jsonl", use_cache=True)
        
        # Your test questions
        test_questions = [
            "ما هي المستندات الرسمية التي يجب على الموظف الجديد تقديمها عند التعيين في مؤسسة النيزك؟",
            "ما هي شروط الأهلية الأساسية للتقديم على الوظائف في مؤسسة النيزك؟",
            "ما هي الخطوات الإجرائية التي تمر بها عملية التوظيف داخل مؤسسة النيزك بدءًا من الإعلان وحتى توقيع العقد؟",
            "ما هو نطاق سياسة التوظيف في مؤسسة النيزك؟ وهل تشمل الموظفين المؤقتين والمستشارين أم فقط العاملين الدائمين؟",
            "هل توجد استثناءات في سياسة التوظيف بمؤسسة النيزك مثل تعيين خبراء بعقود خاصة أو لمهام محددة؟"
        ]
        
        print("🧪 Running benchmark with your detailed questions...")
        rag_system.benchmark(test_questions)
        
        print("\n💬 Starting chat...")
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
