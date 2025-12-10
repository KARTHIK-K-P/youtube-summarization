"""
Text Summarizer Module
Uses BART/T5 for offline abstractive summarization with chunking
"""

import logging
from typing import Optional
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)


class TextSummarizer:
    """Summarize text using transformer models"""
    
    def __init__(self, model_name: str = 'facebook/bart-large-cnn', device: str = 'auto'):
        """
        Initialize text summarizer
        
        Args:
            model_name: HuggingFace model name
            device: Device to run model on (0 for GPU, -1 for CPU, 'auto' for auto-detect)
        """
        self.model_name = model_name
        
        # Auto-detect device
        if device == 'auto':
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device
        
        logger.info(f"Initializing summarization model: {model_name}")
        logger.info(f"Device: {'GPU' if self.device == 0 else 'CPU'}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Move model to device
            if self.device == 0:
                self.model = self.model.to('cuda')
            
            # Create pipeline
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
            )
            
            # Get model max length
            self.max_input_length = self.tokenizer.model_max_length
            if self.max_input_length > 100000:  # Some models return huge values
                self.max_input_length = 1024
            
            logger.info(f"Model loaded successfully! Max input length: {self.max_input_length}")
        
        except Exception as e:
            logger.error(f"Error loading summarization model: {str(e)}")
            raise
    
    def _chunk_text(self, text: str, max_chunk_size: int = 1024) -> list:
        """
        Split text into chunks that fit model's context window
        
        Args:
            text: Input text to chunk
            max_chunk_size: Maximum tokens per chunk
            
        Returns:
            List of text chunks
        """
        # Split by sentences (simple approach)
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # Rough estimate: 1 word ≈ 1.3 tokens
            sentence_length = len(sentence.split()) * 1.3
            
            if current_length + sentence_length > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def summarize(self, text: str, max_length: int = 300, min_length: int = 100) -> Optional[str]:
        """
        Summarize input text with automatic chunking for long texts
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary in tokens
            min_length: Minimum length of summary in tokens
            
        Returns:
            Summary text or None if failed
        """
        try:
            if not text or len(text.strip()) == 0:
                logger.error("Empty text provided for summarization")
                return None
            
            logger.info(f"Starting summarization of text ({len(text)} characters)...")
            
            # Count tokens in input
            input_tokens = len(self.tokenizer.encode(text, truncation=False))
            logger.info(f"Input tokens: {input_tokens}")
            
            # If text is too long, use chunking strategy
            if input_tokens > self.max_input_length - 100:
                logger.info(f"Text too long ({input_tokens} tokens), using chunking strategy...")
                return self._summarize_long_text(text, max_length, min_length)
            
            # Direct summarization for shorter texts
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            
            result = summary[0]['summary_text']
            logger.info(f"Summary generated: {len(result)} characters")
            
            return result
        
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            return None
    
    def _summarize_long_text(self, text: str, max_length: int = 300, min_length: int = 100) -> Optional[str]:
        """
        Summarize long text using hierarchical chunking approach
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of final summary
            min_length: Minimum length of final summary
            
        Returns:
            Summary text or None if failed
        """
        try:
            # Step 1: Split text into chunks
            chunks = self._chunk_text(text, max_chunk_size=900)
            logger.info(f"Split text into {len(chunks)} chunks")
            
            # Step 2: Summarize each chunk
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Summarizing chunk {i+1}/{len(chunks)}...")
                
                summary = self.summarizer(
                    chunk,
                    max_length=150,
                    min_length=50,
                    do_sample=False,
                    truncation=True
                )
                
                chunk_summaries.append(summary[0]['summary_text'])
            
            # Step 3: Combine chunk summaries
            combined_summary = " ".join(chunk_summaries)
            logger.info(f"Combined chunk summaries: {len(combined_summary)} characters")
            
            # Step 4: Final summarization if combined is still too long
            combined_tokens = len(self.tokenizer.encode(combined_summary, truncation=False))
            
            if combined_tokens > self.max_input_length - 100:
                logger.info("Combined summary still too long, performing final summarization...")
                # Recursively summarize
                return self._summarize_long_text(combined_summary, max_length, min_length)
            else:
                # Final summarization
                final_summary = self.summarizer(
                    combined_summary,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True
                )
                
                result = final_summary[0]['summary_text']
                logger.info(f"Final summary generated: {len(result)} characters")
                
                return result
        
        except Exception as e:
            logger.error(f"Error during long text summarization: {str(e)}")
            return None