"""
Audio Transcriber Module
Uses Faster-Whisper for offline speech-to-text
"""

import os
import logging
from typing import Optional
from faster_whisper import WhisperModel
import torch

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """Transcribe audio to text using Whisper model"""
    
    def __init__(self, model_size: str = 'medium', device: str = 'auto'):
        """
        Initialize audio transcriber with Whisper model
        
        Args:
            model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run model on ('cpu', 'cuda', or 'auto')
        """
        self.model_size = model_size
        
        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Set compute type based on device
        if self.device == 'cuda':
            self.compute_type = 'float16'
        else:
            self.compute_type = 'int8'  # Quantized for CPU
        
        logger.info(f"Initializing Whisper model: {model_size} on {self.device}")
        logger.info(f"Compute type: {self.compute_type}")
        
        # Initialize Faster-Whisper model
        try:
            self.model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=os.path.join(os.getcwd(), 'models', 'whisper')
            )
            logger.info("Whisper model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            raise
    
    def transcribe(self, audio_path: str, language: str = None) -> Optional[str]:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es'). None for auto-detection
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return None
            
            logger.info(f"Starting transcription of: {audio_path}")
            logger.info("This may take several minutes depending on video length...")
            
            # Transcribe with Whisper
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                vad_filter=True,  # Voice Activity Detection for better accuracy
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    max_speech_duration_s=float('inf'),
                    min_silence_duration_ms=2000,
                    window_size_samples=1024,
                    speech_pad_ms=400
                )
            )
            
            # Log detected language
            logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
            
            # Collect all segments into full transcript
            transcript_parts = []
            for segment in segments:
                transcript_parts.append(segment.text.strip())
            
            full_transcript = " ".join(transcript_parts)
            
            logger.info(f"Transcription completed! Length: {len(full_transcript)} characters")
            
            return full_transcript
        
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            return None
    
    def transcribe_with_timestamps(self, audio_path: str) -> Optional[list]:
        """
        Transcribe audio with timestamps for each segment
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of segments with timestamps or None if failed
        """
        try:
            segments, info = self.model.transcribe(audio_path, beam_size=5)
            
            result = []
            for segment in segments:
                result.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip()
                })
            
            return result
        
        except Exception as e:
            logger.error(f"Error during transcription with timestamps: {str(e)}")
            return None