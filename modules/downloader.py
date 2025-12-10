"""
YouTube Audio Downloader Module
Uses yt-dlp for reliable audio extraction
"""

import os
import logging
from typing import Tuple, Optional, Dict
import yt_dlp

logger = logging.getLogger(__name__)


class YouTubeDownloader:
    """Download audio from YouTube videos"""
    
    def __init__(self, download_dir: str = 'downloads'):
        """
        Initialize YouTube downloader
        
        Args:
            download_dir: Directory to save downloaded audio files
        """
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)
        
        # Configure yt-dlp options
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(self.download_dir, '%(id)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
            'keepvideo': False,
        }
    
    def download_audio(self, youtube_url: str) -> Tuple[Optional[str], Dict]:
        """
        Download audio from YouTube video
        
        Args:
            youtube_url: YouTube video URL
            
        Returns:
            Tuple of (audio_file_path, video_info_dict)
        """
        try:
            logger.info(f"Starting download from: {youtube_url}")
            
            # Extract video info first
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                video_info = {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'view_count': info.get('view_count', 0),
                    'video_id': info.get('id', 'unknown')
                }
            
            logger.info(f"Video Info: {video_info['title']} ({video_info['duration']}s)")
            
            # Download audio
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                ydl.download([youtube_url])
            
            # Construct the expected audio file path
            audio_path = os.path.join(self.download_dir, f"{video_info['video_id']}.wav")
            
            if os.path.exists(audio_path):
                logger.info(f"Audio downloaded successfully: {audio_path}")
                return audio_path, video_info
            else:
                logger.error(f"Audio file not found at expected path: {audio_path}")
                return None, video_info
        
        except Exception as e:
            logger.error(f"Error downloading audio: {str(e)}")
            return None, {'error': str(e)}
    
    def validate_url(self, url: str) -> bool:
        """
        Validate if URL is a valid YouTube URL
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                ydl.extract_info(url, download=False)
            return True
        except:
            return False