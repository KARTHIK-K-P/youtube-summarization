
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import logging
from datetime import datetime
import traceback

# Import our custom modules
from modules.downloader import YouTubeDownloader
from modules.transcriber import AudioTranscriber
from modules.summarizer import TextSummarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Create necessary directories
DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Initialize components (lazy loading for first request)
downloader = None
transcriber = None
summarizer = None


def initialize_models():
    """Initialize AI models on first request"""
    global downloader, transcriber, summarizer
    
    if downloader is None:
        logger.info("Initializing YouTube Downloader...")
        downloader = YouTubeDownloader(download_dir=DOWNLOAD_DIR)
    
    if transcriber is None:
        logger.info("Initializing Speech-to-Text Model (Whisper)...")
        transcriber = AudioTranscriber(model_size='medium')
    
    if summarizer is None:
        logger.info("Initializing Summarization Model (BART)...")
        summarizer = TextSummarizer(model_name='facebook/bart-large-cnn')
    
    logger.info("All models initialized successfully!")


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/summarize', methods=['POST'])
def summarize():
    """Main endpoint to process YouTube video"""
    try:
        # Initialize models if not already done
        initialize_models()
        
        # Get YouTube URL from request
        data = request.get_json()
        youtube_url = data.get('url', '').strip()
        
        if not youtube_url:
            return jsonify({
                'success': False,
                'error': 'Please provide a YouTube URL'
            }), 400
        
        logger.info(f"Processing YouTube URL: {youtube_url}")
        
        # Step 1: Download audio
        logger.info("Step 1/3: Downloading audio from YouTube...")
        audio_path, video_info = downloader.download_audio(youtube_url)
        
        if not audio_path:
            return jsonify({
                'success': False,
                'error': 'Failed to download audio from YouTube'
            }), 400
        
        logger.info(f"Audio downloaded successfully: {audio_path}")
        
        # Step 2: Transcribe audio
        logger.info("Step 2/3: Transcribing audio to text (this may take a while)...")
        transcript = transcriber.transcribe(audio_path)
        
        if not transcript:
            return jsonify({
                'success': False,
                'error': 'Failed to transcribe audio'
            }), 500
        
        logger.info(f"Transcription completed. Length: {len(transcript)} characters")
        
        # Step 3: Summarize transcript
        logger.info("Step 3/3: Generating summary...")
        summary = summarizer.summarize(transcript)
        
        if not summary:
            return jsonify({
                'success': False,
                'error': 'Failed to generate summary'
            }), 500
        
        logger.info("Summary generated successfully!")
        
        # Clean up audio file
        try:
            os.remove(audio_path)
            logger.info(f"Cleaned up audio file: {audio_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up audio file: {e}")
        
        # Return results
        return jsonify({
            'success': True,
            'video_info': video_info,
            'transcript': transcript,
            'summary': summary,
            'stats': {
                'transcript_length': len(transcript),
                'transcript_words': len(transcript.split()),
                'summary_length': len(summary),
                'summary_words': len(summary.split()),
                'compression_ratio': f"{len(summary) / len(transcript) * 100:.1f}%"
            }
        })
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("YouTube Video Summarizer - Offline AI System")
    print("=" * 60)
    print("\nStarting Flask server...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nNote: First request will take longer as models are loaded.")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
