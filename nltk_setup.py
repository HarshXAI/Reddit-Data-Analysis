import os
import nltk
import logging
import sys

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_nltk():
    """Setup NLTK resources safely for both local and deployment environments"""
    # Try multiple possible data directories for deployment environments
    possible_dirs = [
        os.path.join(os.getcwd(), "nltk_data"),
        "/home/appuser/nltk_data",
        "/app/nltk_data"
    ]
    
    # Create and add directories to NLTK's search path
    for nltk_data_dir in possible_dirs:
        try:
            if not os.path.exists(nltk_data_dir):
                os.makedirs(nltk_data_dir, exist_ok=True)
            
            # Add the custom path to NLTK's search paths
            nltk.data.path.append(nltk_data_dir)
            logger.info(f"Added NLTK data directory: {nltk_data_dir}")
        except Exception as e:
            logger.warning(f"Could not set up NLTK data directory {nltk_data_dir}: {str(e)}")
    
    # Add the current directory explicitly
    current_dir_nltk = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
    if current_dir_nltk not in nltk.data.path:
        try:
            if not os.path.exists(current_dir_nltk):
                os.makedirs(current_dir_nltk, exist_ok=True)
            nltk.data.path.append(current_dir_nltk)
        except Exception as e:
            logger.warning(f"Could not add current directory NLTK path: {str(e)}")
    
    # Print NLTK data path for debugging
    logger.info(f"NLTK data path: {nltk.data.path}")
    
    # List of required NLTK resources
    required_resources = [
        ('punkt', 'tokenizers/punkt'),
        ('stopwords', 'corpora/stopwords'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger')
    ]
    
    # Download required resources to each directory in the path until successful
    for resource_name, resource_path in required_resources:
        success = False
        
        # Try to find the resource
        try:
            nltk.data.find(resource_path)
            logger.info(f"NLTK resource '{resource_name}' already exists")
            success = True
        except LookupError:
            # Resource not found, try downloading
            pass
        
        if not success:
            # Try downloading to each directory in the path
            for nltk_path in nltk.data.path:
                try:
                    logger.info(f"Attempting to download '{resource_name}' to {nltk_path}")
                    nltk.download(resource_name, download_dir=nltk_path, quiet=True)
                    # Verify the download worked
                    nltk.data.find(resource_path)
                    logger.info(f"Successfully downloaded '{resource_name}' to {nltk_path}")
                    success = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to download '{resource_name}' to {nltk_path}: {str(e)}")
            
        if not success:
            # Last resort: try downloading to default location
            try:
                logger.info(f"Attempting to download '{resource_name}' to default location")
                nltk.download(resource_name, quiet=True)
                logger.info(f"Successfully downloaded '{resource_name}' to default location")
            except Exception as e:
                logger.error(f"All attempts to download '{resource_name}' failed: {str(e)}")

if __name__ == "__main__":
    # This allows running the setup directly
    setup_nltk()
    print("NLTK setup complete.")
