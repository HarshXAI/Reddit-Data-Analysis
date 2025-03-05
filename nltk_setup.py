import os
import nltk
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_nltk():
    """Setup NLTK resources safely for both local and deployment environments"""
    # Set custom NLTK data path if needed
    nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Add the custom path to NLTK's search paths
    nltk.data.path.append(nltk_data_dir)
    logger.info(f"Added NLTK data directory: {nltk_data_dir}")
    
    # List of required NLTK resources
    required_resources = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
    ]
    
    # Download required resources
    for resource_path, resource_name in required_resources:
        try:
            # Check if resource exists
            try:
                nltk.data.find(resource_path)
                logger.info(f"NLTK resource '{resource_name}' already exists")
            except LookupError:
                # Download the resource
                logger.info(f"Downloading NLTK resource '{resource_name}'")
                nltk.download(resource_name, download_dir=nltk_data_dir, quiet=True)
                logger.info(f"Successfully downloaded '{resource_name}'")
        except Exception as e:
            logger.error(f"Error setting up NLTK resource '{resource_name}': {str(e)}")
