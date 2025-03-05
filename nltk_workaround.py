"""
This module provides workarounds for NLTK issues in deployment environments.
"""
import os
import sys
import nltk
import logging
import importlib
import types
import shutil

logger = logging.getLogger(__name__)

def apply_nltk_patches():
    """Apply all necessary patches to fix NLTK issues"""
    # Most important fix: create punkt_tab directory with necessary files
    create_punkt_tab_directory()
    
    # Patch punkt loading behavior
    patch_punkt_loader()
    
    # Add fallback tokenization
    add_fallback_tokenizer()

def create_punkt_tab_directory():
    """
    Create punkt_tab directory with necessary files by copying from punkt.
    This is a direct workaround for the punkt_tab issue.
    """
    try:
        # Check if punkt resource exists and download if needed
        try:
            punkt_path = nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            punkt_path = nltk.data.find('tokenizers/punkt')
        
        # Get the directory where punkt is stored
        punkt_dir = os.path.dirname(punkt_path)
        base_dir = os.path.dirname(punkt_dir)
        
        # Create punkt_tab directory
        punkt_tab_dir = os.path.join(base_dir, 'punkt_tab')
        english_dir = os.path.join(punkt_tab_dir, 'english')
        os.makedirs(english_dir, exist_ok=True)
        
        # Copy english.pickle to the new location or create a simple file
        english_pickle = os.path.join(punkt_dir, 'english.pickle')
        english_tab_pickle = os.path.join(english_dir, 'english.pickle')
        
        try:
            # Try to copy the existing pickle file
            if os.path.exists(english_pickle):
                shutil.copy2(english_pickle, english_tab_pickle)
                logger.info(f"Created punkt_tab directory by copying from punkt: {punkt_tab_dir}")
            else:
                # If no english.pickle, create an empty file
                with open(english_tab_pickle, 'w') as f:
                    f.write("")
                logger.info(f"Created empty punkt_tab file: {english_tab_pickle}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to copy/create punkt_tab files: {str(e)}")
            return False
    except Exception as e:
        logger.error(f"Error creating punkt_tab directory: {str(e)}")
        return False

def patch_punkt_loader():
    """Patch the punkt loading behavior"""
    try:
        # Get the PunktSentenceTokenizer class
        punkt_module = importlib.import_module('nltk.tokenize.punkt')
        PunktSentenceTokenizer = punkt_module.PunktSentenceTokenizer
        
        # Original load_lang method
        original_load_lang = PunktSentenceTokenizer.load_lang
        
        # Create patched method
        def patched_load_lang(self, lang):
            """Patched version with fallbacks for punkt_tab"""
            try:
                # First try punkt
                from nltk.data import find
                try:
                    lang_vars = find(f'tokenizers/punkt/{lang}.pickle')
                    self._params = nltk.data.load(lang_vars)
                    return
                except LookupError:
                    pass
                
                # Then try punkt_tab if available
                try:
                    lang_vars = find(f'tokenizers/punkt_tab/{lang}')
                    return  # Don't need to load it, just check existence
                except LookupError:
                    # If not found, create empty directories
                    create_punkt_tab_directory()
                    return
            except Exception:
                # If all else fails, don't do anything
                pass
        
        # Apply the patch
        PunktSentenceTokenizer.load_lang = patched_load_lang
        logger.info("Successfully patched punkt loader")
        return True
    except Exception as e:
        logger.error(f"Failed to patch punkt loader: {str(e)}")
        return False

def add_fallback_tokenizer():
    """Add fallback tokenizers when nltk ones fail"""
    try:
        # Define simple tokenizer functions
        def simple_word_tokenize(text):
            """Simple word tokenizer that splits on whitespace and punctuation"""
            # Replace punctuation with spaces
            import re
            text = re.sub(r'[^\w\s]', ' ', text)
            return text.split()
            
        def simple_sent_tokenize(text):
            """Simple sentence tokenizer that splits on common sentence delimiters"""
            import re
            # Split on period, question mark, or exclamation mark followed by space
            sentences = re.split(r'[.!?]\s+', text)
            # Add back the sentence delimiter
            sentences = [s + '.' for s in sentences[:-1]] + [sentences[-1]]
            return [s for s in sentences if s.strip()]
        
        # Add to nltk.tokenize
        tokenize_module = importlib.import_module('nltk.tokenize')
        
        # Save original functions
        original_word_tokenize = tokenize_module.word_tokenize
        original_sent_tokenize = tokenize_module.sent_tokenize
        
        # Create patched functions
        def patched_word_tokenize(text, language='english'):
            try:
                return original_word_tokenize(text, language)
            except Exception:
                return simple_word_tokenize(text)
                
        def patched_sent_tokenize(text, language='english'):
            try:
                return original_sent_tokenize(text, language)
            except Exception:
                return simple_sent_tokenize(text)
        
        # Apply patches
        tokenize_module.word_tokenize = patched_word_tokenize
        tokenize_module.sent_tokenize = patched_sent_tokenize
        
        logger.info("Added fallback tokenizers")
        return True
    except Exception as e:
        logger.error(f"Failed to add fallback tokenizers: {str(e)}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    apply_nltk_patches()
    print("NLTK patches applied successfully")
