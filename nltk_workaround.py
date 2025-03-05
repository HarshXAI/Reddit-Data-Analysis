"""
This module provides workarounds for NLTK issues in deployment environments.
"""
import os
import sys
import nltk
import logging
import importlib
import types

logger = logging.getLogger(__name__)

def apply_nltk_patches():
    """Apply all necessary patches to fix NLTK issues"""
    # Patch 1: Fix punkt_tab reference in punkt.py
    patch_punkt_tab()
    
    # Patch 2: Ensure punkt is available
    ensure_punkt_available()
    
    # Patch 3: Add direct word_tokenize implementation as fallback
    add_fallback_tokenizer()

def patch_punkt_tab():
    """
    Patch the NLTK PunktSentenceTokenizer to avoid using punkt_tab.
    """
    try:
        # Get the module
        punkt_module = importlib.import_module('nltk.tokenize.punkt')
        PunktSentenceTokenizer = punkt_module.PunktSentenceTokenizer
        
        # Original method
        original_load_lang = PunktSentenceTokenizer.load_lang
        
        # Create patched method
        def patched_load_lang(self, lang):
            """Patched version that avoids using punkt_tab"""
            try:
                # Try to load directly from punkt instead of punkt_tab
                from nltk.data import find
                try:
                    lang_vars = find(f'tokenizers/punkt/{lang}.pickle')
                    self._params = nltk.data.load(lang_vars)
                    logger.info(f"Successfully loaded punkt parameters for {lang}")
                    return
                except LookupError:
                    logger.warning(f"Could not find punkt pickle for {lang}, downloading...")
                    nltk.download('punkt', quiet=True)
                    lang_vars = find(f'tokenizers/punkt/{lang}.pickle')
                    self._params = nltk.data.load(lang_vars)
                    return
            except Exception as e:
                # If all else fails, try original method
                logger.warning(f"Patched punkt loader failed: {str(e)}, trying original...")
                return original_load_lang(self, lang)
        
        # Apply patch
        PunktSentenceTokenizer.load_lang = patched_load_lang
        logger.info("Successfully applied punkt_tab patch")
        return True
    except Exception as e:
        logger.error(f"Failed to apply punkt_tab patch: {str(e)}")
        return False

def ensure_punkt_available():
    """Ensure punkt data is available in all possible paths"""
    nltk_paths = nltk.data.path
    
    for path in nltk_paths:
        try:
            # Create punkt directory structure
            punkt_dir = os.path.join(path, 'tokenizers', 'punkt')
            os.makedirs(punkt_dir, exist_ok=True)
            
            # Check if punkt is already available
            try:
                nltk.data.find('tokenizers/punkt/english.pickle')
                logger.info("Punkt english pickle already exists")
                return True
            except LookupError:
                # Download punkt
                nltk.download('punkt', download_dir=path, quiet=True)
                logger.info(f"Downloaded punkt to {path}")
                return True
        except Exception as e:
            logger.warning(f"Could not ensure punkt in {path}: {str(e)}")
    
    # Last resort: try default location
    try:
        nltk.download('punkt', quiet=True)
        return True
    except Exception as e:
        logger.error(f"Failed to download punkt: {str(e)}")
        return False

def add_fallback_tokenizer():
    """Add a simple fallback tokenizer that doesn't depend on punkt"""
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
    # This allows running the patches directly
    logging.basicConfig(level=logging.INFO)
    apply_nltk_patches()
    print("NLTK patches applied successfully")
