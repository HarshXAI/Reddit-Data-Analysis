#!/bin/bash

# Setup script for NLTK resources
echo "Setting up environment and downloading NLTK resources..."
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger -d ./nltk_data

# Add any other setup steps here

echo "Setup complete!"
