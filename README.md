# Amazon_Product_Pricing_Challenge
Amazon ML Challenge 2025: Smart Product Pricing
Project Overview
This repository contains the complete solution for the Amazon ML Challenge 2025, a competition focused on predicting product prices using a multimodal dataset. The primary goal was to build a machine learning model that could accurately forecast the price of a product based on its text description (catalog_content) and product image (image_link).

Our final model, a sophisticated multimodal pipeline combining state-of-the-art NLP and Computer Vision techniques, achieved a competitive SMAPE score of 53.70, representing a significant improvement of over 3.2 points from our initial baseline.

The Journey: A Tale of Three Models
Our approach evolved through a series of iterative, data-driven decisions, reflecting a real-world machine learning workflow.

Model V1: The Baseline (SMAPE ~56.99)
Architecture: A simple Ridge linear regression model.

Features: Basic text features derived from TfidfVectorizer and two custom-engineered features: text_length and ipq (Item Pack Quantity), which was extracted from the raw text using regular expressions.

Insight: This provided a solid, reliable baseline and proved the value of our ipq feature.

Model V2: The Engine Upgrade (SMAPE ~55.79)
Architecture: Upgraded the regression model to a more powerful LightGBM Gradient Boosting model.

Features: Same text-based features as V1.

Insight: The more complex model successfully captured more intricate patterns in the text data, resulting in a significant 1.2-point score improvement. This confirmed that a non-linear model was better suited for this problem.

Model V4: The Multimodal Breakthrough (SMAPE ~53.70)
This was our final and most powerful model, representing a significant leap in complexity and performance.

Architecture: A LightGBM model trained on a combined feature set.

Features:

Text Features: TfidfVectorizer (10,000 features) + text_length + ipq.

Image Features: We built a complete computer vision pipeline.

Feature Extraction: Used a pre-trained ResNet50 model (trained on ImageNet) to extract a 2,048-dimension feature vector for each of the 150,000 product images.

Noise Reduction (The Key Insight): Our initial attempt to combine raw image features (V3) made the model worse. We correctly diagnosed this as a "signal vs. noise" problem. To solve it, we implemented Principal Component Analysis (PCA) to compress the 2,048 noisy image features into the 128 most significant visual patterns.

Insight: This "noise-cancelling" PCA step was the key to unlocking the power of the image data. By feeding the model a smaller, cleaner, more potent set of visual features, we achieved a massive 2.09-point score improvement, proving the effectiveness of our multimodal strategy.

Technical Challenges & Professional Workflow
A significant part of this project was overcoming real-world technical hurdles.

Environment Instability: Our initial work on a local Jupyter environment was blocked by a corrupted Anaconda installation with conflicting library versions (pandas, numpy).

The Solution: Migration to the Cloud: We successfully migrated the entire project to Google Colab, a professional cloud-based environment. This solved all installation issues and gave us access to free, high-performance GPUs.

Data Pipeline Resilience: We encountered frequent disconnections in Colab during the 5+ hour process of downloading and processing 150,000 images. To solve this, we architected a robust, resilient data pipeline using Google Drive as a permanent data store.

The script was designed to be idempotent, automatically checking for existing files and resuming from its last checkpoint, thus protecting hours of work.

We implemented a "Cargo Container" strategy, zipping the 150,000 small image files into two large archives. This dramatically reduced the data transfer bottleneck between Google Drive and Colab, cutting the processing time from an estimated 30+ hours to just under 2 hours.

How to Run the Final Model
Ensure the dataset folder, train_image_features.npy, and test_image_features.npy are in your Google Drive.

Open the Final_Model_V4.1.ipynb notebook in Google Colab.

Enable a GPU runtime.

Run the cells. The script will connect to your Drive, build the model, and generate the final tuned_model_submission.csv.
