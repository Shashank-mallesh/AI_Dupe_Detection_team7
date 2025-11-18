import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
import requests
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Bag Dupe Detector",
    page_icon="👜",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f1f1f;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .product-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .original-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 1rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .dupe-badge {
        background-color: #FF6B6B;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 1rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .match-badge {
        background-color: #6c5ce7;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .price-save {
        color: #4CAF50;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .rating {
        color: #FFA500;
        font-weight: bold;
    }
    .view-product-btn {
        background-color: #FF6B6B;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        text-align: center;
        margin-bottom: 2rem;
    }
    .similarity-bar {
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    .similarity-fill {
        height: 100%;
        background: linear-gradient(90deg, #FF6B6B, #4CAF50);
        transition: width 0.5s ease;
    }
    .confidence-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FFA500;
        font-weight: bold;
    }
    .confidence-low {
        color: #FF6B6B;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Bag feature database (in production, this would be a proper database)
BAG_FEATURES_DB = {
    "designer_bags": [
        {
            "id": "lv_speedy_1",
            "name": "Louis Vuitton Speedy 30",
            "brand": "Louis Vuitton",
            "original_price": 1580.00,
            "features": np.random.randn(512).tolist(),  # Placeholder for actual features
            "image_url": "https://via.placeholder.com/300x300?text=LV+Speedy",
            "description": "Classic monogram canvas, leather trim"
        },
        {
            "id": "chanel_flap_1",
            "name": "Chanel Classic Flap Bag",
            "brand": "Chanel",
            "original_price": 7800.00,
            "features": np.random.randn(512).tolist(),
            "image_url": "https://via.placeholder.com/300x300?text=Chanel+Flap",
            "description": "Quilted lambskin, gold-tone hardware"
        },
        {
            "id": "hermes_birkin_1",
            "name": "Hermès Birkin 30",
            "brand": "Hermès",
            "original_price": 12000.00,
            "features": np.random.randn(512).tolist(),
            "image_url": "https://via.placeholder.com/300x300?text=Hermès+Birkin",
            "description": "Clemence leather, palladium hardware"
        }
    ],
    "affordable_alternatives": [
        {
            "id": "dupe_lv_1",
            "name": "Vintage Style Shoulder Bag",
            "brand": "Amazon Fashion",
            "price": 45.99,
            "original_price": 1580.00,
            "features": np.random.randn(512).tolist(),
            "match_score": 25,
            "image_url": "https://via.placeholder.com/300x300?text=LV+Dupe",
            "description": "Similar monogram pattern, synthetic leather"
        },
        {
            "id": "dupe_chanel_1",
            "name": "Quilted Crossbody Bag",
            "brand": "Fashion Retailer",
            "price": 39.99,
            "original_price": 7800.00,
            "features": np.random.randn(512).tolist(),
            "match_score": 28,
            "image_url": "https://via.placeholder.com/300x300?text=Chanel+Dupe",
            "description": "Quilted pattern, gold-color chain"
        },
        {
            "id": "dupe_hermes_1",
            "name": "Structured Tote Bag",
            "brand": "Budget Bags Co.",
            "price": 52.99,
            "original_price": 12000.00,
            "features": np.random.randn(512).tolist(),
            "match_score": 22,
            "image_url": "https://via.placeholder.com/300x300?text=Hermès+Dupe",
            "description": "Structured design, faux leather"
        }
    ]
}

class BagFeatureExtractor:
    def __init__(self):
        # Use a pre-trained ResNet model for feature extraction
        self.model = models.resnet50(pretrained=True)
        # Remove the final classification layer
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval()
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_features(self, image):
        """Extract features from bag image"""
        try:
            # Convert to RGB if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Apply transformations
            image_tensor = self.transform(image).unsqueeze(0)
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.squeeze().numpy()
            
            return features
        except Exception as e:
            st.error(f"Feature extraction error: {e}")
            return np.random.randn(512)  # Fallback to random features

def calculate_similarity(features1, features2):
    """Calculate cosine similarity between two feature vectors"""
    try:
        # Ensure features are 2D arrays
        features1 = np.array(features1).reshape(1, -1)
        features2 = np.array(features2).reshape(1, -1)
        
        similarity = cosine_similarity(features1, features2)[0][0]
        return max(0, min(100, similarity * 100))  # Convert to percentage
    except:
        return random.uniform(0, 30)  # Fallback for demo

def detect_bag_type(image):
    """Simple bag type detection based on color and shape analysis"""
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Simple analysis (in production, use proper ML model)
        avg_brightness = np.mean(hsv[:,:,2])
        avg_saturation = np.mean(hsv[:,:,1])
        
        if avg_brightness > 150:
            return "Light-colored bag"
        elif avg_saturation > 100:
            return "Colorful bag"
        else:
            return "Dark-colored bag"
    except:
        return "Fashion bag"

def find_similar_bags(uploaded_features, threshold=30):
    """Find similar bags with similarity scores below threshold"""
    similar_bags = []
    
    # Check against designer bags first
    for bag in BAG_FEATURES_DB["designer_bags"]:
        similarity = calculate_similarity(uploaded_features, bag["features"])
        
        if similarity < threshold:
            similar_bags.append({
                **bag,
                "similarity_score": similarity,
                "type": "designer"
            })
    
    # Check against affordable alternatives
    for bag in BAG_FEATURES_DB["affordable_alternatives"]:
        similarity = calculate_similarity(uploaded_features, bag["features"])
        
        if similarity < threshold:
            similar_bags.append({
                **bag,
                "similarity_score": similarity,
                "type": "affordable"
            })
    
    # Sort by similarity score (ascending - lower score means more original)
    similar_bags.sort(key=lambda x: x["similarity_score"])
    return similar_bags[:5]  # Return top 5 matches

def preprocess_image(image):
    """Preprocess uploaded image for analysis"""
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize while maintaining aspect ratio
        h, w = image.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        return image
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return image

# Initialize feature extractor
@st.cache_resource
def load_feature_extractor():
    return BagFeatureExtractor()

feature_extractor = load_feature_extractor()

# Header Section
st.markdown('<div class="main-header">👜 Bag Dupe Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time bag authenticity detection • Under 30% similarity = Original</div>', unsafe_allow_html=True)

# Upload Section
st.markdown("### Upload Bag Image for Analysis")
st.markdown('<div class="upload-section">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a bag image...", 
    type=['jpg', 'jpeg', 'png'],
    label_visibility="collapsed",
    help="Upload clear images of bags for authenticity detection"
)

st.markdown('</div>', unsafe_allow_html=True)

# Real-time camera input
st.markdown("### Or Use Camera for Real-time Detection")
use_camera = st.checkbox("Use camera for real-time detection")

if use_camera:
    camera_image = st.camera_input("Take a picture of your bag")
    if camera_image:
        uploaded_file = camera_image

# Process uploaded image
if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns([1, 2])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, width=300, caption="Uploaded Bag Image")
        
        # Basic image analysis
        bag_type = detect_bag_type(image)
        st.markdown(f"**Detected:** {bag_type}")
    
    with col2:
        # Analysis progress
        with st.spinner('🔍 Analyzing bag features...'):
            # Preprocess image
            processed_image = preprocess_image(image)
            
            # Extract features
            uploaded_features = feature_extractor.extract_features(processed_image)
            
            # Simulate processing time for demo
            time.sleep(2)
            
            # Find similar bags
            similar_bags = find_similar_bags(uploaded_features, threshold=30)
            
            # Calculate overall originality score
            if similar_bags:
                avg_similarity = np.mean([bag["similarity_score"] for bag in similar_bags])
                originality_score = max(0, 100 - avg_similarity)
            else:
                originality_score = 100  # No similar bags found
    
    # Display results
    st.markdown("## 🎯 Detection Results")
    
    # Originality indicator
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if originality_score >= 70:
            st.markdown('<div class="original-badge">✅ LIKELY ORIGINAL</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence-high">Confidence: {originality_score:.1f}%</div>', unsafe_allow_html=True)
        elif originality_score >= 50:
            st.markdown('<div class="dupe-badge">⚠️ UNCERTAIN</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence-medium">Confidence: {originality_score:.1f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="dupe-badge">❌ LIKELY DUPE</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence-low">Confidence: {originality_score:.1f}%</div>', unsafe_allow_html=True)
        
        # Similarity visualization
        st.markdown(f"**Overall Originality Score:** {originality_score:.1f}%")
        st.markdown('<div class="similarity-bar"><div class="similarity-fill" style="width: {}%;"></div></div>'.format(originality_score), unsafe_allow_html=True)
    
    # Display similar bags found
    if similar_bags:
        st.markdown(f"### 📊 Similarity Analysis ({len(similar_bags)} matches found)")
        
        for i, bag in enumerate(similar_bags, 1):
            st.markdown("---")
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.image(bag.get("image_url", "https://via.placeholder.com/200x200?text=Bag+Image"), width=150)
                similarity_color = "confidence-high" if bag["similarity_score"] < 15 else "confidence-medium" if bag["similarity_score"] < 25 else "confidence-low"
                st.markdown(f'<div class="{similarity_color}">Similarity: {bag["similarity_score"]:.1f}%</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"#### {bag['name']}")
                st.markdown(f"**Brand:** {bag['brand']}")
                
                if bag['type'] == 'designer':
                    st.markdown(f"**Price:** ${bag['original_price']:,.2f}")
                    st.markdown("**Status:** 🏷️ Designer Original")
                else:
                    st.markdown(f"**Price:** ${bag['price']:.2f}")
                    st.markdown(f"**Original Price:** ${bag['original_price']:,.2f}")
                    st.markdown("**Status:** 💰 Affordable Alternative")
                
                st.markdown(f"**Description:** {bag['description']}")
                
                # Action buttons
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    st.button("View Details", key=f"view_{i}")
                with col_btn2:
                    st.button("Compare", key=f"compare_{i}")
    else:
        st.success("🎉 No similar bags found! This appears to be highly original.")
        
        # Show some random alternatives for reference
        st.markdown("### 💡 For Reference - Popular Designer Bags")
        ref_col1, ref_col2, ref_col3 = st.columns(3)
        
        reference_bags = BAG_FEATURES_DB["designer_bags"][:3]
        for i, bag in enumerate(reference_bags):
            with [ref_col1, ref_col2, ref_col3][i]:
                st.image(bag["image_url"], width=150)
                st.markdown(f"**{bag['name']}**")
                st.markdown(f"${bag['original_price']:,.2f}")

else:
    # Demo section when no image is uploaded
    st.markdown("### 🚀 How It Works")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1. Upload Image")
        st.markdown("Take a clear photo of your bag or upload an existing image")
        
    with col2:
        st.markdown("#### 2. AI Analysis")
        st.markdown("Our system analyzes shape, pattern, and design features")
        
    with col3:
        st.markdown("#### 3. Get Results")
        st.markdown("Discover if your bag is original or find affordable alternatives")
    
    st.markdown("---")
    st.markdown("### 📈 Detection Threshold")
    st.markdown("""
    - **< 15% similarity**: Highly original design
    - **15-25% similarity**: Moderately similar  
    - **25-30% similarity**: Some similarities detected
    - **> 30% similarity**: Potential dupe identified
    """)

# Footer
st.markdown("---")
st.markdown("**Bag Dupe Detector** • Real-time AI Analysis • Under 30% Similarity = Original")
st.markdown("*Upload a bag image to check authenticity and find alternatives*")
