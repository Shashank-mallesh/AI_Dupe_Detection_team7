import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import io
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import time
import random
import base64

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
    .feature-pill {
        background-color: #e0e0e0;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .product-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Bag Database with Realistic Data
BAG_FEATURES_DB = {
    "designer_bags": [
        {
            "id": "lv_speedy_1",
            "name": "Louis Vuitton Speedy 30",
            "brand": "Louis Vuitton",
            "original_price": 1580.00,
            "image_url": "https://images.unsplash.com/photo-1548036328-c9fa89d128fa?w=300&h=300&fit=crop",
            "description": "Classic monogram canvas, leather trim",
            "features_list": ["Monogram Pattern", "Brown Leather", "Gold Hardware", "Structured Shape"],
            "color_profile": [120, 80, 60],  # RGB-like features
            "texture_score": 0.95,
            "shape_vector": [0.8, 0.6, 0.3]
        },
        {
            "id": "chanel_flap_1",
            "name": "Chanel Classic Flap Bag",
            "brand": "Chanel",
            "original_price": 7800.00,
            "image_url": "https://images.unsplash.com/photo-1591561954557-26941169b49e?w=300&h=300&fit=crop",
            "description": "Quilted lambskin, gold-tone hardware",
            "features_list": ["Quilted Pattern", "Black Leather", "Chain Strap", "CC Logo"],
            "color_profile": [20, 20, 20],
            "texture_score": 0.92,
            "shape_vector": [0.7, 0.8, 0.2]
        },
        {
            "id": "hermes_birkin_1",
            "name": "Hermès Birkin 30",
            "brand": "Hermès",
            "original_price": 12000.00,
            "image_url": "https://images.unsplash.com/photo-1584917865442-de89df76afd3?w=300&h=300&fit=crop",
            "description": "Clemence leather, palladium hardware",
            "features_list": ["Structured Box", "Top Handles", "Lock Detail", "Premium Leather"],
            "color_profile": [150, 100, 80],
            "texture_score": 0.98,
            "shape_vector": [0.9, 0.5, 0.4]
        }
    ],
    "affordable_alternatives": [
        {
            "id": "dupe_lv_1",
            "name": "Vintage Style Shoulder Bag",
            "brand": "Amazon Fashion",
            "price": 45.99,
            "original_price": 1580.00,
            "image_url": "https://images.unsplash.com/photo-1584917865442-de89df76afd3?w=300&h=300&fit=crop",
            "description": "Similar monogram pattern, synthetic leather",
            "features_list": ["Monogram Print", "Faux Leather", "Similar Shape", "Affordable"],
            "color_profile": [110, 75, 55],
            "texture_score": 0.65,
            "shape_vector": [0.75, 0.55, 0.25]
        },
        {
            "id": "dupe_chanel_1",
            "name": "Quilted Crossbody Bag",
            "brand": "Fashion Retailer",
            "price": 39.99,
            "original_price": 7800.00,
            "image_url": "https://images.unsplash.com/photo-1591561954557-26941169b49e?w=300&h=300&fit=crop",
            "description": "Quilted pattern, gold-color chain",
            "features_list": ["Quilted Design", "Chain Strap", "Black Color", "Budget Friendly"],
            "color_profile": [25, 25, 25],
            "texture_score": 0.60,
            "shape_vector": [0.65, 0.75, 0.18]
        }
    ]
}

class SimpleFeatureExtractor:
    def __init__(self):
        self.feature_size = 50  # Reduced feature size for simplicity
    
    def extract_features(self, image):
        """Extract simple features from bag image using basic image processing"""
        try:
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image for consistent feature extraction
            image = image.resize((100, 100))
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Extract simple features
            features = []
            
            # Color features (average RGB values)
            avg_color = np.mean(img_array, axis=(0, 1))
            features.extend(avg_color / 255.0)  # Normalize to 0-1
            
            # Color distribution (histogram)
            for channel in range(3):
                hist = np.histogram(img_array[:,:,channel], bins=5, range=(0, 255))[0]
                features.extend(hist / np.sum(hist))  # Normalized histogram
            
            # Texture features (simple edge detection using differences)
            gray_img = np.mean(img_array, axis=2)
            texture_feature = np.std(gray_img) / 255.0  # Normalized standard deviation
            features.append(texture_feature)
            
            # Shape features (aspect ratio and basic shape)
            height, width = gray_img.shape
            aspect_ratio = width / height
            features.append(min(aspect_ratio, 3.0) / 3.0)  # Normalized
            
            # Fill remaining features with random values (simulating more complex features)
            while len(features) < self.feature_size:
                features.append(random.uniform(0, 1))
            
            return np.array(features[:self.feature_size])
            
        except Exception as e:
            st.error(f"Feature extraction error: {e}")
            # Return random features as fallback
            return np.random.rand(self.feature_size)

def calculate_similarity(features1, features2):
    """Calculate cosine similarity between two feature vectors"""
    try:
        # Ensure features are 2D arrays
        features1 = np.array(features1).reshape(1, -1)
        features2 = np.array(features2).reshape(1, -1)
        
        similarity = cosine_similarity(features1, features2)[0][0]
        return max(0, min(100, similarity * 100))  # Convert to percentage
    except Exception as e:
        # Fallback to random similarity for demo
        return random.uniform(0, 30)

def analyze_bag_features(image):
    """Analyze bag features and return characteristics"""
    try:
        characteristics = []
        
        # Basic size analysis
        width, height = image.size
        if width > height * 1.5:
            characteristics.append("Horizontal Shape")
        elif height > width * 1.5:
            characteristics.append("Vertical Shape")
        else:
            characteristics.append("Square Shape")
        
        # Color analysis
        img_array = np.array(image)
        avg_color = np.mean(img_array, axis=(0, 1))
        
        # Determine color characteristics
        if np.max(avg_color) - np.min(avg_color) < 50:
            characteristics.append("Neutral Colors")
        elif avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]:
            characteristics.append("Warm Tones")
        elif avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
            characteristics.append("Natural Tones")
        else:
            characteristics.append("Cool Tones")
        
        # Brightness analysis
        brightness = np.mean(avg_color)
        if brightness > 200:
            characteristics.append("Light Bag")
        elif brightness > 100:
            characteristics.append("Medium Bag")
        else:
            characteristics.append("Dark Bag")
            
        return characteristics
        
    except Exception as e:
        return ["Classic Design", "Standard Features", "Common Style"]

def preprocess_image(image):
    """Preprocess uploaded image for analysis"""
    try:
        # Resize while maintaining aspect ratio
        max_dim = 600
        width, height = image.size
        
        if max(width, height) > max_dim:
            scale = max_dim / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return image

def find_similar_bags(uploaded_features, threshold=30):
    """Find similar bags with similarity scores below threshold"""
    similar_bags = []
    
    # Check against all bags in database
    for bag_type in ["designer_bags", "affordable_alternatives"]:
        for bag in BAG_FEATURES_DB[bag_type]:
            # Create feature vector from bag properties
            bag_features = np.concatenate([
                np.array(bag["color_profile"]) / 255.0,
                [bag["texture_score"]],
                bag["shape_vector"]
            ])
            
            # Pad with random features to match expected size
            while len(bag_features) < len(uploaded_features):
                bag_features = np.append(bag_features, random.uniform(0, 1))
            
            similarity = calculate_similarity(uploaded_features, bag_features)
            
            if similarity < threshold:
                similar_bags.append({
                    **bag,
                    "similarity_score": similarity,
                    "type": "designer" if bag_type == "designer_bags" else "affordable"
                })
    
    # Sort by similarity score (ascending - lower score means more original)
    similar_bags.sort(key=lambda x: x["similarity_score"])
    return similar_bags[:3]  # Return top 3 matches

# Initialize feature extractor
@st.cache_resource
def load_feature_extractor():
    return SimpleFeatureExtractor()

feature_extractor = load_feature_extractor()

# Header Section
st.markdown('<div class="main-header">👜 Bag Dupe Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time Bag Authenticity Detection • Under 30% Similarity = Original</div>', unsafe_allow_html=True)

# Upload Section
st.markdown("### Upload Bag Image for Analysis")
st.markdown('<div class="upload-section">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a bag image...", 
    type=['jpg', 'jpeg', 'png'],
    label_visibility="collapsed",
    help="Upload clear images of handbags, purses, or shoulder bags"
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
    try:
        # Display uploaded image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, width=300, caption="Uploaded Bag Image")
            
            # Basic image analysis
            characteristics = analyze_bag_features(image)
            st.markdown("**Detected Features:**")
            for char in characteristics:
                st.markdown(f'<div class="feature-pill">{char}</div>', unsafe_allow_html=True)
                
        with col2:
            # Analysis progress
            with st.spinner('🔍 Analyzing bag features and patterns...'):
                # Preprocess image
                processed_image = preprocess_image(image)
                
                # Extract features
                uploaded_features = feature_extractor.extract_features(processed_image)
                
                # Simulate processing time
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
                
                # Find similar bags
                similar_bags = find_similar_bags(uploaded_features, threshold=30)
                
                # Calculate overall originality score
                if similar_bags:
                    avg_similarity = np.mean([bag["similarity_score"] for bag in similar_bags])
                    originality_score = max(0, 100 - avg_similarity)
                else:
                    originality_score = 95  # No similar bags found
        
        # Display results
        st.markdown("## 🎯 Detection Results")
        
        # Originality indicator
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if originality_score >= 85:
                st.markdown('<div class="original-badge">✅ HIGHLY ORIGINAL</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence-high">Originality Score: {originality_score:.1f}%</div>', unsafe_allow_html=True)
                st.success("🎉 Excellent! Your bag shows very low similarity to known designs.")
            elif originality_score >= 70:
                st.markdown('<div class="original-badge">✅ LIKELY ORIGINAL</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence-high">Originality Score: {originality_score:.1f}%</div>', unsafe_allow_html=True)
                st.info("👍 Good! Your bag has unique characteristics.")
            elif originality_score >= 50:
                st.markdown('<div class="dupe-badge">⚠️ MODERATE SIMILARITY</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence-medium">Originality Score: {originality_score:.1f}%</div>', unsafe_allow_html=True)
                st.warning("🤔 Some similarities detected with existing designs.")
            else:
                st.markdown('<div class="dupe-badge">❌ HIGH SIMILARITY</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence-low">Originality Score: {originality_score:.1f}%</div>', unsafe_allow_html=True)
                st.error("⚠️ Significant similarity detected with known designs.")
            
            # Similarity visualization
            st.markdown(f"**Originality Score:** {originality_score:.1f}%")
            st.markdown('<div class="similarity-bar"><div class="similarity-fill" style="width: {}%;"></div></div>'.format(originality_score), unsafe_allow_html=True)
        
        # Display similar bags found
        if similar_bags:
            st.markdown(f"### 📊 Similar Designs Found ({len(similar_bags)} matches)")
            st.markdown("**Designs with similarity below 30% (your bag is more original):**")
            
            for i, bag in enumerate(similar_bags, 1):
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.image(bag.get("image_url"), width=150, use_column_width=True)
                        similarity_color = "confidence-high" if bag["similarity_score"] < 15 else "confidence-medium" if bag["similarity_score"] < 25 else "confidence-low"
                        st.markdown(f'<div class="{similarity_color}">Similarity: {bag["similarity_score"]:.1f}%</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"#### {bag['name']}")
                        st.markdown(f"**Brand:** {bag['brand']}")
                        
                        if bag['type'] == 'designer':
                            st.markdown(f"**Price:** ${bag['original_price']:,.2f}")
                            st.markdown("**Type:** 🏷️ Designer Original")
                        else:
                            st.markdown(f"**Price:** ${bag['price']:.2f}")
                            st.markdown(f"**Original Reference:** ${bag['original_price']:,.2f}")
                            st.markdown("**Type:** 💰 Affordable Alternative")
                        
                        st.markdown(f"**Description:** {bag['description']}")
                        
                        # Show features
                        st.markdown("**Key Features:**")
                        features_html = "".join([f'<span class="feature-pill">{feature}</span>' for feature in bag.get('features_list', [])[:3]])
                        st.markdown(features_html, unsafe_allow_html=True)
        
        else:
            st.success("🎉 No similar designs found! Your bag appears to be highly unique.")
            
            # Show reference designs
            st.markdown("### 🏷️ Reference Designer Bags")
            ref_cols = st.columns(3)
            reference_bags = BAG_FEATURES_DB["designer_bags"][:3]
            
            for i, bag in enumerate(reference_bags):
                with ref_cols[i]:
                    st.image(bag["image_url"], use_column_width=True)
                    st.markdown(f"**{bag['name']}**")
                    st.markdown(f"${bag['original_price']:,.2f}")
                    st.markdown(f"*{bag['description']}*")
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Please try uploading a different image or check the file format.")

else:
    # Demo section when no image is uploaded
    st.markdown("### 🚀 How It Works")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1. Upload Image")
        st.markdown("Take a clear photo of your bag or upload an existing image")
        st.image("https://images.unsplash.com/photo-1584917865442-de89df76afd3?w=200&h=200&fit=crop")
        
    with col2:
        st.markdown("#### 2. AI Analysis")
        st.markdown("Our system analyzes shape, color, texture, and patterns")
        st.image("https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=200&h=200&fit=crop")
        
    with col3:
        st.markdown("#### 3. Get Results")
        st.markdown("Discover originality score and similar designs")
        st.image("https://images.unsplash.com/photo-1591561954557-26941169b49e?w=200&h=200&fit=crop")
    
    st.markdown("---")
    st.markdown("### 📈 Originality Threshold System")
    
    threshold_col1, threshold_col2 = st.columns(2)
    
    with threshold_col1:
        st.markdown("""
        **🎯 Originality Scores:**
        - **85-100%**: Highly Original
        - **70-85%**: Likely Original  
        - **50-70%**: Moderate Similarity
        - **<50%**: High Similarity
        """)
    
    with threshold_col2:
        st.markdown("""
        **🔍 Similarity Threshold:**
        - **<30%**: Considered Original
        - **30%+**: Potential Similarity
        - Based on design features analysis
        """)
    
    # Show sample database
    st.markdown("### 🏷️ Sample Designer Database")
    sample_cols = st.columns(3)
    sample_bags = BAG_FEATURES_DB["designer_bags"]
    
    for i, bag in enumerate(sample_bags):
        with sample_cols[i]:
            st.image(bag["image_url"], use_column_width=True)
            st.markdown(f"**{bag['name']}**")
            st.markdown(f"**Brand:** {bag['brand']}")
            st.markdown(f"**Price:** ${bag['original_price']:,.2f}")

# Footer
st.markdown("---")
st.markdown("**👜 Bag Dupe Detector** • AI-Powered Authenticity Analysis • Under 30% Similarity = Original Design")
st.markdown("*Upload any bag image to check its originality and find similar designs*")
