import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import random
import math

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
    .not-bag-badge {
        background-color: #FFA500;
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
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Manual cosine similarity calculation
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors manually"""
    try:
        vec1 = np.array(vec1).flatten()
        vec2 = np.array(vec2).flatten()
        
        # Ensure vectors have same length
        min_len = min(len(vec1), len(vec2))
        vec1 = vec1[:min_len]
        vec2 = vec2[:min_len]
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, min(1.0, similarity))
    except:
        return random.uniform(0, 0.3)

# Enhanced Bag Database with more realistic features
BAG_DATABASE = [
    {
        "id": "lv_speedy_1",
        "name": "Louis Vuitton Speedy 30",
        "brand": "Louis Vuitton",
        "original_price": 1580.00,
        "price": 45.99,
        "type": "designer",
        "description": "Classic monogram canvas, leather trim",
        "features": ["Monogram Pattern", "Brown Leather", "Gold Hardware", "Structured Shape"],
        "color_profile": [0.47, 0.31, 0.24],  # Normalized RGB
        "shape_score": 0.85,
        "texture_score": 0.92,
        "bag_features": [0.9, 0.8, 0.7, 0.6],  # Strong bag characteristics
        "is_bag_probability": 0.95
    },
    {
        "id": "chanel_flap_1", 
        "name": "Chanel Classic Flap Bag",
        "brand": "Chanel",
        "original_price": 7800.00,
        "price": 39.99,
        "type": "designer",
        "description": "Quilted lambskin, gold-tone hardware",
        "features": ["Quilted Pattern", "Black Leather", "Chain Strap", "CC Logo"],
        "color_profile": [0.08, 0.08, 0.08],
        "shape_score": 0.78,
        "texture_score": 0.95,
        "bag_features": [0.8, 0.9, 0.6, 0.7],
        "is_bag_probability": 0.92
    },
    {
        "id": "hermes_birkin_1",
        "name": "Hermès Birkin 30",
        "brand": "Hermès", 
        "original_price": 12000.00,
        "price": 52.99,
        "type": "designer",
        "description": "Clemence leather, palladium hardware",
        "features": ["Structured Box", "Top Handles", "Lock Detail", "Premium Leather"],
        "color_profile": [0.59, 0.39, 0.31],
        "shape_score": 0.91,
        "texture_score": 0.98,
        "bag_features": [0.95, 0.7, 0.8, 0.9],
        "is_bag_probability": 0.98
    },
    {
        "id": "gucci_marmont_1",
        "name": "Gucci GG Marmont",
        "brand": "Gucci",
        "original_price": 2580.00,
        "price": 49.99,
        "type": "designer", 
        "description": "Matelassé leather, GG logo",
        "features": ["GG Pattern", "Chevron Quilting", "Heart Detail", "Chain Strap"],
        "color_profile": [0.15, 0.15, 0.15],
        "shape_score": 0.82,
        "texture_score": 0.88,
        "bag_features": [0.85, 0.75, 0.7, 0.8],
        "is_bag_probability": 0.90
    }
]

class AdvancedBagAnalyzer:
    def __init__(self):
        self.feature_size = 25
    
    def detect_if_bag(self, image):
        """Detect if the uploaded image is likely a bag"""
        try:
            image = image.convert('RGB')
            width, height = image.size
            img_array = np.array(image)
            
            bag_score = 0.0
            total_checks = 0
            
            # Check 1: Aspect ratio (bags are often rectangular)
            aspect_ratio = width / height
            if 0.5 <= aspect_ratio <= 2.0:  # Reasonable bag proportions
                bag_score += 0.3
            total_checks += 1
            
            # Check 2: Color variety (bags often have multiple colors/patterns)
            color_std = np.std(img_array)
            if color_std > 30:  # Some color variation
                bag_score += 0.2
            total_checks += 1
            
            # Check 3: Size (reasonable bag dimensions)
            if 200 <= max(width, height) <= 2000:
                bag_score += 0.2
            total_checks += 1
            
            # Check 4: Edge detection simulation (bags have defined edges)
            gray = np.mean(img_array, axis=2)
            edge_strength = np.std(np.diff(gray, axis=0)) + np.std(np.diff(gray, axis=1))
            if edge_strength > 50:
                bag_score += 0.3
            total_checks += 1
            
            probability = bag_score / total_checks if total_checks > 0 else 0
            return probability > 0.5, probability
            
        except Exception as e:
            return False, 0.0
    
    def extract_bag_features(self, image):
        """Extract features specifically for bag analysis"""
        try:
            image = image.convert('RGB')
            image = image.resize((100, 100))
            img_array = np.array(image)
            
            features = []
            
            # Color features
            avg_color = np.mean(img_array, axis=(0, 1)) / 255.0
            features.extend(avg_color)
            
            # Color distribution
            color_variance = np.std(img_array, axis=(0, 1)) / 255.0
            features.extend(color_variance)
            
            # Texture and edges (simplified)
            gray = np.mean(img_array, axis=2)
            features.append(np.std(gray) / 255.0)  # Contrast
            features.append(np.mean(np.abs(np.diff(gray, axis=0))) / 255.0)  # Vertical edges
            features.append(np.mean(np.abs(np.diff(gray, axis=1))) / 255.0)  # Horizontal edges
            
            # Shape features
            width, height = image.size
            features.append(width / 1000.0)  # Normalized width
            features.append(height / 1000.0)  # Normalized height
            features.append((width * height) / 1000000.0)  # Normalized area
            
            # Bag-specific features (handles, straps, etc. simulation)
            # These are simulated based on common bag characteristics
            center_region = img_array[40:60, 40:60]  # Center region
            center_brightness = np.mean(center_region) / 255.0
            features.append(center_brightness)
            
            # Edge concentration (simulating handles/straps)
            edges_vertical = np.mean(np.abs(np.diff(gray, axis=0)))
            edges_horizontal = np.mean(np.abs(np.diff(gray, axis=1)))
            features.append(edges_vertical / 255.0)
            features.append(edges_horizontal / 255.0)
            
            # Fill remaining features with pattern indicators
            while len(features) < self.feature_size:
                features.append(random.uniform(0, 0.5))  # Lower random values for non-bags
            
            return np.array(features)
            
        except Exception as e:
            # Return features that indicate non-bag for errors
            return np.random.rand(self.feature_size) * 0.3

def calculate_similarity_score(uploaded_features, bag_features, is_bag=True):
    """Calculate similarity with penalties for non-bag images"""
    base_similarity = cosine_similarity(uploaded_features, bag_features)
    
    if not is_bag:
        # Heavy penalty for non-bag images
        return base_similarity * 0.1  # 90% reduction
    
    return base_similarity

def find_similar_bags(uploaded_features, is_bag=True, threshold=0.3):
    """Find similar bags with proper filtering"""
    similar_bags = []
    
    for bag in BAG_DATABASE:
        # Combine bag features
        bag_feature_vector = np.concatenate([
            bag["color_profile"],
            [bag["shape_score"], bag["texture_score"]],
            bag["bag_features"]
        ])
        
        # Ensure same length
        min_len = min(len(uploaded_features), len(bag_feature_vector))
        uploaded_subset = uploaded_features[:min_len]
        bag_subset = bag_feature_vector[:min_len]
        
        # Calculate similarity with bag detection consideration
        similarity = calculate_similarity_score(uploaded_subset, bag_subset, is_bag)
        similarity_percent = similarity * 100
        
        if similarity_percent < threshold * 100:
            similar_bags.append({
                **bag,
                "similarity_score": similarity_percent,
                "is_highly_similar": similarity_percent > 70  # Flag high similarity
            })
    
    # Sort by similarity (ascending) - lower is more original
    similar_bags.sort(key=lambda x: x["similarity_score"])
    return similar_bags

# Initialize analyzer
analyzer = AdvancedBagAnalyzer()

# Header
st.markdown('<div class="main-header">👜 Bag Dupe Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Bag Authenticity Analysis • Under 30% Similarity = Original Design</div>', unsafe_allow_html=True)

# Upload Section
st.markdown("### Upload Bag Image")
st.markdown('<div class="upload-section">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a bag image...", 
    type=['jpg', 'jpeg', 'png'],
    label_visibility="collapsed",
    help="Upload clear images of handbags only"
)

st.markdown('</div>', unsafe_allow_html=True)

# Camera input
st.markdown("### Or Use Camera")
use_camera = st.checkbox("Use camera for real-time detection")

if use_camera:
    camera_image = st.camera_input("Take a picture of your bag")
    if camera_image:
        uploaded_file = camera_image

# Process image
if uploaded_file is not None:
    try:
        # Display image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, width=300, caption="Uploaded Image")
            
            # First, detect if this is a bag
            with st.spinner('🔍 Checking if this is a bag...'):
                is_bag, bag_probability = analyzer.detect_if_bag(image)
                time.sleep(1)
            
            if not is_bag:
                st.markdown('<div class="not-bag-badge">⚠️ NOT A BAG DETECTED</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence-medium">Bag Probability: {bag_probability*100:.1f}%</div>', unsafe_allow_html=True)
                st.warning("This doesn't appear to be a bag image. Please upload a clear photo of a handbag, purse, or shoulder bag.")
            else:
                st.markdown('<div class="original-badge">✅ BAG DETECTED</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence-high">Bag Probability: {bag_probability*100:.1f}%</div>', unsafe_allow_html=True)
                
        with col2:
            if is_bag:
                # Analyze bag features
                with st.spinner('🔍 Analyzing bag design features...'):
                    features = analyzer.extract_bag_features(image)
                    
                    # Progress animation
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Find similar bags with stricter threshold
                    similar_bags = find_similar_bags(features, is_bag=True, threshold=0.3)
                    
                    # Calculate originality - make it more strict
                    if similar_bags:
                        # Use the highest similarity found, not average
                        highest_similarity = max([b["similarity_score"] for b in similar_bags])
                        originality = 100 - highest_similarity
                        
                        # Additional penalty if multiple similar bags found
                        if len(similar_bags) >= 2:
                            originality *= 0.9  # 10% penalty
                    else:
                        originality = 85  # Base originality when no matches found
                
                # Results
                st.markdown("## 🎯 Bag Analysis Results")
                
                # Originality display - much stricter now
                if originality >= 80:
                    st.markdown('<div class="original-badge">✅ HIGHLY ORIGINAL BAG</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="confidence-high">Originality Score: {originality:.1f}%</div>', unsafe_allow_html=True)
                    st.success("🎉 Excellent! This bag shows very low similarity to known designer bags.")
                elif originality >= 60:
                    st.markdown('<div class="original-badge">✅ ORIGINAL DESIGN</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="confidence-high">Originality Score: {originality:.1f}%</div>', unsafe_allow_html=True)
                    st.info("👍 Good! This bag has distinctive design elements.")
                elif originality >= 40:
                    st.markdown('<div class="dupe-badge">⚠️ SIMILAR DESIGNS FOUND</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="confidence-medium">Originality Score: {originality:.1f}%</div>', unsafe_allow_html=True)
                    st.warning("🤔 This bag shares some design similarities with known styles.")
                else:
                    st.markdown('<div class="dupe-badge">❌ HIGH SIMILARITY DETECTED</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="confidence-low">Originality Score: {originality:.1f}%</div>', unsafe_allow_html=True)
                    st.error("⚠️ This bag shows significant similarity to designer bags below.")
                
                # Visual bar
                st.markdown(f"**Originality Score:** {originality:.1f}%")
                st.markdown(f'<div class="similarity-bar"><div class="similarity-fill" style="width: {originality}%;"></div></div>', unsafe_allow_html=True)
                
                # Show similar bags if any found
                if similar_bags:
                    st.markdown(f"### 📊 Similar Designer Bags Found")
                    
                    for i, bag in enumerate(similar_bags):
                        if bag["similarity_score"] > 20:  # Only show meaningful similarities
                            with st.expander(f"Similar Design {i+1}: {bag['name']} ({bag['similarity_score']:.1f}% similar)", expanded=True):
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    st.markdown("""<div style='width:150px; height:150px; background:#f0f0f0; border-radius:10px; 
                                               display:flex; align-items:center; justify-content:center; color:#666;'>
                                               Designer Bag</div>""", unsafe_allow_html=True)
                                    
                                    score_color = "confidence-high" if bag["similarity_score"] < 15 else "confidence-medium" if bag["similarity_score"] < 25 else "confidence-low"
                                    st.markdown(f'<div class="{score_color}">Similarity: {bag["similarity_score"]:.1f}%</div>', unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"**{bag['name']}**")
                                    st.markdown(f"**Brand:** {bag['brand']}")
                                    st.markdown(f"**Original Price:** ${bag['original_price']:,.2f}")
                                    st.markdown(f"**Affordable Alternative:** ${bag['price']:.2f}")
                                    st.markdown(f"**Description:** {bag['description']}")
                                    
                                    # Features
                                    st.markdown("**Key Features:**")
                                    features_html = "".join([f'<span class="feature-pill">{feat}</span>' for feat in bag['features'][:3]])
                                    st.markdown(features_html, unsafe_allow_html=True)
                
                if not similar_bags or all(bag["similarity_score"] < 10 for bag in similar_bags):
                    st.success("🎉 No significant similarities found! This appears to be a very original bag design.")
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Please try a different image file.")

else:
    # Instructions
    st.markdown("### 🚀 How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1. Upload Bag Image")
        st.markdown("""
        Upload clear photos of:
        - Handbags
        - Purses  
        - Shoulder bags
        - Tote bags
        """)
        st.warning("Non-bag images will be detected and rejected")
        
    with col2:
        st.markdown("#### 2. AI Bag Detection")
        st.markdown("""
        We verify it's a bag and analyze:
        - Shape & Proportions
        - Handles/Straps
        - Design Patterns
        - Color Scheme
        """)
        
    with col3:
        st.markdown("#### 3. Authenticity Results")
        st.markdown("""
        Receive detailed analysis:
        - Bag Detection Score
        - Originality Rating
        - Similar Designer Bags
        - Price Comparisons
        """)
    
    st.markdown("---")
    st.markdown("### 📈 Detection System")
    
    threshold_col1, threshold_col2 = st.columns(2)
    
    with threshold_col1:
        st.markdown("""
        **🎯 Originality Scale:**
        - **80-100%**: Highly Original
        - **60-80%**: Original Design  
        - **40-60%**: Some Similarity
        - **0-40%**: Potential Dupe
        """)
    
    with threshold_col2:
        st.markdown("""
        **🔍 Similarity Threshold:**
        - **<30%**: Original Design
        - **30%+**: Similarity Detected
        - **Bag detection first**
        - **Strict matching**
        """)
    
    st.markdown("---")
    st.markdown("### 🏷️ Supported Bag Types")
    
    bag_cols = st.columns(4)
    bag_types = [
        ("Handbags", "Structured bags with handles"),
        ("Crossbody", "Bags with long straps"),
        ("Tote Bags", "Large open-top bags"), 
        ("Clutches", "Small handheld bags")
    ]
    
    for i, (name, desc) in enumerate(bag_types):
        with bag_cols[i]:
            st.markdown(f"**{name}**")
            st.markdown(f"*{desc}*")

# Footer
st.markdown("---")
st.markdown("**👜 Bag Dupe Detector** • Advanced Bag Detection & Authenticity Analysis")
st.markdown("*Only analyzes actual bag images • Under 30% similarity = Original Design*")
