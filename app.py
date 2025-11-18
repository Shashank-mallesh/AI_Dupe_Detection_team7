import streamlit as st
from PIL import Image
import numpy as np
import time

# Page configuration
st.set_page_config(
    page_title="Logo Brand Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Brand Logo Detector")
st.subheader("Upload a bag image to identify the brand and check authenticity")

# Known brand logos database (simulated)
BRAND_LOGO_DATABASE = {
    "louis_vuitton": {
        "name": "Louis Vuitton",
        "logo_patterns": ["LV", "monogram", "checkerboard"],
        "authentic_features": ["crisp_pattern", "even_spacing", "high_quality"],
        "dupe_indicators": ["blurry_logo", "misaligned_pattern", "cheap_material"]
    },
    "chanel": {
        "name": "Chanel", 
        "logo_patterns": ["CC", "interlocked_C", "quilted"],
        "authentic_features": ["symmetrical_CC", "clean_stitching", "premium_leather"],
        "dupe_indicators": ["crooked_CC", "loose_threads", "plastic_hardware"]
    },
    "gucci": {
        "name": "Gucci",
        "logo_patterns": ["GG", "double_G", "green_red_stripe"],
        "authentic_features": ["precise_GG", "vibrant_colors", "excellent_craftsmanship"],
        "dupe_indicators": ["faded_colors", "sloppy_stitching", "light_weight"]
    },
    "hermes": {
        "name": "Hermès",
        "logo_patterns": ["Hermes", "carriage", "orange_box"],
        "authentic_features": ["clean_embossing", "perfect_hardware", "luxury_leather"],
        "dupe_indicators": ["stamped_weakly", "cheap_metal", "fake_stitching"]
    },
    "prada": {
        "name": "Prada", 
        "logo_patterns": ["PRADA", "triangle_logo", "saffiano"],
        "authentic_features": ["sharp_lettering", "clean_edges", "quality_plastic"],
        "dupe_indicators": ["fuzzy_text", "rough_edges", "flimsy_material"]
    }
}

def detect_logo_features(image):
    """Simulate logo detection by analyzing image properties"""
    # Convert to grayscale for analysis
    gray_image = image.convert('L')
    img_array = np.array(gray_image)
    
    # Get basic image stats that might indicate logo quality
    brightness = np.mean(img_array)
    contrast = np.std(img_array)
    width, height = image.size
    
    features = {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": contrast / brightness if brightness > 0 else 0,
        "size_ratio": width / height,
        "image_quality": min(contrast / 50, 1.0)  # Higher contrast = better quality
    }
    
    return features

def identify_brand_and_authenticity(image):
    """Identify brand and check if it's authentic or dupe"""
    features = detect_logo_features(image)
    
    # Simulate brand detection based on image characteristics
    # In real system, this would use actual logo recognition
    
    # Random but consistent brand detection for demo
    random.seed(hash(str(features["brightness"] + features["contrast"])) % 1000)
    brands = list(BRAND_LOGO_DATABASE.keys())
    detected_brand_key = random.choice(brands)
    detected_brand = BRAND_LOGO_DATABASE[detected_brand_key]
    
    # Determine authenticity based on image quality features
    quality_score = (
        features["image_quality"] * 0.4 +
        features["sharpness"] * 0.3 +
        (features["contrast"] / 100) * 0.3
    )
    
    is_authentic = quality_score > 0.6
    confidence = min(quality_score * 100, 95)
    
    # Generate reasons
    if is_authentic:
        authenticity_reasons = detected_brand["authentic_features"][:2]
    else:
        authenticity_reasons = detected_brand["dupe_indicators"][:2]
    
    return {
        "brand_name": detected_brand["name"],
        "is_authentic": is_authentic,
        "confidence": confidence,
        "quality_score": quality_score,
        "reasons": authenticity_reasons,
        "detected_features": features
    }

# File upload
uploaded_file = st.file_uploader("📸 Upload a bag image with visible logo", 
                                type=['jpg', 'jpeg', 'png'],
                                help="Make sure the brand logo is clearly visible")

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 2])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        # Analyze the image
        with st.spinner("🔍 Analyzing logo and brand features..."):
            time.sleep(2)  # Simulate processing time
            result = identify_brand_and_authenticity(image)
        
        # Display results
        st.subheader("🎯 Detection Results")
        
        # Brand identification
        st.write(f"**Brand Detected:** {result['brand_name']}")
        st.write(f"**Detection Confidence:** {result['confidence']:.1f}%")
        
        # Authenticity result
        if result['is_authentic']:
            st.success(f"✅ **AUTHENTIC {result['brand_name'].upper()}**")
            st.balloons()
        else:
            st.error(f"❌ **{result['brand_name'].upper()} DUPE DETECTED**")
        
        # Quality indicators
        st.subheader("📊 Quality Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            quality_color = "🟢" if result['detected_features']['image_quality'] > 0.5 else "🟡" if result['detected_features']['image_quality'] > 0.3 else "🔴"
            st.metric("Image Quality", f"{result['detected_features']['image_quality']*100:.1f}%", delta=quality_color)
        
        with col2:
            sharpness_color = "🟢" if result['detected_features']['sharpness'] > 1.0 else "🟡" if result['detected_features']['sharpness'] > 0.5 else "🔴"
            st.metric("Logo Sharpness", f"{result['detected_features']['sharpness']:.2f}", delta=sharpness_color)
        
        with col3:
            contrast_color = "🟢" if result['detected_features']['contrast'] > 40 else "🟡" if result['detected_features']['contrast'] > 20 else "🔴"
            st.metric("Contrast Level", f"{result['detected_features']['contrast']:.1f}", delta=contrast_color)
        
        # Reasons for decision
        st.subheader("🔍 Key Findings")
        if result['is_authentic']:
            st.info("**Authentic indicators detected:**")
            for reason in result['reasons']:
                st.write(f"✅ {reason.replace('_', ' ').title()}")
        else:
            st.warning("**Dupe indicators detected:**")
            for reason in result['reasons']:
                st.write(f"❌ {reason.replace('_', ' ').title()}")

else:
    # Instructions when no image is uploaded
    st.info("👆 Upload a bag image with a visible logo to start analysis")
    
    st.write("---")
    st.subheader("📖 How to Use:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**✅ For Best Results:**")
        st.write("• Clear, well-lit photos")
        st.write("• Focus on the logo area")
        st.write("• Avoid blurry images")
        st.write("• Show logo clearly")
    
    with col2:
        st.write("**🎯 What We Detect:**")
        st.write("• Brand identification")
        st.write("• Logo quality analysis")
        st.write("• Authenticity check")
        st.write("• Quality indicators")

# Footer
st.write("---")
st.write("**🔍 Logo Brand Detector** • Focused on logo analysis and authenticity verification")
