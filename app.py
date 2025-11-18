import streamlit as st
from PIL import Image
import numpy as np
import time

# Page configuration
st.set_page_config(
    page_title="Bag Dupe Detector",
    page_icon="👜",
    layout="wide"
)

st.title("👜 Bag Dupe Detector")
st.subheader("Upload a bag image - We'll detect if it's branded or a dupe")

def simple_dupe_detection(image):
    """Simple analysis to detect if bag is branded or dupe"""
    # Convert to array for analysis
    img_array = np.array(image.convert('RGB'))
    
    # Get image dimensions
    height, width = img_array.shape[:2]
    
    # Analyze basic image properties
    brightness = np.mean(img_array)
    contrast = np.std(img_array)
    
    # Simple pattern detection (logos usually have high contrast)
    gray = np.mean(img_array, axis=2)
    edges = np.mean(np.abs(np.diff(gray, axis=0))) + np.mean(np.abs(np.diff(gray, axis=1)))
    
    # Determine if branded based on image characteristics
    pattern_strength = (contrast / 100) + (edges / 100)
    
    if pattern_strength > 0.8:
        result = "BRANDED BAG"
        confidence = min(pattern_strength * 100, 95)
        reason = "Strong patterns and contrast detected - likely authentic brand"
    elif pattern_strength > 0.5:
        result = "POSSIBLE BRANDED BAG" 
        confidence = pattern_strength * 80
        reason = "Moderate patterns detected - could be branded"
    else:
        result = "DUPE/GENERIC BAG"
        confidence = (1 - pattern_strength) * 100
        reason = "Low pattern quality - likely dupe or generic bag"
    
    return {
        "result": result,
        "confidence": confidence,
        "reason": reason,
        "pattern_strength": pattern_strength,
        "brightness": brightness,
        "contrast": contrast
    }

# File upload
uploaded_file = st.file_uploader("📸 Upload a bag image", 
                                type=['jpg', 'jpeg', 'png'],
                                help="We'll analyze if it's branded or a dupe")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Bag", use_column_width=True)
    
    with col2:
        # Analyze the image
        with st.spinner("🔍 Analyzing bag patterns..."):
            time.sleep(2)
            result = simple_dupe_detection(image)
        
        # Display results
        st.subheader("🎯 Detection Result")
        
        if "BRANDED" in result["result"]:
            st.success(f"✅ **{result['result']}**")
        else:
            st.error(f"❌ **{result['result']}**")
        
        st.write(f"**Confidence:** {result['confidence']:.1f}%")
        st.write(f"**Reason:** {result['reason']}")
        
        # Show technical details
        st.subheader("📊 Analysis Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Pattern Strength", f"{result['pattern_strength']:.2f}")
        
        with col2:
            st.metric("Brightness", f"{result['brightness']:.1f}")
        
        with col3:
            st.metric("Contrast", f"{result['contrast']:.1f}")

else:
    # Instructions
    st.info("👆 Upload a bag image to check if it's branded or a dupe")
    
    st.write("---")
    st.subheader("🎯 What We Detect:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**✅ Branded Bags Have:**")
        st.write("• High contrast patterns")
        st.write("• Clear logos")
        st.write("• Quality materials")
        st.write("• Strong edges")
    
    with col2:
        st.write("**❌ Dupes/Generic Have:**")
        st.write("• Low contrast")
        st.write("• Blurry patterns") 
        st.write("• Poor quality")
        st.write("• Weak edges")

# Footer
st.write("---")
st.write("**👜 Simple Bag Dupe Detector** • No complex analysis • Just tells you if it's branded or not")
