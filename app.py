import streamlit as st
import pandas as pd
import random

# Page configuration
st.set_page_config(
    page_title="Fashion Dupe Finder",
    page_icon="🛍️",
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
    .match-badge {
        background-color: #4CAF50;
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
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="main-header">Fashion Dupe Finder</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Find affordable alternatives to your favorite fashion items</div>', unsafe_allow_html=True)

# Upload Section
st.markdown("### Upload Product Image")
st.markdown('<div class="upload-section">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=['jpg', 'jpeg', 'png'],
    label_visibility="collapsed"
)

st.markdown('</div>', unsafe_allow_html=True)

# Mock data for similar products (replace with your actual ML model results)
def generate_mock_products():
    return [
        {
            "name": "Designer Style Crossbody Bag",
            "retailer": "Amazon Fashion",
            "price": 35.99,
            "original_price": 198.00,
            "savings": 162.01,
            "savings_percent": 81.8,
            "rating": 4.3,
            "reviews": 892,
            "shipping": "FREE delivery",
            "match_score": 89
        },
        {
            "name": "Fashion Leather Crossbody",
            "retailer": "Fashion Retailer",
            "price": 42.50,
            "original_price": 185.00,
            "savings": 142.50,
            "savings_percent": 77.0,
            "rating": 4.1,
            "reviews": 456,
            "shipping": "FREE shipping",
            "match_score": 85
        },
        {
            "name": "Trendy Shoulder Bag",
            "retailer": "Style Outlet",
            "price": 28.99,
            "original_price": 120.00,
            "savings": 91.01,
            "savings_percent": 75.8,
            "rating": 4.4,
            "reviews": 321,
            "shipping": "$4.99 delivery",
            "match_score": 82
        }
    ]

# Display uploaded image and results
if uploaded_file is not None:
    # Display uploaded image
    st.markdown("### Uploaded Product")
    st.image(uploaded_file, width=300)
    
    # Show loading and then display results
    with st.spinner('Finding similar products...'):
        # Simulate processing time
        import time
        time.sleep(2)
        
        st.markdown("### Similar Products Found")
        st.markdown("**Showing 3 affordable alternatives**")
        
        # Get mock products
        products = generate_mock_products()
        
        # Display each product card
        for i, product in enumerate(products, 1):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"#### Alternative #{i}")
                st.markdown(f"**{product['name']}**")
                st.markdown(f"**Retailer:** {product['retailer']}")
                st.markdown(f"**Price:** ${product['price']}")
                st.markdown(f'<div class="price-save">Save ${product["savings"]} ({product["savings_percent"]}%)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="rating">Rating: {product["rating"]}/5 ({product["reviews"]} reviews)</div>', unsafe_allow_html=True)
                st.markdown(f"**Shipping:** {product['shipping']}")
            
            with col2:
                st.markdown(f'<div class="match-badge">{product["match_score"]}% Match</div>', unsafe_allow_html=True)
                st.button("View Product", key=f"btn_{i}")
            
            st.markdown("---")

else:
    st.info("👆 Please upload a product image to find affordable alternatives")

# Footer
st.markdown("---")
st.markdown("*Fashion Dupe Finder · Streamlit*")
