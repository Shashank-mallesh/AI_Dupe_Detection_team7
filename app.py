import streamlit as st
import pandas as pd
import random
from PIL import Image
import io

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
    .category-badge {
        background-color: #6c5ce7;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Mock product database for different categories
PRODUCT_DATABASE = {
    "bags": [
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
        }
    ],
    "shoes": [
        {
            "name": "Running Sneakers - Premium Comfort",
            "retailer": "Shoe Palace",
            "price": 45.99,
            "original_price": 150.00,
            "savings": 104.01,
            "savings_percent": 69.3,
            "rating": 4.5,
            "reviews": 1245,
            "shipping": "FREE delivery",
            "match_score": 92
        },
        {
            "name": "Casual Lifestyle Shoes",
            "retailer": "Footwear Express",
            "price": 39.99,
            "original_price": 120.00,
            "savings": 80.01,
            "savings_percent": 66.7,
            "rating": 4.2,
            "reviews": 678,
            "shipping": "FREE shipping",
            "match_score": 87
        }
    ],
    "shirts": [
        {
            "name": "Premium Cotton T-Shirt",
            "retailer": "Clothing Co.",
            "price": 19.99,
            "original_price": 65.00,
            "savings": 45.01,
            "savings_percent": 69.2,
            "rating": 4.4,
            "reviews": 1567,
            "shipping": "FREE delivery",
            "match_score": 91
        },
        {
            "name": "Designer Style Button-Down Shirt",
            "retailer": "Fashion Outlet",
            "price": 29.99,
            "original_price": 95.00,
            "savings": 65.01,
            "savings_percent": 68.4,
            "rating": 4.3,
            "reviews": 892,
            "shipping": "$2.99 delivery",
            "match_score": 88
        }
    ],
    "dresses": [
        {
            "name": "Elegant Summer Dress",
            "retailer": "Style Boutique",
            "price": 34.99,
            "original_price": 120.00,
            "savings": 85.01,
            "savings_percent": 70.8,
            "rating": 4.6,
            "reviews": 2341,
            "shipping": "FREE delivery",
            "match_score": 94
        }
    ],
    "jeans": [
        {
            "name": "Slim Fit Denim Jeans",
            "retailer": "Denim World",
            "price": 41.99,
            "original_price": 110.00,
            "savings": 68.01,
            "savings_percent": 61.8,
            "rating": 4.3,
            "reviews": 1567,
            "shipping": "FREE delivery",
            "match_score": 86
        }
    ]
}

def detect_category_from_filename(filename):
    """Simple category detection based on filename keywords"""
    filename_lower = filename.lower()
    
    if any(word in filename_lower for word in ['shoe', 'sneaker', 'boot', 'footwear']):
        return "shoes"
    elif any(word in filename_lower for word in ['shirt', 'top', 'tshirt', 'blouse']):
        return "shirts"
    elif any(word in filename_lower for word in ['dress', 'gown', 'frock']):
        return "dresses"
    elif any(word in filename_lower for word in ['jean', 'pant', 'trouser']):
        return "jeans"
    elif any(word in filename_lower for word in ['bag', 'purse', 'handbag', 'backpack']):
        return "bags"
    else:
        # Return random category if no match found
        return random.choice(list(PRODUCT_DATABASE.keys()))

def get_similar_products(category, num_products=3):
    """Get similar products based on category"""
    if category in PRODUCT_DATABASE:
        products = PRODUCT_DATABASE[category]
        # Return requested number of products (or all available if less)
        return products[:min(num_products, len(products))]
    else:
        # Return random products if category not found
        random_category = random.choice(list(PRODUCT_DATABASE.keys()))
        return PRODUCT_DATABASE[random_category][:num_products]

# Header Section
st.markdown('<div class="main-header">Fashion Dupe Finder</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Find affordable alternatives to your favorite fashion items</div>', unsafe_allow_html=True)

# Upload Section
st.markdown("### Upload Product Image")
st.markdown('<div class="upload-section">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=['jpg', 'jpeg', 'png'],
    label_visibility="collapsed",
    help="Upload images of bags, shoes, shirts, dresses, or any fashion item"
)

st.markdown('</div>', unsafe_allow_html=True)

# Display uploaded image and results
if uploaded_file is not None:
    # Detect category from filename
    detected_category = detect_category_from_filename(uploaded_file.name)
    
    # Display uploaded image
    st.markdown("### Uploaded Product")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(uploaded_file, width=300, caption=f"Detected: {detected_category.title()}")
    
    with col2:
        st.markdown(f'<div class="category-badge">Category: {detected_category.title()}</div>', unsafe_allow_html=True)
        st.info(f"🔍 Finding affordable {detected_category} alternatives...")
    
    # Show loading and then display results
    with st.spinner(f'Finding similar {detected_category}...'):
        # Simulate processing time
        import time
        time.sleep(2)
        
        st.markdown("### Similar Products Found")
        st.markdown(f"**Showing 3 affordable {detected_category} alternatives**")
        
        # Get products based on detected category
        products = get_similar_products(detected_category, 3)
        
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
    st.info("👆 Please upload a fashion item image (shoes, bags, shirts, dresses, jeans, etc.)")

# Footer
st.markdown("---")
st.markdown("*Fashion Dupe Finder · Streamlit*")
st.markdown("**Supported categories:** Bags, Shoes, Shirts, Dresses, Jeans, and more!")
