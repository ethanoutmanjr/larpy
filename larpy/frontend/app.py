"""Larpy Streamlit Frontend — Object Price Estimator."""

import streamlit as st
import requests
import tempfile
import os
from PIL import Image
import json
import time

# Page config
st.set_page_config(
    page_title="Larpy — Object Price Estimator",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #e0e0e0;
    }
    .stButton>button {
        background: #e94560;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background: #ff6b81;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("🏷️ Larpy — Object Price Estimator")
    st.markdown("**Upload an image to estimate how expensive an object is on a scale of 1-10**")

    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    model_name = st.sidebar.selectbox(
        "Model",
        ["efficientnet_b0", "efficientnet_b3", "efficientnet_b7", "vit_base", "resnet50", "clip_vit_l14"],
        help="Select the ML model for price estimation",
    )
    api_url = st.sidebar.text_input(
        "API URL",
        "http://localhost:8000",
        help="URL of the Larpy API server",
    )

    st.sidebar.markdown("---")
    st.sidebar.info("""
    ### How It Works
    1. Upload an image of an object
    2. The ML model predicts a price tier (1-10)
    3. Results are processed asynchronously via Redis Queue
    """)

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a photo of the object you want to evaluate",
        )

        if uploaded_file:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Save to temp file for API
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp_file.write(uploaded_file.getvalue())
            temp_file.close()

            st.markdown("---")

            # Estimate button
            if st.button("🔮 Estimate Price", use_container_width=True):
                with st.spinner("Estimating price..."):
                    try:
                        # Upload to API
                        with open(temp_file.name, "rb") as f:
                            files = {"file": (uploaded_file.name, f, uploaded_file.type)}
                            data = {"model_name": model_name}
                            response = requests.post(
                                f"{api_url}/estimate/sync",
                                files=files,
                                data=data,
                                timeout=30,
                            )

                        if response.status_code == 200:
                            result = response.json()
                        else:
                            st.error(f"API Error: {response.text}")
                            result = None

                    except requests.exceptions.ConnectionError:
                        st.error("❌ Could not connect to the API server. Make sure it's running!")
                        result = None
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        result = None
                    finally:
                        os.unlink(temp_file.name)

                # Display results
                if result:
                    with col2:
                        st.subheader("📊 Results")

                        if result.get("status") == "success":
                            price_tier = result.get("price_tier", "?")
                            raw_score = result.get("raw_score", 0)

                            # Display price tier with visual gauge
                            st.markdown(f"### 🏷️ Estimated Price Tier: **{price_tier}/10**")

                            # Progress bar as price gauge
                            st.progress(price_tier / 10, text=f"Price Level: {price_tier}/10")

                            # Raw score
                            st.metric("Raw Score", f"{raw_score:.4f}")

                            # Interpretation
                            if price_tier <= 3:
                                st.success("💰 Budget-friendly item")
                            elif price_tier <= 6:
                                st.info("💵 Mid-range item")
                            elif price_tier <= 8:
                                st.warning("💎 Premium item")
                            else:
                                st.error("👑 Luxury item")

                        elif result.get("status") == "queued":
                            job_id = result.get("job_id")
                            st.info(f"⏳ Job queued! ID: `{job_id}`")

                            # Check status
                            for _ in range(10):
                                time.sleep(1)
                                status_resp = requests.get(f"{api_url}/queue/status/{job_id}")
                                if status_resp.status_code == 200:
                                    status_data = status_resp.json()
                                    if status_data.get("status") == "finished":
                                        st.success("✅ Estimation complete!")
                                        break
                                    elif status_data.get("status") == "failed":
                                        st.error("❌ Job failed")
                                        break

                        st.markdown("---")
                        st.caption(f"Model: {model_name} | API: {api_url}")

    # Info section
    with st.expander("📖 How to Use", expanded=False):
        st.markdown("""
        ### Setup Instructions

        1. **Start Redis**
           ```bash
           redis-server
           ```

        2. **Start the RQ Worker**
           ```bash
           python -m larpy.workers.worker
           ```

        3. **Start the API Server**
           ```bash
           python -m larpy.api.server
           ```

        4. **Start the Frontend**
           ```bash
           streamlit run larpy/frontend/app.py
           ```

        ### Training a Model
        ```bash
        python -m larpy.models.train --model efficientnet_b0 --data ./data/
        ```

        ### Available Models
        | Model | Description | Speed | Accuracy |
        |-------|-------------|-------|----------|
        | EfficientNetB0 | Lightweight, fast | ⚡⚡⚡ | ⭐⭐⭐ |
        | EfficientNetB7 | High accuracy | ⚡ | ⭐⭐⭐⭐⭐ |
        | ViT Base | State-of-art vision | ⚡⚡ | ⭐⭐⭐⭐ |
        | ResNet50 | Reliable baseline | ⚡⚡⚡ | ⭐⭐⭐ |
        | CLIP ViT-L/14 | Multimodal (image+text) | ⚡ | ⭐⭐⭐⭐ |
        """)

    # Footer
    st.markdown("---")
    st.caption("Built with ❤️ using Python, RQ, Streamlit, and Machine Learning")


if __name__ == "__main__":
    main()
