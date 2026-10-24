# =============================================================================
elif menu == "🌱 Crop Prediction":
    st.markdown("<div class='main-header'>🌱 Crop Recommendation & Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Interactive Recommendation Engine Powered by Trained Machine Learning</div>", unsafe_allow_html=True)

    if best_model is None or label_encoder is None:
        st.warning("⚠️ Trained model not found. Please train models first:")
        if st.button("🚀 Train Model Now"):
            trigger_training(dataset_path, model_dir)
        st.stop()

    st.write(
        "Enter the soil chemical properties and local environmental/climate parameters below. "
        "The trained machine learning classifier will analyze these inputs and predict the most suitable crop."
    )

    # Input Form
    with st.form("crop_prediction_form"):
        col_soil, col_env = st.columns(2)

        with col_soil:
            st.markdown("#### 🧪 Soil Nutrient Parameters")
            n_val = st.number_input("Nitrogen (N) - kg/ha", min_value=0.0, max_value=150.0, value=90.0, step=1.0,
                                    help="Nitrogen content in the soil (typical range: 0 - 140)")
            p_val = st.number_input("Phosphorus (P) - kg/ha", min_value=5.0, max_value=150.0, value=42.0, step=1.0,
                                    help="Phosphorus content in the soil (typical range: 5 - 145)")
            k_val = st.number_input("Potassium (K) - kg/ha", min_value=5.0, max_value=210.0, value=43.0, step=1.0,
                                    help="Potassium content in the soil (typical range: 5 - 205)")
            ph_val = st.number_input("Soil pH Level", min_value=3.5, max_value=10.0, value=6.5, step=0.1,
                                     help="Soil acidity or alkalinity (typical agricultural range: 4.5 - 8.5)")

        with col_env:
            st.markdown("#### ⛅ Environmental & Climate Parameters")
            temp_val = st.number_input("Temperature (°C)", min_value=5.0, max_value=50.0, value=20.9, step=0.1,
                                       help="Ambient temperature in degrees Celsius (typical range: 8 - 45°C)")
            hum_val = st.number_input("Relative Humidity (%)", min_value=10.0, max_value=100.0, value=82.0, step=0.5,
                                      help="Atmospheric relative humidity percentage (typical range: 15 - 100%)")
            rain_val = st.number_input("Rainfall (mm)", min_value=10.0, max_value=350.0, value=202.9, step=1.0,
                                       help="Annual or seasonal rainfall in millimeters (typical range: 20 - 300 mm)")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🌱 Recommend Suitable Crop")

    # Prediction Processing
    if submitted:
        # Input Validation & Sanity Warnings
        warnings = []
        if ph_val < 4.5:
            warnings.append("Soil pH is strongly acidic (< 4.5). Most crops prefer slightly acidic to neutral soils (5.5 - 7.5).")
        elif ph_val > 8.5:
            warnings.append("Soil pH is strongly alkaline (> 8.5), which can restrict nutrient absorption.")

        if rain_val < 30.0:
            warnings.append("Rainfall is exceptionally low (< 30 mm). Supplemental irrigation will be necessary.")
        elif rain_val > 280.0:
            warnings.append("Rainfall is very high (> 280 mm). Ensure adequate drainage to avoid waterlogging.")

        for w in warnings:
            st.warning(f"⚠️ **Agricultural Note:** {w}")

        # Prepare feature vector with explicit feature names
        input_data = pd.DataFrame(
            [[n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val]],
            columns=feature_list
        )
        
        # Predict class
        pred_encoded = best_model.predict(input_data)[0]
        crop_recommended = label_encoder.inverse_transform([pred_encoded])[0]
        crop_clean = crop_recommended.strip().lower()

        # Display Prominent Recommendation Banner
        st.markdown(f"""
        <div class='recommendation-banner'>
            <div style='font-size: 1.1rem; opacity: 0.9;'>🌱 RECOMMENDED CROP</div>
            <div class='crop-name'>{crop_recommended.upper()}</div>
            <div style='font-size: 1.1rem; opacity: 0.9;'>
                Based on the provided soil and environmental conditions, the trained machine learning model recommends this crop.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Detailed Agronomic Insights
        col_rec_info, col_rec_prob = st.columns([1.2, 1])

        with col_rec_info:
            st.markdown("### 🌾 Crop Agronomic Profile")
            info = CROP_INFO.get(crop_clean, {
                "desc": "Agricultural crop suitable for the input soil and climatic parameters.",
                "season": "Seasonal",
                "soil": "Fertile Loamy Soil"
            })
            st.markdown(f"**Description:** {info['desc']}")
            st.markdown(f"**Preferred Cropping Season:** {info['season']}")
            st.markdown(f"**Typical Soil Requirement:** {info['soil']}")

        with col_rec_prob:
            st.markdown("### 📊 Prediction Confidence")
            if hasattr(best_model, "predict_proba"):
                probs = best_model.predict_proba(input_data)[0]
                top_indices = np.argsort(probs)[::-1][:3]
                
                top_crops = label_encoder.inverse_transform(top_indices)
                top_scores = probs[top_indices]

                for crop, score in zip(top_crops, top_scores):
                    st.write(f"**{crop.capitalize()}**: `{score*100:.1f}%`")
                    st.progress(float(score))
            else:
                st.info("Direct prediction from Decision Tree classifier.")

        # st.markdown("""
        # <div style='background-color: #f1f8e9; border: 1px solid #c8e6c9; border-radius: 8px; padding: 1rem; margin-top: 1.5rem;'>
        #     <b>⚠️ Important Notice:</b> This is an educational Data Science recommendation derived from historical data.
        #     Local factors such as soil micronutrients, irrigation availability, and regional weather variations should always
        #     be consulted with local agricultural experts before planting.
        # </div>
        # """, unsafe_allow_html=True)
