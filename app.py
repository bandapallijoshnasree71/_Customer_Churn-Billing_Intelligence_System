"""
================================================================================
Customer Churn & Billing Predictor Web Application
Built with Streamlit, Plotly, and Scikit-Learn
================================================================================
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

# ------------------------------------------------------------------------------
# 1. Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn & Billing Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# 2. Load Saved Pipeline Artifacts
# ------------------------------------------------------------------------------
@st.cache_resource
def load_pipeline_artifacts():
    return joblib.load("customer_churn_pipeline.joblib")

try:
    artifacts = load_pipeline_artifacts()
    clf_model = artifacts["classification_model"]
    clf_scaler = artifacts["classification_scaler"]
    clf_cols = artifacts["classification_features"]

    reg_model = artifacts["regression_model"]
    reg_scaler = artifacts["regression_scaler"]
    reg_cols = artifacts["regression_features"]
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure 'customer_churn_pipeline.joblib' exists.")
    st.stop()

# ------------------------------------------------------------------------------
# 3. Sidebar Navigation
# ------------------------------------------------------------------------------
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Mode", ["Single Customer Predictor", "Batch CSV Predictor"])

# ==============================================================================
# MODE 1: SINGLE CUSTOMER PREDICTOR
# ==============================================================================
if app_mode == "Single Customer Predictor":
    st.title("📊 Single Customer Churn & Billing Predictor")
    st.write("Configure customer attributes below to predict monthly charges and churn risk in real time.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Contract & Account")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=38)
        payment_method = st.selectbox(
            "Payment Method", 
            ["Credit card (automatic)", "Electronic check", "Mailed check", "Bank transfer (automatic)"]
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col2:
        st.subheader("Services & Billing")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        total_charges = st.number_input("Total Billed Charges ($)", min_value=0.0, max_value=15000.0, value=1000.0)

    st.markdown("---")

    if st.button("Assess Customer Profile", type="primary", use_container_width=True):
        customer_profile = {
            "Contract": contract,
            "tenure": tenure,
            "InternetService": internet_service,
            "TechSupport": tech_support,
            "OnlineSecurity": online_security,
            "PaymentMethod": payment_method,
            "PaperlessBilling": paperless,
            "TotalCharges": total_charges,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "gender": "Female",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No"
        }
        
        input_df = pd.DataFrame([customer_profile])
        encoded_df = pd.get_dummies(input_df)
        
        # Align Classification Features
        clf_features = pd.DataFrame(0, index=[0], columns=clf_cols)
        for col in clf_cols:
            if col in encoded_df.columns:
                clf_features[col] = encoded_df[col].values
                
        # Align Regression Features
        reg_features = pd.DataFrame(0, index=[0], columns=reg_cols)
        for col in reg_cols:
            if col in encoded_df.columns:
                reg_features[col] = encoded_df[col].values
                
        # Predict
        scaled_clf = clf_scaler.transform(clf_features)
        churn_prob = clf_model.predict_proba(scaled_clf)[0][1] * 100
        churn_pred = clf_model.predict(scaled_clf)[0]
        
        scaled_reg = reg_scaler.transform(reg_features)
        est_monthly = reg_model.predict(scaled_reg)[0]
        
        # Display Results
        st.subheader("Evaluation Results")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Estimated Monthly Charges", f"${est_monthly:.2f}")
        res_col2.metric("Churn Risk Probability", f"{churn_prob:.1f}%")
        
        # Churn Risk Indicator Bar
        st.progress(int(churn_prob))

        if churn_prob >= 60.0:
            st.error(f"⚠️ **High Risk:** Customer has a {churn_prob:.1f}% probability of churning. Immediate retention action is recommended!")
        else:
            st.success(f"✅ **Low Risk:** Customer has only a {churn_prob:.1f}% probability of churning. Profile is stable.")

# ==============================================================================
# MODE 2: BATCH CSV PREDICTOR
# ==============================================================================
elif app_mode == "Batch CSV Predictor":
    st.title("📂 Batch Customer Churn & Billing Predictor")
    st.write("Upload a CSV file containing multiple customer records to run automated batch scoring and visual analytics.")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        st.success("CSV file uploaded successfully!")
        
        with st.expander("Preview Uploaded Dataset"):
            st.dataframe(batch_data.head())

        # Prediction execution trigger
        if st.button("Run Batch Predictions", type="primary", use_container_width=True):
            with st.spinner("Processing customer records and executing models..."):
                df_clean = batch_data.copy()
                if "TotalCharges" in df_clean.columns:
                    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce").fillna(0.0)

                # One-hot encode inputs
                batch_encoded = pd.get_dummies(df_clean.drop(columns=["customerID", "Churn"], errors="ignore"))

                # Align features
                X_clf = pd.DataFrame(0, index=batch_encoded.index, columns=clf_cols)
                for col in clf_cols:
                    if col in batch_encoded.columns:
                        X_clf[col] = batch_encoded[col]

                X_reg = pd.DataFrame(0, index=batch_encoded.index, columns=reg_cols)
                for col in reg_cols:
                    if col in batch_encoded.columns:
                        X_reg[col] = batch_encoded[col]

                # Run inferences
                scaled_clf_batch = clf_scaler.transform(X_clf)
                scaled_reg_batch = reg_scaler.transform(X_reg)

                probs = clf_model.predict_proba(scaled_clf_batch)[:, 1]
                preds = clf_model.predict(scaled_clf_batch)
                charges = reg_model.predict(scaled_reg_batch)

                # Assemble results DataFrame
                output_df = batch_data.copy()
                output_df["Predicted_Status"] = np.where(preds == 1, "Likely to Churn", "Likely to Stay")
                output_df["Churn_Risk_Probability_%"] = np.round(probs * 100, 1)
                output_df["Retention_Action_Required"] = np.where(output_df["Churn_Risk_Probability_%"] >= 60.0, "YES", "NO")
                output_df["Predicted_Monthly_Charges"] = np.round(charges, 2)

                # Order columns logically
                priority_cols = [
                    "customerID", 
                    "Predicted_Status", 
                    "Churn_Risk_Probability_%", 
                    "Retention_Action_Required", 
                    "Predicted_Monthly_Charges"
                ]
                existing_priority = [col for col in priority_cols if col in output_df.columns]
                remaining_cols = [col for col in output_df.columns if col not in existing_priority]
                
                st.session_state["batch_results"] = output_df[existing_priority + remaining_cols]

        # Display results if available in session state
        if "batch_results" in st.session_state:
            results = st.session_state["batch_results"]

            st.markdown("---")
            st.subheader("📈 Retention Campaign Summary")

            # High-level Metrics Cards
            total_customers = len(results)
            action_required = (results["Retention_Action_Required"] == "YES").sum()
            action_pct = (action_required / total_customers) * 100
            avg_risk = results["Churn_Risk_Probability_%"].mean()

            # Fixed metric layout to prevent CSS shrinking
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            with kpi_col1:
                st.metric(label="Evaluated Customers", value=f"{total_customers:,}")
            with kpi_col2:
                st.metric(label="Action Required (Risk >= 60%)", value=f"{action_required:,}", delta=f"{action_pct:.1f}% of total")
            with kpi_col3:
                st.metric(label="Average Churn Risk", value=f"{avg_risk:.1f}%")

            st.markdown("---")
            st.subheader("📋 Scored Customer Records")

            view_mode = st.radio(
                "Filter Table View:",
                ["All Evaluated Customers", "High Risk Customers Only (Action Required)"],
                horizontal=True
            )

            if view_mode == "High Risk Customers Only (Action Required)":
                filtered_table = results[results["Retention_Action_Required"] == "YES"]
            else:
                filtered_table = results

            st.dataframe(filtered_table, use_container_width=True)

            # Scatter Plot Visualization
            st.markdown("---")
            st.subheader("Visual Analytics Dashboard")

            fig_scatter = px.scatter(
                results,
                x="Predicted_Monthly_Charges",
                y="Churn_Risk_Probability_%",
                color="Retention_Action_Required",
                hover_data=[c for c in ["customerID", "Contract", "tenure"] if c in results.columns],
                title="Customer Risk vs. Predicted Monthly Charges",
                color_discrete_map={"YES": "#ef4444", "NO": "#10b981"},
                template="plotly_dark"
            )
            fig_scatter.add_hline(y=60.0, line_dash="dash", line_color="red", annotation_text="Action Threshold (60%)")
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Download Predictions
            csv_download = filtered_table.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Displayed Records as CSV",
                data=csv_download,
                file_name="retention_analysis_export.csv",
                mime="text/csv",
                use_container_width=True
            )