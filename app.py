import streamlit as st
import joblib
from lib.modeling import predict_all_airfoils, select_best_airfoil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load saved objects
model = joblib.load("./artifacts/best_model.pkl")
scaler = joblib.load("./artifacts/scaler.pkl")
airfoil_labels = joblib.load("./artifacts/labelencoder.pkl")

st.title("Airfoil Selector App")
st.write("Predict Cl/Cd for all airfoils and select the best one.")
st.write("This app uses a trained machine learning model to predict the lift-to-drag ratio (Cl/Cd) for different airfoils based on the angle of attack (alpha) and Reynolds number. You can input your desired alpha and Reynolds number, and the app will show you the predicted Cl/Cd for each airfoil, as well as highlight the best-performing airfoil.")

st.write("\nThe inspiration for this app came from a thesis done in 2023, which was trying to see the possibility of using machine learning to create a morphing wing functionality. The idea is to use the model to quickly evaluate many airfoil shapes and find the optimal one for given flight conditions, which could then be used in a morphing wing design.")
st.write("More information can be found in the slides in this web page. However, for more details, please use thanakorn.jark@gmail.com to contact me directly.")
st.write("**The range of alpha trained is from -20 to 20 degrees, and Reynolds number is from 50,000 to 1,000,000. The model may not perform well outside of these ranges.**")
st.write("Enter the angle of attack (alpha) and Reynolds number to get predictions. This is to simulate a scenario where you want to find the best airfoil for specific flight conditions real time.")

# Streamlit inputs
alpha = st.number_input("Angle of attack (alpha)", value=5.0)
reynolds = st.number_input("Reynolds number", value=500000)

if st.button("Run Prediction"):
    # Step 1: predict Cl/Cd for all airfoils
    preds = predict_all_airfoils(model, scaler, alpha, reynolds, airfoil_labels)

    # if not select_best:
    st.subheader("Predicted Cl/Cd for all airfoils")
    for label, pred in zip(airfoil_labels, preds):
        st.write(f"**{label}**: {pred:.4f}")
    # else:
    # Step 2: pick the best airfoil
    best_airfoil, best_clcd = select_best_airfoil(preds, airfoil_labels)
    st.success(f"Best airfoil: **{best_airfoil}** with Predicted Cl/Cd: **{best_clcd:.4f}**")
    # st.write(f"Predicted Cl/Cd: **{best_clcd:.4f}**")

    tab_predicted, tab_info, slide = st.tabs([
    "Predicted Cl/Cd vs Alpha",
    "Model Information",
    "Slides"
])

    with tab_predicted:
        alphas = np.linspace(-20, 20, 100)
        best_idx = list(airfoil_labels).index(best_airfoil)
        X = pd.DataFrame({
            "Alpha": alphas,
            "Reynolds": [reynolds] * len(alphas),
            "Airfoil": [best_idx] * len(alphas)
        })
        X_scaled = scaler.transform(X)
        preds_curve = model.predict(X_scaled)

        # Find max Cl/Cd point (slope = 0)
        max_idx = np.argmax(preds_curve)
        alpha_peak = alphas[max_idx]
        clcd_peak = preds_curve[max_idx]

        fig, ax = plt.subplots()
        ax.plot(alphas, preds_curve, color="red", linewidth=2)

        # Mark the peak
        ax.scatter(alpha_peak, clcd_peak, color="black", s=160, alpha=1, label=f"Cl/Cd Max={clcd_peak:.2f} \nat α={alpha_peak:.2f}")
        # ax.annotate(
        #     f"\nCl/Cd Max={clcd_peak:.2f}\n at α={alpha_peak:.2f}",
        #     (alpha_peak, clcd_peak),
        #     textcoords="offset points",
        #     xytext=(-30, -40),
        #     ha='left'
        # )
        ax.scatter(alpha, best_clcd, color="green", s=60, alpha=1, label=f"Cl/Cd={best_clcd:.2f} \nat α={alpha}")
        # ax.axvline(alpha_peak, color='green', linestyle='--', linewidth=1)
        ax.set_xlabel("Alpha (deg)")
        ax.set_ylabel("Predicted Cl/Cd")
        ax.set_title(f"Predicted Cl/Cd vs Alpha for {best_airfoil} at Re={reynolds}")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)
        st.pyplot(fig)

    with tab_info:
        st.subheader("Model Performance Metrics")
        st.write("These metrics were computed on the test set:")

        st.metric("Mean Absolute Error", "3.046455")
        st.metric("Mean Squared Error", "17.416994")
        st.metric("R² Score", "0.990954")

        st.info("Model: Multi-Layer Perceptron (MLP)")

    with slide:
        st.subheader("Presentation Slides")
        st.write("Here are the slides that provide more details about the project, including the motivation, methodology, and results.")
        st.write("Please note that the slides are done during 2023 as a bachelor thesis, so the content may not be fully up to date with the latest code and results.")
        # st.markdown("[Download Slides](
        with open("Aerodynamic Coefficients Prediction Presentation.pdf", "rb") as f:
            st.download_button(
                label="Download PDF",
                data=f,
                file_name="Aerodynamic Coefficients Prediction Presentation.pdf",
                mime="application/pdf"
            )

        st.pdf("Aerodynamic Coefficients Prediction Presentation.pdf")

        # with open("Aerodynamic Coefficients Prediction Presentation.pdf", "rb") as f:
        #     base64_pdf = base64.b64encode(f.read()).decode("utf-8")

        # pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px"></iframe>'
        # st.markdown(pdf_display, unsafe_allow_html=True)
