import pandas as pd
import numpy as np

def predict_all_airfoils(model, scaler, alpha, reynolds, airfoil_labels):
    n_airfoils = len(airfoil_labels)
    encoded_airfoils = list(range(n_airfoils))

    # Build input rows
    X_query = pd.DataFrame({
        'Alpha': [alpha] * n_airfoils,
        'Reynolds': [reynolds] * n_airfoils,
        'Airfoil': encoded_airfoils
    })

    # Scale
    X_scaled = scaler.transform(X_query)

    # Predict Cl/Cd
    preds = model.predict(X_scaled)

    return preds

def select_best_airfoil(preds, airfoil_labels):
    best_idx = np.argmax(preds)
    best_airfoil = airfoil_labels[best_idx]
    best_clcd = preds[best_idx]

    return best_airfoil, best_clcd