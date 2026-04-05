from src.models.modelA import ModelA
from src.models.modelB import ModelB
from src.models.modelC import ModelC

from src.train import train_model
from src.inference import infer
from src.ensemble import weighted_ensemble
from src.postprocess import smooth

import numpy as np


def main():
    print("🚀 PM2.5 Forecasting Pipeline Started")

    # =========================
    # Initialize Models
    # =========================
    modelA = ModelA()
    modelB = ModelB()
    modelC = ModelC()

    print("✅ Models initialized (A, B, C)")

    # =========================
    # (Placeholder) Data loading
    # =========================
    print("📂 Load your dataset here")

    # Example placeholders (replace with actual pipeline)
    dummy_data = np.random.randn(2, 10, 16, 140, 124)
    dummy_last = np.random.randn(2, 140, 124)

    # =========================
    # Inference (example)
    # =========================
    print("🔮 Running inference...")

    predA = infer(modelA, dummy_data, dummy_last)
    predB = infer(modelB, dummy_data, dummy_last)
    predC = infer(modelC, dummy_data, dummy_last)

    # =========================
    # Ensemble
    # =========================
    final_pred = weighted_ensemble(predA, predB, predC)

    # =========================
    # Postprocessing
    # =========================
    final_pred = smooth(final_pred)

    print("✅ Pipeline complete")
    print("📊 Output shape:", final_pred.shape)


if __name__ == "__main__":
    main()
