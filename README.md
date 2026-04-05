# 🇮🇳 PM2.5 Multi-Model Ensemble Forecasting (AISEHack Phase 2)

## 🚀 Overview
This project focuses on forecasting short-term PM2.5 concentration across India using advanced spatio-temporal deep learning models. The goal is to achieve robust performance in both normal conditions and extreme pollution episodes.

We propose a multi-model ensemble framework combining recurrent, convolutional, and spectral learning techniques to capture both local spatial patterns and global atmospheric dynamics.

---

## 🧠 Problem Type
- Time Series Forecasting  
- Spatio-temporal Modeling  
- Environmental AI  

---

## 📊 Dataset
- Source: WRF-Chem simulation data  
- Spatial Resolution: 140 × 124 grid (~25 km)  
- Temporal Input: 10 timesteps  
- Prediction Horizon: 16 timesteps  

---

## 🎯 Output Requirement
Final prediction shape:
(218, 140, 124, 16)

---

## 🧠 Model Architecture

### Multi-Model Ensemble

We use an ensemble of three models:

- Model A: ConvLSTM + FNO Hybrid  
- Model B: ConvGRU + CNN  
- Model C: Re-trained Model A (different initialization)

---

## ⚙️ Key Features

- Episode-aware learning strategy  
- Log1p transformation for skewed PM2.5 distribution  
- Direct SMAPE optimization  
- Pearson correlation loss integration  
- Multi-model ensemble for robustness  
- Test Time Augmentation (TTA)  
- Spike-preserving Gaussian smoothing  

---

## 🔬 Innovation Highlights

- Hybrid spectral + spatial modeling (FNO + CNN)  
- Ensemble learning for improved generalization  
- Episode-focused optimization for extreme events  
- Domain-aware preprocessing (log transform)  
- Metric-aligned loss function  

---

## 📈 Training Strategy

- Sliding window temporal sampling  
- Feature-wise normalization + log transform  
- Multi-model independent training  
- Validation-based model selection  
- Ensemble weighting for final predictions  

---

## 🔁 Pipeline

Raw Data  
↓  
Log Transformation  
↓  
Normalization  
↓  
Dataset Construction  
↓  
Model Training (A, B, C)  
↓  
Validation & Selection  
↓  
Inference  
↓  
Ensemble  
↓  
Post-processing  
↓  
Final preds.npy  

---

## 📊 Evaluation Metrics

- Global SMAPE  
- Episode SMAPE  
- Episode Correlation  

---

## 🧪 Post-processing

- Gaussian smoothing to reduce noise  
- Spike preservation to maintain episodic peaks  

---

## 🛠️ Tech Stack

### Core
- Python  
- PyTorch  

### Data Processing
- NumPy  
- SciPy  

### Time Series
- Statsmodels  

### Utilities
- tqdm  
- scikit-learn  

### Hardware
- CUDA (GPU acceleration)  

---

## 📁 Project Structure

pm25-multimodel-ensemble/  
│── src/  
│   ├── config.py  
│   ├── data_loader.py  
│   ├── dataset.py  
│   ├── loss.py  
│   ├── train.py  
│   ├── inference.py  
│   ├── ensemble.py  
│   ├── postprocess.py  
│   │  
│   ├── models/  
│   │   ├── modelA.py  
│   │   ├── modelB.py  
│   │   ├── modelC.py  
│  
│── main.py  
│── requirements.txt  
│── README.md  
│── LICENSE  

---

## 🚀 How to Run

pip install -r requirements.txt  
python main.py  

---

## 🔗 Kaggle Notebook

https://www.kaggle.com/code/aadrikagupta21/notebook16/edit  

---

## 🔗 Model Checkpoints

- Model A , Model B , Model C: https://www.kaggle.com/code/aadrikagupta21/notebook16/edit 


---

## ❌ Negative Results / What Didn’t Work

- Single-model approaches plateaued early  
- CNN-only models failed to capture temporal dependencies  
- No log transformation led to unstable SMAPE  
- Heavy smoothing reduced episodic accuracy  

---

## 👥 Team

- Aadrika Gupta  
- Sangyan Hari Pushkar  
- Mayur Mundada  

---

## 📜 License

This project follows the ANRF Open License.  
Full license document is available in the repository.

---

## 🌍 Impact

Accurate PM2.5 forecasting enables:
- Early warnings for pollution spikes  
- Better urban planning  
- Public health protection  
- Policy-level decision making  

---

## 🔥 Final Note

This project is designed for real-world deployment, ensuring reliable predictions during critical pollution episodes where accuracy matters the most.
