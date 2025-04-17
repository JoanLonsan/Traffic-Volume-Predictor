# 🚗 Traffic Volume Predictor with LSTM

Build and train a time series prediction model using PyTorch to forecast traffic volume based on historical data — originally sourced from the Metro Interstate Traffic dataset (UCI), and adaptable to any city (e.g., Madrid 🇪🇸).

---

## 🚀 Getting Started

✅ Prerequisites  
Before you begin:

- Make sure you have Python 3.8+ and `pip` installed.
- Install the required packages listed in the notebook using `pip install -r requirements.txt` or run the `%pip install` cells directly inside the notebook.
- For visualization, make sure `plotly` is installed (`pip install plotly`).

---

## 📘 Project Overview

In this notebook, you will:

- Load and preprocess a real-world traffic dataset.
- Create training and testing splits while respecting temporal order (no shuffling).
- Construct time-based sequences for supervised learning.
- Define and train an **LSTM neural network** using PyTorch.
- Evaluate model performance using **MSE** and visualize predictions.
- Learn how to scale this solution to your city or location (e.g. Madrid's Open Data).

---

## 🧠 Key Concepts

- Time series modeling with sequence windows
- `torch.nn.LSTM` architecture
- Supervised learning from temporal data
- DataLoader batching and sequence handling
- Inverse transformation of normalized predictions for real-world interpretability

---

## 📂 Files

- `traffic_volume_Predictor.ipynb` – Main notebook with all code, training, evaluation, and plots.
- `README.md` – You're reading it!

---

## 🌍 Customizing for Your Location

This project was built using data from the UCI ML Repository, but it can be easily adapted to any city with traffic data.

Check the final section in the notebook for a step-by-step guide on how to do this.

---

## 📈 Example Output

The model achieves low MSE on the test set and produces time-aligned predictions that closely follow the true traffic pattern:

> Test MSE: ~0.002  
> *(depends on training seed and sequence length)*

---

## 💡 Pro Tip

Want to test your own dataset? Just replace the CSV and update the preprocessing section. The LSTM model and sequence logic remain the same!

---

## 📎 Extras

- Consider tuning hyperparameters like `seq_length`, `hidden_size`, or `batch_size` for better results.
- Add external features like weather or event data for improved forecasting.
