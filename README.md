# 💻 Laptop Price Predictor (Hybrid Ensemble Architecture)

This project is a professional **Machine Learning-based Backend** solution capable of predicting prices with high consistency even on noisy real-world datasets. The core focus is not just training a model, but overcoming data constraints and variance through advanced engineering approaches.

---

## 🖼️ Application Interface
![Application Screenshot](https://github.com/mehmetyesildev/Laptop-Price-Predictor-Ensemble-ML/raw/main/screenshot.png) 


---

## 🚀 Key Engineering Solutions

* **Hybrid Ensemble Learning**: To minimize variance and error rates of a single algorithm, **LightGBM**, **XGBoost**, and **CatBoost** models are combined using a `StackingRegressor` architecture.
* **Price Segmentation**: The dataset is divided into 5 distinct price segments (Low, Mid-Low, Mid-High, High, Ultra-High) based on hardware power, and specialized sub-models are trained for each.
* **Dynamic Metadata Strategy**: A dynamic inference structure was established that references real brand-based price weights (`segment_metadata.pkl`) extracted during training.
* **Anomaly Detection & Cleaning**: Irrational price deviations (e.g., entry-level processors with exorbitant prices) are automatically cleaned to improve model learning quality.
* **Guardrails (Safety Layers)**: Code-level upper and lower limit controls (e.g., minimum price protection for Apple devices) are implemented to prevent the model from producing outlier results.

---



## 🛠️ Technology Stack

* **Languages & Interface**: Python, Streamlit.
* **Machine Learning**: Scikit-learn, LightGBM, CatBoost, XGBoost.
* **Data Engineering**: Pandas, Numpy, Joblib, Category Encoders (Target Encoding).
* **Architecture**: Pipeline-based Preprocessing, Metadata-driven Inference.

---

## 📂 Project Structure

* `app.py`: Streamlit-based UI and intelligent segment estimation logic.
* `train_pipeline.py`: Data cleaning, segmentation, bootstrapping, and model training pipeline.
* `models/`: Trained models and dynamic metadata files.
* `research/`: Experimental studies and baseline analyses from the development phase.
* `data/`: The dataset used for training (`LaptopDataset.csv`).

---

## ⚙️ Installation and Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
2. Train the Model (Optional): To re-train the model and regenerate metadata with new data:
   ```bash
   python train_pipeline.py
3. Launch the Application:
    ```bash
    streamlit run app.py
💡 Engineering Note
High price deviations were observed in standard regression models due to constraints and noise in the dataset. This project was developed to demonstrate how these deviations can be minimized through Ensemble Learning and Strategic Segmentation, providing a production-ready backend architecture.
