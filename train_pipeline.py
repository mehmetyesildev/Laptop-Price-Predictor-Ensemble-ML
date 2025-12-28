import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from category_encoders import TargetEncoder
import time
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

if not os.path.exists('models'):
    os.makedirs('models')

start_time = time.time()

df = pd.read_csv("data/LaptopDataset.csv")

anomalies = df[
    ((df['Processor'].str.contains('i3|Ryzen 3')) & (df['Price ($)'] > 1500)) |
    ((df['Graphics Card'].str.contains('UHD|Iris|Integrated')) & (df['Price ($)'] > 2200))
].index
df = df.drop(anomalies)
print(f"Mantıksız yüksek fiyatlı {len(anomalies)} veri temizlendi.")


df.loc[df['Brand'] == 'Apple', 'Operating System'] = 'MacOS'


Q1 = df['Price ($)'].quantile(0.25)
Q3 = df['Price ($)'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Price ($)'] < (Q1 - 1.5 * IQR)) | (df['Price ($)'] > (Q3 + 1.5 * IQR)))]


bins = [0, 1300, 1800, 2300, 2800, float('inf')]
labels = ['Düşük', 'Orta-Düşük', 'Orta-Yüksek', 'Yüksek', 'Ultra-Yüksek']
df['Price Segment'] = pd.cut(df['Price ($)'], bins=bins, labels=labels, include_lowest=True)

segments_data = {label: df[df['Price Segment'] == label].copy() for label in labels}

np.random.seed(42)
for label, seg_df in segments_data.items():
    if len(seg_df) < 400:
        bootstrap = seg_df.sample(n=300, replace=True, random_state=42)
        segments_data[label] = pd.concat([seg_df, bootstrap], ignore_index=True)


def extract_processor_info(proc_str):
    if not isinstance(proc_str, str): return pd.Series(["Diğer", "Bilinmeyen", "Bilinmeyen", 0])
    proc_str = proc_str.lower()
    brand, series, generation, gen_weight = "Diğer", "Bilinmeyen", "Bilinmeyen", 0
    
    if "intel" in proc_str:
        brand = "Intel"
        series = "i3" if "i3" in proc_str else "i5" if "i5" in proc_str else "i7" if "i7" in proc_str else "i9" if "i9" in proc_str else "Diğer"
        gen_match = re.search(r'(\d{1,2})(?:th|st|nd|rd)', proc_str)
        generation = gen_match.group(1) if gen_match else "11" # Varsayılan
        gen_weight = int(generation)
    elif "ryzen" in proc_str:
        brand = "AMD"
        series = "Ryzen 3" if "3" in proc_str else "Ryzen 5" if "5" in proc_str else "Ryzen 7" if "7" in proc_str else "Ryzen 9" if "9" in proc_str else "Diğer"
        gen_match = re.search(r'(\d{4})', proc_str)
        generation = gen_match.group(1) if gen_match else "5000"
        gen_weight = int(generation[0])
    elif "m1" in proc_str or "m2" in proc_str:
        brand, series = "Apple", "M-Serisi"
        generation = "M1" if "m1" in proc_str else "M2"
        gen_weight = 12
    return pd.Series([brand, series, generation, gen_weight])

def processor_performance_weight(series):
    mapping = {'i3':1, 'Ryzen 3':1, 'i5':2, 'Ryzen 5':2, 'i7':3, 'Ryzen 7':3, 'M-Serisi':3, 'i9':4, 'Ryzen 9':4}
    return mapping.get(series, 1)

def extract_gpu_info(gpu):
    if not isinstance(gpu, str): return pd.Series(["Diğer", 0, "Entegre"])
    gpu = gpu.lower()
    if "nvidia" in gpu: return pd.Series(["NVIDIA", int(re.search(r'\d{3,4}', gpu).group(0)) if re.search(r'\d{3,4}', gpu) else 3050, "Ayrık"])
    if "amd" in gpu: return pd.Series(["AMD", 0, "Ayrık"])
    return pd.Series(["Entegre", 0, "Entegre"])

def train_segment_model(df_segment, segment_name):
    if df_segment.empty: return None, None

    df_segment[['Processor Brand', 'Processor Series', 'Processor Generation', 'Processor_Generation_Weight']] = df_segment['Processor'].apply(extract_processor_info)
    df_segment['Processor Performance Weight'] = df_segment['Processor Series'].apply(processor_performance_weight)
    df_segment[['GPU Brand', 'GPU Model Number', 'GPU Type']] = df_segment['Graphics Card'].apply(extract_gpu_info)
    df_segment['GPU_Model_log'] = np.log1p(df_segment['GPU Model Number'])

    brand_avg_price_series = df_segment.groupby('Brand')['Price ($)'].mean()
    segment_median_price = brand_avg_price_series.median()
    
    segment_stats = {
        'brand_map': brand_avg_price_series.to_dict(),
        'default_price': segment_median_price
    }
    
    df_segment['Brand Avg Price'] = df_segment['Brand'].map(brand_avg_price_series).fillna(segment_median_price)
    
 
    df_segment['RAM_log'] = np.log1p(df_segment['RAM (GB)'])
    df_segment['Storage_log'] = np.log1p(df_segment['Storage (GB)'])
    df_segment['Price_log'] = np.log1p(df_segment['Price ($)'])
    
   
    features = ['RAM_log', 'Storage_log', 'Warranty (years)', 'GPU_Model_log', 'Brand Avg Price', 
                'Processor_Generation_Weight', 'Processor Performance Weight', 'Battery Life (hours)']
    categorical_features = ['Brand', 'Processor Brand', 'GPU Brand']
    
    X = df_segment[features + categorical_features]
    y = df_segment['Price_log']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), features),
        ('cat', TargetEncoder(), categorical_features)
    ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1))
    ])
    
    model.fit(X, y)
    
 
    preds = np.expm1(model.predict(X))
    mae = mean_absolute_error(np.expm1(y), preds)
    print(f"Model ({segment_name}) eğitildi. Eğitim MAE: {mae:.2f}")
    
    return model, segment_stats


segment_metadata_store = {}

for segment_name, df_seg in segments_data.items():
    print(f"\nEğitim: {segment_name} (Veri sayısı: {len(df_seg)})")
    model, stats = train_segment_model(df_seg, segment_name)
    if model:
        joblib.dump(model, f'models/model_{segment_name}.pkl')
        segment_metadata_store[segment_name] = stats

joblib.dump(segment_metadata_store, 'models/segment_metadata.pkl')
print("\nYeni modeller ve metadata başarıyla kaydedildi.")