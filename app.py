import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re


MODEL_DIR = "models"
METADATA_PATH = os.path.join(MODEL_DIR, "segment_metadata.pkl")

@st.cache_resource
def load_resources():
    if os.path.exists(METADATA_PATH):
        return joblib.load(METADATA_PATH)
    return {}

def load_model(segment):
    path = os.path.join(MODEL_DIR, f"model_{segment}.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def extract_processor_info(proc_str):
    proc_str = str(proc_str).lower()
    if "intel" in proc_str: 
        series = "i7" if "i7" in proc_str else "i5" if "i5" in proc_str else "i3" if "i3" in proc_str else "i9"
        gen = 12 if "12" in proc_str else 11
        return "Intel", series, str(gen), gen
    if "ryzen" in proc_str:
        series = "Ryzen 7" if "7" in proc_str else "Ryzen 5" if "5" in proc_str else "Ryzen 3" if "3" in proc_str else "Ryzen 9"
        gen = 5000
        return "AMD", series, str(gen), 5
    if "m1" in proc_str: return "Apple", "M-Serisi", "M1", 11
    if "m2" in proc_str: return "Apple", "M-Serisi", "M2", 12
    if "m3" in proc_str: return "Apple", "M-Serisi", "M3", 13 
    return "Diğer", "Bilinmeyen", "Bilinmeyen", 0

def processor_performance_weight(series):
    mapping = {'i3':1, 'Ryzen 3':1, 'i5':2, 'Ryzen 5':2, 'i7':3, 'Ryzen 7':3, 'M-Serisi':3, 'i9':4, 'Ryzen 9':4}
    return mapping.get(series, 1)

def extract_gpu_info(gpu):
    gpu = str(gpu).lower()
    if "nvidia" in gpu: 
        
        match = re.search(r'\d{3,4}', gpu)
        model = int(match.group(0)) if match else 3050
        return "NVIDIA", model, "Ayrık"
    if "apple" in gpu:
        return "Apple", 0, "Entegre"
    return "Entegre", 0, "Entegre"


def guess_segment(proc_perf_weight, ram_gb, storage_gb, gpu_type, gpu_model):
   
    score = proc_perf_weight * 2 + (ram_gb / 8) + (storage_gb / 512)
    
   
    if gpu_type == "Ayrık":
        score += 1.5
       
        if gpu_model >= 3060: 
            score += 4.0 
            
   
    if score < 6.0: return "Düşük"
    elif score < 9.0: return "Orta-Düşük" 
    elif score < 12.0: return "Orta-Yüksek"
    elif score < 15.0: return "Yüksek"
    else: return "Ultra-Yüksek"


def main():
    st.set_page_config(page_title="Laptop Fiyat Tahmini", layout="centered")
    st.title("💻 Laptop Fiyat Tahmincisi")
    
    metadata = load_resources()

   
    col1, col2 = st.columns(2)
    
    with col1:
        brand = st.selectbox("Marka", ["Apple", "Dell", "HP", "Lenovo", "Asus", "Acer", "MSI", "Razer", "Diğer"])
        
     
        if brand == "Apple":
            proc_list = ["Apple M1", "Apple M1 Pro", "Apple M2", "Apple M2 Pro", "Apple M3"]
            default_proc = 0
            
            
            gpu_list = ["Apple 7-Core GPU", "Apple 8-Core GPU", "Apple 10-Core GPU (M1 Pro/Max)"]
            default_gpu = 1
            
            os_type_fixed = "Mac"
            
        else:
           
            proc_list = [
                "Intel i3 11th Gen", "Intel i5 11th Gen", "Intel i5 12th Gen", 
                "Intel i7 11th Gen", "Intel i7 12th Gen", "Intel i9 12th Gen",
                "AMD Ryzen 3 3250U", "AMD Ryzen 5 5600H", "AMD Ryzen 7 5800H", "AMD Ryzen 9 5900HX"
            ]
            default_proc = 1
            
            gpu_list = [
                "Intel UHD Graphics", "Intel Iris Xe",
                "NVIDIA RTX 3050", "NVIDIA RTX 3050 Ti", "NVIDIA RTX 3060", 
                "NVIDIA RTX 3070", "NVIDIA RTX 4050", "NVIDIA RTX 4060", "NVIDIA RTX 4070"
            ]
            default_gpu = 2
            
            os_type_fixed = "Windows"

        # Listeleri dinamik olarak ata
        proc_input = st.selectbox("İşlemci", proc_list, index=default_proc)
        ram_gb = st.number_input("RAM (GB)", 4, 128, 8)
        
    with col2:
        gpu_input = st.selectbox("Ekran Kartı", gpu_list, index=default_gpu)
        storage_gb = st.number_input("Depolama (GB)", 128, 4000, 256)
        warranty = st.selectbox("Garanti (Yıl)", [1, 2, 3])
        battery = st.slider("Pil Ömrü (Saat)", 1.0, 24.0, 8.0)

 
    proc_brand, proc_series, proc_gen, proc_gen_w = extract_processor_info(proc_input)
    proc_perf_w = processor_performance_weight(proc_series)
    gpu_brand, gpu_model, gpu_type = extract_gpu_info(gpu_input)
    
  
    predicted_segment = guess_segment(proc_perf_w, ram_gb, storage_gb, gpu_type, gpu_model)
    
  
    segment_stats = metadata.get(predicted_segment, {})
    brand_prices = segment_stats.get('brand_map', {})
    fallback_price = 800 if predicted_segment == "Düşük" else 1500
    dynamic_brand_avg_price = brand_prices.get(brand, segment_stats.get('default_price', fallback_price))
    
    st.info(f"🔍 Algılanan Segment: **{predicted_segment}**\n\n"
            f"📈 Referans Marka Değeri ({brand}): **{dynamic_brand_avg_price:.0f}$**")

    if st.button("Fiyatı Tahmin Et"):
        model = load_model(predicted_segment)
        if model is None:
            st.error("Model yüklenemedi. Lütfen önce eğitimi çalıştırın.")
            return

        input_data = pd.DataFrame({
            'Brand': [brand],
            'Processor Brand': [proc_brand],
            'GPU Brand': [gpu_brand],
            'RAM_log': [np.log1p(ram_gb)],
            'Storage_log': [np.log1p(storage_gb)],
            'Warranty (years)': [warranty],
            'GPU_Model_log': [np.log1p(gpu_model)],
            'Brand Avg Price': [dynamic_brand_avg_price],
            'Processor_Generation_Weight': [proc_gen_w],
            'Processor Performance Weight': [proc_perf_w],
            'Battery Life (hours)': [battery]
        })
        
        try:
            pred_log = model.predict(input_data)[0]
            price = np.expm1(pred_log)
            
    
            if predicted_segment == "Düşük" and price > 1300: price = 1300
            if brand == "Apple" and price < 900: price = 900 
            
            st.success(f"💰 Tahmini Fiyat: **${price:,.2f}**")
        except Exception as e:
            st.error(f"Hata: {e}")

if __name__ == "__main__":
    main()