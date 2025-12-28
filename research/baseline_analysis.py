import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Veri Yükleme
df = pd.read_csv("LaptopDataset.csv")  # Dosya adı doğru olmalı

# 2. Gereksiz sütun kaldır (Model genellikle benzersiz ve anlamsızdır)
if 'Model' in df.columns:
    df.drop(columns=['Model'], inplace=True)

# 3. İşlemci markasını al (Ör: "Intel Core i7" -> "Intel")
df['Processor Brand'] = df['Processor'].str.split().str[0]

# 4. GPU markasını ve modelini ayıkla
df['Graphics Brand'] = df['Graphics Card'].str.split().str[0]

# GPU model stringini çıkar, yoksa 'Unknown' olarak işaretle
def extract_gpu_model_str(gpu_str):
    if pd.isna(gpu_str):
        return 'Unknown'
    # Önemli modelleri yakala
    match = re.search(r'(GTX \d{3,4}|RTX \d{3,4}|UHD|Radeon)', gpu_str, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # Yukarıdakiler yoksa, ilk kelimeyi döndür
        return gpu_str.split()[0]

df['GPU Model'] = df['Graphics Card'].apply(extract_gpu_model_str)

# 5. Ekran boyutuna göre sınıflandırma
df['Screen Size Class'] = pd.cut(df['Screen Size (inches)'],
                                 bins=[0, 14, 15.6, 18],
                                 labels=['Small', 'Medium', 'Large'])

# 6. Hedef ve özellikleri ayır
X = df.drop(columns=['Price ($)', 'Graphics Card', 'Processor'])
y = df['Price ($)']

# 7. Kategorik ve sayısal sütunlar
categorical_features = ['Brand', 'Processor Brand', 'Graphics Brand', 'Operating System', 'Screen Size Class', 'GPU Model']
numerical_features = ['RAM (GB)', 'Storage (GB)', 'Screen Size (inches)', 'Weight (kg)', 'Battery Life (hours)', 'Warranty (years)']

# 8. Ön işleme
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 9. Model pipeline'ı oluştur
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# 10. Veri setini eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 11. Hiperparametre araması için parametreler
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5],
}

# 12. GridSearchCV ile en iyi parametreyi bul
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# 13. En iyi model ile tahmin yap
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 14. Performans metrikleri
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nGridSearch ile en iyi model performansı:")
print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 4))

# 15. En büyük hatalar
errors = abs(y_pred - y_test)
error_df = X_test.copy()
error_df['Actual'] = y_test
error_df['Predicted'] = y_pred
error_df['Error'] = errors

print("\nEn büyük hataların bazıları:")
print(error_df.sort_values(by='Error', ascending=False).head(10))
