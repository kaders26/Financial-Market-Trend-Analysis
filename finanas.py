import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Sahte finansal veri seti oluşturun
np.random.seed(42)
date_rng = pd.date_range(start='2022-01-01', end='2022-12-31', freq='B')
price_data = np.random.normal(loc=0.1, scale=1, size=(len(date_rng)))
df = pd.DataFrame(data={'Date': date_rng, 'Price': np.cumsum(price_data) + 100})

# Zaman serisini görselleştirin
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Price'], label='Hisse Senedi Fiyatları')
plt.title('Hisse Senedi Fiyatları Zaman Serisi')
plt.xlabel('Tarih')
plt.ylabel('Fiyat')
plt.legend()
plt.show()

# ARIMA modelini kullanarak zaman serisi analizi
model = sm.tsa.ARIMA(df['Price'], order=(1, 1, 1))
results = model.fit()

# Tahminleri görselleştirin
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Price'], label='Gerçek Fiyatlar')
plt.plot(df['Date'], results.fittedvalues, label='Tahmin Edilen Fiyatlar', color='red')
plt.title('ARIMA Modeli ile Hisse Senedi Fiyatları Tahmini')
plt.xlabel('Tarih')
plt.ylabel('Fiyat')
plt.legend()
plt.show()

# Modelin istatistiksel özetini görüntüleyin
print(results.summary())
