from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('datasets_competencia_02_DE_1y.csv')

# Muestreo estratificado
fraction = 0.1  # Fracción deseada
_, sampled_df = train_test_split(data, test_size=fraction, stratify=data['foto_mes'])

print(sampled_df)
print(sampled_df.shape)
print(sampled_df['foto_mes'].unique())

sampled_df.to_csv('competencia_02_DE_1y_sample.csv', index=False)
