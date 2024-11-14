import pandas as pd
import matplotlib.pyplot as plt

ganancias_valuesDF = pd.read_csv("ganancia_baja3.csv")
print(ganancias_valuesDF.mean(),ganancias_valuesDF.std())



# Select the column '0.035' for the histogram
column_name = '0.020000000000000004'

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(ganancias_valuesDF[column_name].dropna(), bins=20, edgecolor='black')
plt.title(f'Histogram of Column {column_name}')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()