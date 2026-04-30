import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Generate sample signal data (simulated)
np.random.seed(0)
signal = np.random.normal(0, 1, 1000)

# Feature extraction
rms = np.sqrt(np.mean(signal**2))
mean = np.mean(signal)
energy = np.sum(signal**2)

print("Extracted Features:")
print(f"RMS: {rms}")
print(f"Mean: {mean}")
print(f"Energy: {energy}")

# Create simple dataset
data = pd.DataFrame({
    'RMS': [rms, rms*1.2],
    'Mean': [mean, mean*1.1],
    'Energy': [energy, energy*1.3],
    'Label': [0, 1]  # 0 = Healthy, 1 = Worn
})

X = data[['RMS', 'Mean', 'Energy']]
y = data['Label']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Prediction
prediction = model.predict(X)

print("Predictions:", prediction)

# Plot signal
plt.plot(signal)
plt.title("Simulated Signal")
plt.show()
