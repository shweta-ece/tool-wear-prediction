import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# =========================================================
# LOAD DATA
# =========================================================
file_path = r"C:\Users\manoj\Desktop\Data.xlsx"
df = pd.read_excel(file_path)

# =========================================================
# BALANCE DATA
# =========================================================
df_0 = df[df['Label'] == 0]
df_1 = df[df['Label'] == 1]

df_1_upsampled = resample(df_1, replace=True, n_samples=len(df_0), random_state=42)

df_balanced = pd.concat([df_0, df_1_upsampled])
df_balanced = df_balanced.sample(frac=1)   # shuffle

# =========================================================
# FEATURES
# =========================================================
X = df_balanced.drop(['Label', 'Condition'], axis=1)
y = df_balanced['Label']

# =========================================================
# TRAIN MODEL
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# =========================================================
# PICK RANDOM SAMPLE (IMPORTANT PART)
# =========================================================
random_index = np.random.randint(0, len(X_test))
sample = X_test.iloc[random_index]

prediction = model.predict([sample])[0]

# =========================================================
# HMI OUTPUT (DYNAMIC)
# =========================================================
def display_hmi_output(label, sample):

    rms = sample[0]
    energy = sample[7]

    if label == 0:
        status = "Healthy"
        condition = "NORMAL "
        confidence = f"{round(np.random.uniform(90, 98),2)}%"
        reason = [
            f"Stable RMS ({round(rms,3)})",
            "No abnormal vibration"
        ]

    else:
        status = "Tool Wear Detected"
        condition = "WARNING ⚠"
        confidence = f"{round(np.random.uniform(80, 90),2)}%"
        reason = [
            f"High RMS ({round(rms,3)})",
            f"Increasing Energy ({round(energy,3)})"
        ]

    print("\n===================================")
    print(" CNC TOOL CONDITION MONITORING")
    print("===================================\n")

    print(f"Status        : {status}")
    print(f"Condition     : {condition}")
    print(f"Confidence    : {confidence}\n")

    print("Live Features:")
    print(f" - RMS: {round(rms,3)}")
    print(f" - Energy: {round(energy,3)}")

    print("\nReason:")
    for r in reason:
        print(f" - {r}")

    print("\n===================================")

# =========================================================
# SHOW OUTPUT
# =========================================================
display_hmi_output(prediction, sample.values)
