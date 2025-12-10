import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle
import os
import gzip # GZIP compression ke liye
from sklearn.linear_model import LinearRegression # Sirf test karne ke liye

# ===================== SCRIPT CONFIGURATION =====================
DATASET_FILE = "flights.csv" 
MODEL_OUTPUT_FILE = "model_tree_compressed.pkl.gz" # Final compressed file ka naam
TARGET_COL = "Price"

# Aapki final feature list (Total_Stops excluded)
feature_cols = [
    'Airline',
    'Date_of_Journey',
    'Source',
    'Destination',
    'Route',
    'Additional_Info',
    'Duration_Minutes',
    'Dep_Hour',
    'Arr_Hour'
] 

# ===================== 1. LOAD DATA =====================
print(f"Loading data from {DATASET_FILE}...")
if not os.path.exists(DATASET_FILE):
    print(f"\n❌ ERROR: Dataset file '{DATASET_FILE}' not found.")
    exit()

try:
    df = pd.read_csv(DATASET_FILE)
    
    # Date_of_Journey ko Day of Month mein badalna
    try:
        df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], dayfirst=True).dt.day
    except Exception as e:
        print(f"Warning: Date parsing error on Date_of_Journey: {e}. Using original column.")
    
except Exception as e:
    print(f"\n❌ ERROR loading CSV: {e}")
    exit()

# --------------------------------------------------------
print("⚙️ Preparing data for training...")

# Zaroori columns check karein
required_cols = feature_cols + [TARGET_COL]
if not all(col in df.columns for col in required_cols):
    missing_cols = [col for col in required_cols if col not in df.columns]
    print(f"\n❌ CRITICAL ERROR: Following columns are missing from the CSV: {missing_cols}")
    exit()
    
X = df[feature_cols] 
y = df[TARGET_COL]


# ===================== 2. TRAIN TEST SPLIT =====================
print("Splitting data into training (80%) and testing (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# ===================== 3. TRAIN MODEL (Random Forest) =====================
print("\n⚙️ Training Random Forest Regressor (This will be saved)...")
rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1 # Performance ke liye
)

rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
rf_accuracy = r2_score(y_test, pred_rf)
print(f"✅ Training complete. Random Forest R2 Score: {rf_accuracy:.4f}")

# ===================== 4. SAVE MODEL WITH GZIP COMPRESSION =====================
print(f"\n💾 Saving Random Forest model to {MODEL_OUTPUT_FILE} using GZIP compression...")
# Hum sirf 'rf' model ko save kar rahe hain

try:
    # GZIP file ko open karke, usmein seedha 'pickle.dump' se model save kar dein.
    # 
    with gzip.open(MODEL_OUTPUT_FILE, 'wb') as f:
        pickle.dump(rf, f)
    
    print(f"\n🎉 MODEL SAVED SUCCESSFULLY: {MODEL_OUTPUT_FILE}")
    
except Exception as e:
    print(f"❌ ERROR saving model: {e}")