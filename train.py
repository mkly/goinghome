import pandas as pd
from xgboost import XGBRegressor

# 1. Load the clean data pickle we made in the ETL step
df = pd.read_pickle("mlb_training_data_clean.pkl")

# 2. Separate Features (X) and Target (y)
X = df.drop(columns=["final_game_minutes", "attendance", "temp"])
y = df["final_game_minutes"]

# 3. Train the Model
print("Training model...")
model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)
model.fit(X, y)

# 4. SAVE THE MODEL IN NATIVE JSON FORMAT
model.save_model("xgb_live_model.json")

print("SUCCESS: Model trained and saved as 'xgb_live_model.json'!")

