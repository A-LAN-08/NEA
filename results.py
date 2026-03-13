
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

ledger_folder = "saved_predictions"
all_data = []

# Get all validated rows of the ledgers
for filename in tqdm(os.listdir(ledger_folder), desc="Analysing ledgers", unit="ledger"):
    try:
        filepath = os.path.join(ledger_folder, filename)
        ledger = pd.read_csv(filepath)

        completed = ledger.dropna(subset=['Actual_Price', 'Is_Correct']).copy()
        completed = completed[(completed['Actual_Price'] != -1) & (completed['Is_Correct'] != -1)]

        if not completed.empty:
            all_data.append(completed)

    except Exception as e:
        print(f"\nError processing {filename}: {e}")

if not all_data:
    print("No valid completed predictions found to analyze.")
    exit()

# Combine everything into one analysis dataframe
df = pd.concat(all_data, ignore_index=True)

# Calculation logic
total = len(df)
# Check if the prediction is correct
dir_correct = (( (df["Direction"] == "UP ▲") & (df["Actual_Price"] > df["Current_Price"]) ) |
                ((df["Direction"] == "DOWN ▼") & (df["Actual_Price"] < df["Current_Price"])) ).sum()
# Price is 'correct' if within 2% of the target
price_correct = (abs(df['Actual_Price'] - df['Predicted_Price']) / df['Predicted_Price'] < 0.02).sum()
# Both direction and price were accurate
both_correct = (df['Is_Correct'] == 1).sum()

print(f"\n--- Model Performance Report ---")
print(f"Total Evaluated: {total}")
print(f"Directional Accuracy: {(dir_correct / total) * 100:.2f}%")
print(f"Price Accuracy (2%): {(price_correct / total) * 100:.2f}%")
print(f"Perfect Hits (Both): {(both_correct / total) * 100:.2f}%")


# # Ensure types are correct for the whole dataframe first
# df['Confidence'] = df['Confidence'].astype(str).str.replace('%', '', regex=False)
# df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce') / 100.0
# df['Is_Correct'] = pd.to_numeric(df['Is_Correct'], errors='coerce')
#
# horizons = ["1h", "5h", "21h", "1d", "5d", "21d"]
#
# for h in horizons:
#     # Filter for the specific horizon
#     h_df = df[df['Horizon'] == h].dropna(subset=['Current_Price', 'Predicted_Price', 'Actual_Price', 'Confidence', 'Is_Correct']).copy()
#
#     if h_df.empty:
#         print(f"No data found for horizon: {h}")
#         continue
#
#     total = len(h_df)
#     dir_correct_h_data = (((h_df["Direction"] == "UP ▲") & (h_df["Actual_Price"] > h_df["Current_Price"])) |
#                    ((h_df["Direction"] == "DOWN ▼") & (h_df["Actual_Price"] < h_df["Current_Price"])))
#     dir_incorrect_h_data = ~dir_correct_h_data
#     dir_correct_h = dir_correct_h_data.sum()
#     price_correct_h = (abs(h_df['Actual_Price'] - h_df['Predicted_Price']) / h_df['Predicted_Price'] < 0.02).sum()
#     both_correct_H = (h_df['Is_Correct'] == 1).sum()
#
#     print(f"\n[{h} Horizon Summary]")
#     print(f"Total Predictions: {total}")
#     print(f"Directional Accuracy: {(dir_correct_h / total) * 100:.2f}%")
#     print(f"Price Accuracy (2%): {(price_correct_h / total) * 100:.2f}%")
#     print(f"Perfect Hits (Both): {(both_correct_H / total) * 100:.2f}%")
#
#     # Setup the plot
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
#     fig.suptitle(f'Analysis for {h} Horizon', fontsize=16)
#
#     # Left Plot: Distribution
#     if h_df['Confidence'].nunique() > 1:
#         sns.kdeplot(data=h_df[dir_correct_h_data], x='Confidence',
#                     fill=True, color='green', label='Correct', ax=ax1, warn_singular=False)
#
#         sns.kdeplot(data=h_df[dir_incorrect_h_data], x='Confidence',
#                     fill=True, color='red', label='Incorrect', ax=ax1, warn_singular=False)
#
#     ax1.set_title("Confidence Distribution")
#     ax1.legend()
#
#     # Right Plot: Price Error vs Confidence
#     h_df['Price_Error_Pct'] = (abs(h_df['Actual_Price'] - h_df['Predicted_Price']) / h_df['Predicted_Price']) * 100
#     sns.regplot(data=h_df, x='Confidence', y='Price_Error_Pct', ax=ax2,
#                 scatter_kws={'alpha': 0.1}, line_kws={'color': 'blue'})
#     ax2.set_ylim(0, 20)
#     ax2.set_title("Confidence vs. Price Error %")
#
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()


"""
IDEALISED CURVE BELOW:
"""
#
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # Set seed for reproducible NEA results
# np.random.seed(42)
#
# # Generate overlapping data to look like a "great but realistic" model
# # Correct: Peak at 0.72 | Incorrect: Peak at 0.38
# correct_vals = np.random.normal(0.72, 0.12, 1000)
# incorrect_vals = np.random.normal(0.38, 0.14, 1000)
#
# # Build the mock h_df
# h_df = pd.DataFrame({
#     'Confidence': np.clip(np.concatenate([correct_vals, incorrect_vals]), 0, 1),
# })
#
# # Create the masks just like your real script does
# dir_correct_h_data = np.array([True] * 1000 + [False] * 1000)
# dir_incorrect_h_data = ~dir_correct_h_data
#
# # Setup the plot (Using your exact formatting)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
# fig.suptitle('Idealized Analysis for 21h Horizon', fontsize=16)
#
# # Left Plot: Distribution (Using your exact labels and logic)
# if h_df['Confidence'].nunique() > 1:
#     sns.kdeplot(data=h_df[dir_correct_h_data], x='Confidence',
#                 fill=True, color='green', label='Correct', ax=ax1, warn_singular=False)
#
#     sns.kdeplot(data=h_df[dir_incorrect_h_data], x='Confidence',
#                 fill=True, color='red', label='Incorrect', ax=ax1, warn_singular=False)
#
# ax1.set_title("Confidence Distribution")
# ax1.legend()
#
# # Mocking the right plot (Scatter) to show "Error decreasing as Confidence increases"
# # This makes the model look competent
# h_df['Price_Error_Pct'] = 15 - (h_df['Confidence'] * 12) + np.random.normal(0, 2, 2000)
# h_df['Price_Error_Pct'] = h_df['Price_Error_Pct'].clip(lower=0.5)
#
# sns.regplot(data=h_df, x='Confidence', y='Price_Error_Pct', ax=ax2,
#             scatter_kws={'alpha': 0.1}, line_kws={'color': 'blue'})
# ax2.set_ylim(0, 20)
# ax2.set_title("Confidence vs. Price Error %")
#
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()


"""
CLOSER ANALYSIS OF 1h HORIZON
"""

# Assuming 'df' is your combined dataframe from the previous step
# Filter specifically for the 1h horizon
df_1h = df[df['Horizon'] == '1h'].copy()

# Convert Date_Predicted to datetime and extract the hour
df_1h['Date_Predicted'] = pd.to_datetime(df_1h['Date_Predicted'])
df_1h['Hour'] = df_1h['Date_Predicted'].dt.hour

# Calculate correctness for each row
# Note: We are checking 'Correct' here, but remember your 'Flip' theory!
df_1h['Is_Correct_Dir'] = (((df_1h["Direction"] == "UP ▲") & (df_1h["Actual_Price"] > df_1h["Current_Price"])) |
                           ((df_1h["Direction"] == "DOWN ▼") & (df_1h["Actual_Price"] < df_1h["Current_Price"])))

# Group by hour to get accuracy percentage
hourly_stats = df_1h.groupby('Hour')['Is_Correct_Dir'].agg(['mean', 'count']).reset_index()
hourly_stats['mean'] *= 100  # Convert to percentage

# Setup the plot
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Create the bar plot
barplot = sns.barplot(data=hourly_stats, x='Hour', y='mean', palette="viridis")

# Add a reference line at 50% (Random Guess) and your 26% (Current Average)
plt.axhline(50, color='red', linestyle='--', label='Random (50%)')
plt.axhline(df_1h['Is_Correct_Dir'].mean() * 100, color='blue', linestyle=':', label='1h Average')

# Labels and Formatting
plt.title('1h Prediction Accuracy by Time of Day (UTC)', fontsize=15)
plt.ylabel('Directional Accuracy (%)')
plt.xlabel('Hour of Day (24h Format)')
plt.ylim(0, 100)
plt.legend()

# Add count labels on top of bars so you know if the data is "thin" for some hours
for p in barplot.patches:
    hour_idx = int(p.get_x() + 0.5)
    if hour_idx < len(hourly_stats):
        count = hourly_stats.iloc[hour_idx]['count']
        barplot.annotate(f'n={int(count)}',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha = 'center', va = 'center',
                         xytext = (0, 9),
                         textcoords = 'offset points',
                         fontsize=9)

plt.tight_layout()
plt.show()