import pandas as pd

# Loading the dataset from the CSV file
df = pd.read_csv("Players.csv")

# Filtering the dataset to identify and separating Defenders (DF) and Midfielders (MF)
filtered_data = df[df["Pos"].isin(["DF", "MF"])]


# Extracting the yellow card arrays for each group
yellow_cards_df = filtered_data[filtered_data["Pos"] == "DF"]["CrdY"]
yellow_cards_mid = filtered_data[filtered_data["Pos"] == "MF"]["CrdY"]
# Confirming the sample size
print(len(yellow_cards_df), len(yellow_cards_mid))



# Computing the descriptive statistics for both groups
def_description = yellow_cards_df.describe()
mid_description = yellow_cards_mid.describe()

# Displaying the summary side by side
description_summary = pd.DataFrame({"Defenders (DF)": def_description, "Midfielders (MF)": mid_description})

print(description_summary.round(3))

import scipy.stats as stats

# Calculate 95% Confidence Interval for Defenders
ci_def = stats.t.interval(
    0.95,
    df=len(yellow_cards_df) - 1,
    loc=yellow_cards_df.mean(),
    scale=stats.sem(yellow_cards_df),
)

# Calculate 95% Confidence Interval for Midfielders
ci_mid = stats.t.interval(
    0.95,
    df=len(yellow_cards_mid) - 1,
    loc=yellow_cards_mid.mean(),
    scale=stats.sem(yellow_cards_mid),
)

print("--- 95% CONFIDENCE INTERVALS ---")
print(f"Defenders 95% CI:   ({ci_def[0]:.3f}, {ci_def[1]:.3f})")
print(f"Midfielders 95% CI: ({ci_mid[0]:.3f}, {ci_mid[1]:.3f})")

# Conduct Welch's Two-Sample t-Test
t_stat, p_val = stats.ttest_ind(
    yellow_cards_df, yellow_cards_mid, equal_var=False
)

# Significance level
alpha = 0.05

print("--- TWO-SAMPLE T-TEST RESULTS ---")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value:     {p_val:.4f}")

# Decision logic
if p_val < alpha:
    print(
        "\nDecision: Reject H0 (There IS a statistically significant difference)."
    )
else:
    print(
        "\nDecision: There is insufficient evidence of a statistically significant difference."
    )