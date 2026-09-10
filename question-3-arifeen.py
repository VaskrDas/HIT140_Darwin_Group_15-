import pandas as pd
import math
import scipy.stats as stats

# load the data
df = pd.read_csv("question-3-dataset-arifeen.csv")

# get only defenders and midfielders
filtered_data = df[(df["Pos"] == "DF") | (df["Pos"] == "MF")]

# get yellow cards for each position
yellow_cards_df = filtered_data[filtered_data["Pos"] == "DF"]["CrdY"]
yellow_cards_mid = filtered_data[filtered_data["Pos"] == "MF"]["CrdY"]

print("Sample sizes:")
print("Defenders:", len(yellow_cards_df))
print("Midfielders:", len(yellow_cards_mid))

# basic stats
print("\nDefenders stats:")
print(yellow_cards_df.describe().round(3))

print("\nMidfielders stats:")
print(yellow_cards_mid.describe().round(3))

# get mean, std, and n for defenders
def_mean = stats.tmean(yellow_cards_df)
def_std = stats.tstd(yellow_cards_df)
def_n = len(yellow_cards_df)

# get mean, std, and n for midfielders
mid_mean = stats.tmean(yellow_cards_mid)
mid_std = stats.tstd(yellow_cards_mid)
mid_n = len(yellow_cards_mid)

# z value for 95% confidence
z_star = stats.norm.ppf(0.975)

# standard error
def_se = def_std / math.sqrt(def_n)
mid_se = mid_std / math.sqrt(mid_n)

# confidence interval math
def_lower = round(def_mean - z_star * def_se, 3)
def_upper = round(def_mean + z_star * def_se, 3)

mid_lower = round(mid_mean - z_star * mid_se, 3)
mid_upper = round(mid_mean + z_star * mid_se, 3)

print("\n95% Confidence Intervals:")
print("Defenders: (", def_lower, ",", def_upper, ")")
print("Midfielders: (", mid_lower, ",", mid_upper, ")")

# t test
t_stat, p_val = stats.ttest_ind(yellow_cards_df, yellow_cards_mid, equal_var=False)

t_stat = round(t_stat, 3)
p_val = round(p_val, 3)

alpha = 0.05

print("\nT-test results:")
print("t-statistic:", t_stat)
print("p-value:", p_val)

# check if significant
if p_val < alpha:
    print("\nSince p-value is less than alpha, we reject H0. There is a significant difference.")
else:
    print("\nSince p-value is greater than alpha, we do not reject H0. Not enough evidence of a difference.")