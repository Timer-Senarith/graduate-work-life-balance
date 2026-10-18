# ALY6105 Module 6: Graduate student work-life balance
# All data are simulated. No real students were surveyed.
# AI-assisted draft: review the assumptions and code before submitting.
# Install once in Terminal: python -m pip install numpy pandas matplotlib
# Run this file normally in PyCharm. No command-line flags are needed.
# Running it recreates the CSV with the same data used in the existing report.

# 1. Import the libraries and choose the output folder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Save files next to this Python script, regardless of PyCharm's working folder.
folder = Path(__file__).resolve().parent
CSV = folder / "ALY6105_Module6_Simulated_Data.csv"

# 2. Set up the simulation
# These dictionaries connect each category to its assumed probabilities.
# Each list of probabilities must add up to 1.
HOURS = ["0", "1-19", "20-34", "35+"]
SUPPORT = ["Flexible course scheduling", "Coordinated assignment deadlines",
           "Time management workshops", "Wellbeing support", "No additional support"]
# Invented design inputs, not Northeastern enrollment statistics.
FRAME = {"Full-time": 1200, "Part-time": 800}
SAMPLE = {"Full-time": 120, "Part-time": 80}
HOUR_PROBS = {"Full-time": [.35, .40, .20, .05],
              "Part-time": [.05, .15, .35, .45]}
# Probability of balance scores 1, 2, 3, 4, 5, by hours category.
BALANCE_PROBS = {"0": [.05, .15, .30, .35, .15],
                 "1-19": [.08, .22, .35, .25, .10],
                 "20-34": [.18, .32, .30, .15, .05],
                 "35+": [.30, .35, .22, .10, .03]}
TEXT = {
    SUPPORT[0]: ["More evening class options would help me plan my week.",
                 "I would value more choice in class meeting times."],
    SUPPORT[1]: ["Avoiding several major deadlines in the same week would help.",
                 "Earlier notice of assignment deadlines would help me plan."],
    SUPPORT[2]: ["Practical workshops on planning my study time would help.",
                 "I would like support building a realistic weekly schedule."],
    SUPPORT[3]: ["More accessible wellbeing appointments would be useful.",
                 "I would appreciate clearer information about wellbeing services."],
    SUPPORT[4]: ["My current schedule works well for me.",
                 "I do not need an additional service at the moment."]
}

# 3. Randomly select students and generate their five survey answers
rng = np.random.default_rng(610506)  # Same seed gives the same results.
rows = []
for status, size in FRAME.items():
    if status == "Full-time":
        prefix = "F"
    else:
        prefix = "P"
    frame_ids = [f"SYN-{prefix}-{i:04d}" for i in range(1, size + 1)]
    selected = rng.choice(frame_ids, SAMPLE[status], replace=False)
    for sid in selected:
        hours = str(rng.choice(HOURS, p=HOUR_PROBS[status]))
        balance = int(rng.choice([1, 2, 3, 4, 5], p=BALANCE_PROBS[hours]))
        # Inverse association is an assumption, not a discovery.
        stress = int(np.clip(6 - balance + rng.choice([-1, 0, 1],
                            p=[.20, .60, .20]), 1, 5))
        if balance <= 2:
            probs = [.40, .30, .12, .15, .03]
        else:
            probs = [.27, .25, .18, .15, .15]
        support = str(rng.choice(SUPPORT, p=probs))
        comment = str(rng.choice(TEXT[support]))
        rows.append([sid, status, hours, balance, stress, support, comment])
df = pd.DataFrame(rows, columns=["student_id", "enrollment_status",
    "paid_work_hours", "balance_score", "stress_score",
    "preferred_support", "improvement_comment"])
# Exactly 6 missing stress ratings and 10 blank comments; selected at random.
df.loc[rng.choice(df.index, 6, replace=False), "stress_score"] = np.nan
df.loc[rng.choice(df.index, 10, replace=False), "improvement_comment"] = ""
df["stress_score"] = df["stress_score"].astype("Int64")
df.to_csv(CSV, index=False)

# 4. Load and clean the data
# Read hour categories as text, so "0" remains a category rather than a number.
df = pd.read_csv(CSV, dtype={"paid_work_hours": str})
df = df.drop_duplicates()
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()
df = df.replace("", np.nan)

print("Number of students:", len(df))
print("Missing values:")
print(df.isna().sum())
print("Repeated student IDs:", df["student_id"].duplicated().sum())

# Confirm ratings are valid. Blank stress ratings are allowed.
for column in ["balance_score", "stress_score"]:
    invalid = df[column].dropna().isin([1, 2, 3, 4, 5]) == False
    if invalid.any():
        raise ValueError("Invalid rating in " + column)

# Do not fill missing ratings with zero: zero is not a survey response.
# Pandas skips missing ratings when calculating means, medians and SDs.

# 5. Calculate descriptive statistics and frequencies
statistics = df[["balance_score", "stress_score"]].describe().round(2)
print("Descriptive statistics:")
print(statistics)
statistics.to_csv(folder / "simple_descriptive_statistics.csv")

for column in ["enrollment_status", "paid_work_hours", "balance_score",
               "stress_score", "preferred_support"]:
    print("Frequency table for", column)
    print(df[column].value_counts())
    print("Percent of valid responses:")
    print((df[column].value_counts(normalize=True) * 100).round(1))

# 6. Compare work-hour groups
# A score of 1 or 2 means low balance. True is counted as 1; False as 0.
df["low_balance"] = df["balance_score"] <= 2
df["work_group"] = "Under 20 hours"
df.loc[df["paid_work_hours"].isin(["20-34", "35+"]), "work_group"] = "20+ hours"

print("Overall low balance count:", df["low_balance"].sum())
print("Overall low balance percentage:", round(df["low_balance"].mean() * 100, 1))

# Build a small results table using an ordinary loop.
results = []
for group in ["Under 20 hours", "20+ hours"]:
    students = df[df["work_group"] == group]
    total = len(students)
    low_count = students["low_balance"].sum()
    percentage = low_count / total * 100
    median = students["balance_score"].median()
    results.append([group, total, low_count, percentage, median])

comparison = pd.DataFrame(results, columns=["Work group", "Students",
                          "Low balance count", "Low balance percent", "Median balance"])
print(comparison.round(2).to_string(index=False))
comparison.to_csv(folder / "simple_group_comparison.csv", index=False)

# Check the same comparison separately for full-time and part-time students.
for status in ["Full-time", "Part-time"]:
    students = df[df["enrollment_status"] == status]
    print("Low balance percentage for", status)
    print((students.groupby("work_group")["low_balance"].mean() * 100).round(1))

# Stress percentages use 194 valid ratings, not all 200 students.
stress = df["stress_score"].dropna()
print("High stress count:", (stress >= 4).sum())
print("Valid stress responses:", len(stress))
print("High stress percentage:", round((stress >= 4).mean() * 100, 1))

# 7. Create and save a bar chart
plt.figure(figsize=(7, 4))
plt.bar(comparison["Work group"], comparison["Low balance percent"],
        color=["gray", "steelblue"])
plt.ylim(0, 100)
plt.ylabel("Students reporting low balance (%)")
plt.xlabel("Paid work per week")
plt.title("Simulated graduate student work-life balance")
plt.tight_layout()
plt.savefig(folder / "simple_work_balance.png", dpi=300)
plt.show()

# The association was built into the simulation. It is not evidence about
# actual Northeastern students and does not establish cause and effect.
