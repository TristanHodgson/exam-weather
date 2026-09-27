import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Variables
TEMP_LOWER_BOUND = 15
TEMP_UPPER_BOUND = 21

SCORE_WEIGHT = 0.05  # Weight for rainfall in the score calculation

# Should be greater than 1925 due to changes in methodology in the data collection after that year
MIN_YEAR = 1925


################################
### Load and preprocess data ###
################################

# Load the CSV file into a DataFrame
df = pd.read_csv("data.csv")

# Select specific columns from the DataFrame
df = df[["YYYY", "MM", "DD", "Tmax °C", "Rainfall mm raw incl traces"]]

# Rename the columns for better readability
df.rename(columns={
    "YYYY": "Year",
    "MM": "Month",
    "DD": "Day",
    "Tmax °C": "Max Temp",
    "Rainfall mm raw incl traces": "Rainfall"}, inplace=True)

# Convert to datetime format
df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
# Drop the original Year, Month, and Day columns
df.drop(columns=["Year", "Month", "Day"], inplace=True)
# Set the Date column as the index
df.set_index("Date", inplace=True)

# Trace to 0 in rainfall column
df["Rainfall"] = df["Rainfall"].replace("Tr", 0)
df["Rainfall"] = df["Rainfall"].replace("TR", 0)
df["Rainfall"] = df["Rainfall"].replace("tr", 0)
df["Rainfall"] = df["Rainfall"].replace("trace", 0)
df["Rainfall"] = df["Rainfall"].replace("-", 0)
df["Rainfall"] = df["Rainfall"].replace(" -", 0)
df["Rainfall"] = df["Rainfall"].astype(float)


# Add a week number column
df["Week Number"] = df.index.isocalendar().week


df = df[df.index.year >= MIN_YEAR]

# Count nan values in the DataFrame
nan_counts = df.isna().sum()
print("NaN counts in each column:")
print(nan_counts)
# Print any rows with NaN values
nan_rows = df[df.isna().any(axis=1)]
print("Rows with NaN values:")
print(nan_rows)

df.dropna(inplace=True)

################################
###        Score data        ###
################################


def temp_score(temp, lower_bound=TEMP_LOWER_BOUND, upper_bound=TEMP_UPPER_BOUND):
    if temp >= lower_bound and temp <= upper_bound:
        return 0
    return min(abs(temp - lower_bound), abs(temp - upper_bound))


# Add a new column for temperature score
df["Temp Score"] = df["Max Temp"].apply(temp_score)
df["score"] = df["Temp Score"] + df["Rainfall"] * SCORE_WEIGHT

################################
###  Temperature vs Rainfall ###
################################

# Plot a scatter graph of temperature score vs rainfall, coloured by week number
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df["Temp Score"],
    df["Rainfall"],
    c=df["Week Number"],
    cmap="viridis",
    alpha=0.7,
    s=10
)
plt.colorbar(scatter, label="Week Number")
plt.xlabel("Temperature Score")
plt.ylabel("Rainfall (mm)")
plt.title("Temperature Score vs Rainfall")
plt.grid()
plt.tight_layout()
plt.savefig("temp_score_vs_rainfall.svg", dpi=1000, bbox_inches="tight", transparent=True)
plt.close()

################################
###    Week number to date   ###
################################

# Find the first date for each week in each year, then find the most common
# month/day for each week number across all years.
year_week_starts = (
    df.groupby([df.index.year, "Week Number"])
    .apply(lambda group: group.index.min(), include_groups=False)
    .reset_index(name="Starting Date")
)
year_week_starts["Starting Date"] = year_week_starts["Starting Date"].dt.strftime(
    "%d-%m")
week_start_dates = (
    year_week_starts.groupby("Week Number")["Starting Date"]
    .agg(lambda dates: dates.mode().iloc[0])
    .reset_index()
)

# print(tabulate(week_start_dates, headers="keys", tablefmt="github", showindex=False))

################################
###   Lowest average score   ###
################################

# Find the week number with the lowest average score
lowest_avg_score_week = df.groupby("Week Number")["score"].mean().idxmin()
lowest_avg_score = df.groupby("Week Number")["score"].mean().min()

print(
    f"Week with lowest average score: {lowest_avg_score_week}, Score: {lowest_avg_score}")

################################
###         Box plots        ###
################################

fig, ax = plt.subplots(figsize=(20, 10))

weeks = sorted(df["Week Number"].unique())

for position, week in enumerate(weeks, start=1):
    # Choose line colour
    if week == 30:
        colour = "red"
    elif week == lowest_avg_score_week:
        colour = "blue"
    else:
        colour = "black"
    week_scores = df.loc[df["Week Number"] == week, "score"]
    ax.boxplot(
        week_scores,
        positions=[position],
        widths=0.6,
        showfliers=False,
        boxprops=dict(color=colour),
        whiskerprops=dict(color=colour),
        capprops=dict(color=colour),
        medianprops=dict(color=colour),
    )

# Set week numbers as x-axis labels
ax.set_xticks(range(1, len(weeks) + 1))
ax.set_xticklabels(weeks, rotation=45)
ax.set_title("Score Distribution by Week Number", fontsize=24)
ax.set_xlabel("Week Number", fontsize=18)
ax.set_ylabel("Score", fontsize=18)
fig.tight_layout()
fig.savefig("boxplot.svg", dpi=1000, bbox_inches="tight", transparent=True)
plt.close(fig)


################################
###  Temp vs rain for best   ###
################################

# Plot of temperature against rainfall for the week with the lowest average score with horizontal lines for the lower and upper bounds of the temperature score, coloured by year
week_data = df[df["Week Number"] == lowest_avg_score_week]
plt.figure(figsize=(20, 10))
scatter = plt.scatter(
    week_data["Max Temp"],
    week_data["Rainfall"],
    c=week_data.index.year,
    cmap="viridis",
)
plt.colorbar(scatter, label="Year")
plt.axvline(x=15, color="red", linestyle="--", label=f"Lower Bound ({TEMP_LOWER_BOUND}°C)")
plt.axvline(x=21, color="green", linestyle="--", label=f"Upper Bound ({TEMP_UPPER_BOUND}°C)")
plt.xlabel("Max Temperature (°C)", fontsize=18)
plt.ylabel("Rainfall (mm)", fontsize=18)
plt.title(f"Max Temperature vs Rainfall for Week {lowest_avg_score_week}", fontsize=24)
plt.legend(fontsize=16)
plt.grid()
plt.tight_layout()
plt.savefig(f"scatter.svg", dpi=1000, bbox_inches="tight", transparent=True)
plt.close()
