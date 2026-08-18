import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df1 = pd.read_csv("Unemployment in India.csv")
df2 = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

print("\n========== DATASET INFORMATION ==========")

print("Dataset 1 shape:", df1.shape)
print("Dataset 2 shape:", df2.shape)

print("\nFirst 5 rows of Dataset 1:")
print(df1.head())

print("\nFirst 5 rows of Dataset 2:")
print(df2.head())


df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

print("\nDataset 1 columns:")
print(df1.columns.tolist())

print("\nDataset 2 columns:")
print(df2.columns.tolist())


print("\n========== MISSING VALUES ==========")

print("\nDataset 1:")
print(df1.isnull().sum())

print("\nDataset 2:")
print(df2.isnull().sum())


df1 = df1.dropna()
df2 = df2.dropna()

print("\nDataset 1 shape after cleaning:", df1.shape)
print("Dataset 2 shape after cleaning:", df2.shape)


df1["Date"] = pd.to_datetime(df1["Date"], dayfirst=True)
df2["Date"] = pd.to_datetime(df2["Date"], dayfirst=True)

print("\nDate range in Dataset 1:")
print(df1["Date"].min(), "to", df1["Date"].max())

print("\nDate range in Dataset 2:")
print(df2["Date"].min(), "to", df2["Date"].max())


print("\n========== BASIC STATISTICS ==========")

print(df1.describe())


average_unemployment = df1[
    "Estimated Unemployment Rate (%)"
].mean()

print(
    "\nAverage Unemployment Rate:",
    round(average_unemployment, 2),
    "%"
)


monthly_unemployment = (
    df1.groupby("Date")["Estimated Unemployment Rate (%)"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_unemployment["Date"],
    monthly_unemployment["Estimated Unemployment Rate (%)"],
    marker="o"
)

plt.title("Overall Unemployment Rate Trend in India")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()


state_unemployment = (
    df1.groupby("Region")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values(ascending=False)
)

print("\n========== STATE-WISE UNEMPLOYMENT ==========")
print(state_unemployment)


plt.figure(figsize=(12, 7))

state_unemployment.plot(kind="bar")

plt.title("Average Unemployment Rate by State")
plt.xlabel("State")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


highest_state = state_unemployment.idxmax()
highest_rate = state_unemployment.max()

lowest_state = state_unemployment.idxmin()
lowest_rate = state_unemployment.min()

print("\n========== STATE COMPARISON ==========")

print(
    "Highest unemployment state:",
    highest_state,
    "-",
    round(highest_rate, 2),
    "%"
)

print(
    "Lowest unemployment state:",
    lowest_state,
    "-",
    round(lowest_rate, 2),
    "%"
)


area_unemployment = (
    df1.groupby("Area")["Estimated Unemployment Rate (%)"]
    .mean()
)

print("\n========== RURAL VS URBAN ==========")
print(area_unemployment)


plt.figure(figsize=(7, 5))

area_unemployment.plot(kind="bar")

plt.title("Average Unemployment Rate: Rural vs Urban")
plt.xlabel("Area")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


df1["Year"] = df1["Date"].dt.year

yearly_unemployment = (
    df1.groupby("Year")["Estimated Unemployment Rate (%)"]
    .mean()
)

print("\n========== YEARLY UNEMPLOYMENT ==========")
print(yearly_unemployment)


plt.figure(figsize=(8, 5))

yearly_unemployment.plot(kind="bar")

plt.title("Average Unemployment Rate by Year")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


pre_covid = df1[
    df1["Date"] < "2020-03-01"
]["Estimated Unemployment Rate (%)"].mean()

covid_period = df1[
    df1["Date"] >= "2020-03-01"
]["Estimated Unemployment Rate (%)"].mean()

increase = covid_period - pre_covid

print("\n========== COVID-19 IMPACT ==========")

print(
    "Average unemployment before COVID:",
    round(pre_covid, 2),
    "%"
)

print(
    "Average unemployment during COVID:",
    round(covid_period, 2),
    "%"
)

print(
    "Increase in unemployment:",
    round(increase, 2),
    "percentage points"
)


covid_monthly = (
    df2.groupby("Date")["Estimated Unemployment Rate (%)"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))

plt.plot(
    covid_monthly["Date"],
    covid_monthly["Estimated Unemployment Rate (%)"],
    marker="o"
)

plt.axvline(
    pd.Timestamp("2020-03-01"),
    linestyle="--",
    label="COVID-19 Period"
)

plt.title("Unemployment Trend During COVID-19")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.grid(True)

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


peak_row = covid_monthly.loc[
    covid_monthly[
        "Estimated Unemployment Rate (%)"
    ].idxmax()
]

print("\n========== PEAK UNEMPLOYMENT ==========")

print(
    "Peak unemployment date:",
    peak_row["Date"].strftime("%B %Y")
)

print(
    "Peak unemployment rate:",
    round(
        peak_row["Estimated Unemployment Rate (%)"],
        2
    ),
    "%"
)


labour_participation = (
    df1.groupby("Date")
    ["Estimated Labour Participation Rate (%)"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))

plt.plot(
    labour_participation["Date"],
    labour_participation[
        "Estimated Labour Participation Rate (%)"
    ],
    marker="o"
)

plt.title("Labour Participation Rate Trend")
plt.xlabel("Date")
plt.ylabel("Labour Participation Rate (%)")

plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()


employment_trend = (
    df1.groupby("Date")["Estimated Employed"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))

plt.plot(
    employment_trend["Date"],
    employment_trend["Estimated Employed"],
    marker="o"
)

plt.title("Employment Trend in India")
plt.xlabel("Date")
plt.ylabel("Estimated Employed")

plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()


numeric_columns = [
    "Estimated Unemployment Rate (%)",
    "Estimated Employed",
    "Estimated Labour Participation Rate (%)"
]

correlation = df1[numeric_columns].corr()

print("\n========== CORRELATION ==========")
print(correlation)


plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Employment Indicators")

plt.tight_layout()
plt.show()


df1["Month"] = df1["Date"].dt.month
df1["Month_Name"] = df1["Date"].dt.month_name()

monthly_average = (
    df1.groupby(
        ["Month", "Month_Name"]
    )["Estimated Unemployment Rate (%)"]
    .mean()
    .reset_index()
    .sort_values("Month")
)

print("\n========== MONTHLY AVERAGE ==========")
print(monthly_average)


plt.figure(figsize=(12, 6))

plt.plot(
    monthly_average["Month_Name"],
    monthly_average["Estimated Unemployment Rate (%)"],
    marker="o"
)

plt.title("Monthly Seasonal Unemployment Trend")
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")

plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()


highest_month = monthly_average.loc[
    monthly_average[
        "Estimated Unemployment Rate (%)"
    ].idxmax()
]

lowest_month = monthly_average.loc[
    monthly_average[
        "Estimated Unemployment Rate (%)"
    ].idxmin()
]

print("\n========== MONTHLY COMPARISON ==========")

print(
    "Highest unemployment month:",
    highest_month["Month_Name"],
    "-",
    round(
        highest_month[
            "Estimated Unemployment Rate (%)"
        ],
        2
    ),
    "%"
)

print(
    "Lowest unemployment month:",
    lowest_month["Month_Name"],
    "-",
    round(
        lowest_month[
            "Estimated Unemployment Rate (%)"
        ],
        2
    ),
    "%"
)


top_10 = state_unemployment.head(10)

plt.figure(figsize=(12, 6))

top_10.plot(kind="bar")

plt.title("Top 10 States with Highest Average Unemployment")
plt.xlabel("State")
plt.ylabel("Unemployment Rate (%)")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


print("\n")
print("=" * 60)
print("             FINAL ANALYSIS SUMMARY")
print("=" * 60)

print(
    "\n1. Average unemployment rate:",
    round(average_unemployment, 2),
    "%"
)

print(
    "\n2. Highest unemployment state:",
    highest_state,
    "-",
    round(highest_rate, 2),
    "%"
)

print(
    "\n3. Lowest unemployment state:",
    lowest_state,
    "-",
    round(lowest_rate, 2),
    "%"
)

print(
    "\n4. Pre-COVID unemployment:",
    round(pre_covid, 2),
    "%"
)

print(
    "\n5. COVID-period unemployment:",
    round(covid_period, 2),
    "%"
)

print(
    "\n6. Increase during COVID:",
    round(increase, 2),
    "percentage points"
)

print(
    "\n7. Peak unemployment occurred in:",
    peak_row["Date"].strftime("%B %Y")
)

print(
    "\n8. Peak unemployment rate:",
    round(
        peak_row["Estimated Unemployment Rate (%)"],
        2
    ),
    "%"
)

print("\n========== POLICY INSIGHTS ==========")

print("""
Employment-generation programs can be targeted toward
states with consistently high unemployment.

Skill-development and vocational training can improve
employability.

Small businesses and industries can be supported during
economic crises to protect jobs.

Rural and urban employment programs can be designed
according to regional unemployment patterns.

During future economic disruptions, temporary employment
support and social protection programs can reduce the
impact on workers.

Regular monitoring of unemployment data can help
policymakers respond quickly to sudden increases.
""")

print("=" * 60)
print("              ANALYSIS COMPLETED")
print("=" * 60)