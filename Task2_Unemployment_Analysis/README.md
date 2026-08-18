Task 2: Unemployment Analysis with Python

This project performs Unemployment Analysis using Python. The datasets contain unemployment-related information such as unemployment rate, estimated employed people, labour participation rate, region, and area.

The project includes data loading, data cleaning, data exploration, statistical analysis, visualization, state-wise comparison, rural vs urban analysis, COVID-19 impact analysis, employment trends, seasonal analysis, correlation analysis, and generating policy-related insights.

🛠️ Libraries Used
Library	Purpose
Pandas	Used for loading, cleaning, transforming, and analyzing the unemployment datasets
NumPy	Used for numerical operations and data analysis
Matplotlib	Used for creating and displaying charts and graphs
Seaborn	Used for creating the correlation heatmap and statistical visualizations
Dataset

The project uses two datasets stored as:

Unemployment in India.csv
Unemployment_Rate_upto_11_2020.csv

The datasets contain information related to:

Date
Region
Area
Estimated Unemployment Rate (%)
Estimated Employed
Estimated Labour Participation Rate (%)
🔍 Code Explanation
1. Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Imports the required libraries for data handling, numerical analysis, visualization, and statistical analysis.

2. Loading the Datasets
df1 = pd.read_csv("Unemployment in India.csv")
df2 = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

Loads both CSV datasets into Pandas DataFrames.

3. Exploring the Dataset
print(df1.shape)
print(df2.shape)
print(df1.head())
print(df2.head())

These commands are used to:

Check the number of rows and columns.
View the first five records.
Understand the structure of the datasets.
4. Cleaning Column Names
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

Removes unnecessary spaces from column names to make them easier to use during analysis.

5. Checking Missing Values
print(df1.isnull().sum())
print(df2.isnull().sum())

Checks whether the datasets contain missing values in any column.

6. Removing Missing Values
df1 = df1.dropna()
df2 = df2.dropna()

Removes rows containing missing values so that the analysis can be performed using complete records.

7. Converting Date Column
df1["Date"] = pd.to_datetime(df1["Date"], dayfirst=True)
df2["Date"] = pd.to_datetime(df2["Date"], dayfirst=True)

Converts the Date column into datetime format for time-based analysis.

8. Statistical Analysis
print(df1.describe())

Provides statistical information such as:

Mean
Standard deviation
Minimum value
Maximum value
Quartiles

for the numerical columns.

9. Calculating Average Unemployment Rate
average_unemployment = df1[
    "Estimated Unemployment Rate (%)"
].mean()

Calculates the overall average unemployment rate in the dataset.

10. Overall Unemployment Trend
monthly_unemployment = (
    df1.groupby("Date")["Estimated Unemployment Rate (%)"]
    .mean()
    .reset_index()
)

Groups unemployment data by date and calculates the average unemployment rate for each date.

A line graph is then created to show the unemployment trend over time.

11. State-wise Unemployment Analysis
state_unemployment = (
    df1.groupby("Region")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values(ascending=False)
)

Calculates the average unemployment rate for each state or region.

A bar chart is used to compare unemployment levels between states.

12. Highest and Lowest Unemployment States
highest_state = state_unemployment.idxmax()
lowest_state = state_unemployment.idxmin()

Identifies the state with the highest and lowest average unemployment rates.

13. Rural vs Urban Analysis
area_unemployment = (
    df1.groupby("Area")["Estimated Unemployment Rate (%)"]
    .mean()
)

Groups the data based on the Area column and compares unemployment rates between rural and urban areas.

The results are displayed using a bar chart.

14. Yearly Unemployment Analysis
df1["Year"] = df1["Date"].dt.year

Extracts the year from the date.

The average unemployment rate for each year is then calculated and visualized using a bar chart.

15. COVID-19 Impact Analysis
pre_covid = df1[
    df1["Date"] < "2020-03-01"
]["Estimated Unemployment Rate (%)"].mean()


covid_period = df1[
    df1["Date"] >= "2020-03-01"
]["Estimated Unemployment Rate (%)"].mean()

Compares unemployment before and during the COVID-19 period.

The analysis calculates:

Average unemployment before COVID-19
Average unemployment during COVID-19
Increase in unemployment
16. COVID-19 Monthly Trend
covid_monthly = (
    df2.groupby("Date")["Estimated Unemployment Rate (%)"]
    .mean()
    .reset_index()
)

Analyzes monthly unemployment rates during the COVID-19 period.

A line graph is created to visualize the changes in unemployment.

17. Peak Unemployment
peak_row = covid_monthly.loc[
    covid_monthly[
        "Estimated Unemployment Rate (%)"
    ].idxmax()
]

Identifies the month with the highest unemployment rate in the analyzed COVID-19 data.

18. Labour Participation Rate
labour_participation = (
    df1.groupby("Date")
    ["Estimated Labour Participation Rate (%)"]
    .mean()
    .reset_index()
)

Analyzes the labour participation rate over time and displays it using a line graph.

19. Employment Trend
employment_trend = (
    df1.groupby("Date")["Estimated Employed"]
    .mean()
    .reset_index()
)

Analyzes changes in the estimated number of employed people over time.

A line graph is used to visualize the employment trend.

20. Correlation Analysis
correlation = df1[numeric_columns].corr()

Calculates the correlation between:

Estimated Unemployment Rate
Estimated Employed
Estimated Labour Participation Rate

This helps understand the relationships between different employment indicators.

21. Correlation Heatmap
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

Creates a heatmap to visually represent the correlation between employment-related variables.

22. Seasonal Analysis
df1["Month"] = df1["Date"].dt.month
df1["Month_Name"] = df1["Date"].dt.month_name()

Extracts the month from the date and calculates the average unemployment rate for each month.

This helps identify possible monthly or seasonal patterns.

23. Highest and Lowest Unemployment Months
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

Identifies the months with the highest and lowest average unemployment rates.

24. Top 10 States
top_10 = state_unemployment.head(10)

Selects the 10 states with the highest average unemployment rates and displays them using a bar chart.

25. Final Analysis Summary

The program displays a summary containing:

Average unemployment rate
Highest unemployment state
Lowest unemployment state
Pre-COVID unemployment rate
COVID-period unemployment rate
Increase in unemployment during COVID-19
Peak unemployment period
Peak unemployment rate
26. Policy Insights

The analysis provides insights that can help support economic and social policies, including:

Employment-generation programs for regions with high unemployment.
Skill-development and vocational training programs.
Support for small businesses during economic crises.
Separate employment strategies for rural and urban areas.
Temporary employment and social-support programs during economic disruptions.
Regular monitoring of unemployment data for faster policy responses.
📊 Output

The project generates:

Overall unemployment trend graph
State-wise unemployment graph
Rural vs Urban comparison graph
Yearly unemployment graph
COVID-19 unemployment trend graph
Labour participation trend graph
Employment trend graph
Correlation heatmap
Seasonal unemployment trend graph
Top 10 states unemployment graph

The terminal displays:

Dataset information
Missing values
Statistical summary
Average unemployment rate
State-wise unemployment analysis
Rural vs Urban comparison
COVID-19 impact
Peak unemployment period
Seasonal analysis
Correlation values
Final policy insights
