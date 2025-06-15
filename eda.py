# @title Pizza Sales Analysis
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path='/content/drive/MyDrive/Dataset/Enhanced_pizza_sell_data_2024-25.xlsx'
df=pd.read_excel(path)
df.head()

# @title Inspecting Analysis
# Inspect dataset structure
def column_summary(df):
    summary_data = []

    for col_name in df.columns:
        col_dtype = df[col_name].dtype
        num_of_nulls = df[col_name].isnull().sum()
        num_of_non_nulls = df[col_name].notnull().sum()
        num_of_distinct_values = df[col_name].nunique()

        if num_of_distinct_values <= 10:
            distinct_values_counts = df[col_name].value_counts().to_dict()
        else:
            top_10_values_counts = df[col_name].value_counts().head(10).to_dict()
            distinct_values_counts = {k: v for k, v in sorted(top_10_values_counts.items(), key=lambda item: item[1], reverse=True)}

        summary_data.append({
            'col_name': col_name,
            'col_dtype': col_dtype,
            'num_of_nulls': num_of_nulls,
            'num_of_non_nulls': num_of_non_nulls,
            'num_of_distinct_values': num_of_distinct_values,
            'distinct_values_counts': distinct_values_counts
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df

summary_df = column_summary(df)
display(summary_df)

# @title Data cleaning and preprocessing
# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()
print("\nShape after removing duplicates:", df.shape)

# Convert Order Time and Delivery Time to datetime
df['Order Time'] = pd.to_datetime(df['Order Time'])
df['Delivery Time'] = pd.to_datetime(df['Delivery Time'])
df.head()

# Select numeric columns (e.g., int64, float64)
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
print("Numeric Columns:", numeric_columns)

# Select categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_columns)

# Verify categorical variables
for col in categorical_columns:
    print(f"\n🔹 {col} — Unique values:")
    print(df[col].unique())

# Check for outliers using IQR for numerical columns
for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]
    print(f"\nOutliers in {col}:", outliers.shape[0])

# Validate Delay (min) calculation
df['Calculated Delay'] = df['Delivery Duration (min)'] - df['Estimated Duration (min)']
delay_mismatch = df[abs(df['Delay (min)'] - df['Calculated Delay']) > 0.01]
print("\nDelay Mismatch Rows:", delay_mismatch.shape[0])

# @title Univariate Analysis
import seaborn as sns
print("\nSummary Statistics for Numerical Variables:")
print(df[numeric_columns].describe().T)

# @title Histograms and box plots (Numerical variables)
for col in numeric_columns:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col}')
    plt.tight_layout()
    plt.show()

# @title Histogram plots (Categorical variables)

categorical_cols = ['Restaurant Name', 'Location', 'Pizza Size', 'Pizza Type',
       'Traffic Level', 'Payment Method', 'Order Month', 'Payment Category']
for col in categorical_cols:
    print(f"\nFrequency Count for {col}:")
    print(df[col].value_counts())

    plt.figure(figsize=(10, 5))
    sns.countplot(x=col, data=df, order=df[col].value_counts().index)
    plt.title(f'Count Plot of {col}')
    plt.xticks(rotation=45)
    plt.show()

# @title Bivariate Analysis
# Numerical vs. Numerical: Correlation matrix
corr_matrix = df[numeric_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Duration vs. Distance by Traffic Level: Scatter plot
import matplotlib.pyplot as plt
import seaborn as sns

# Set figure size and style
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# Create scatter plot
scatter = sns.scatterplot(
    data=df,
    x='Distance (km)',
    y='Delivery Duration (min)',
    hue='Traffic Level',
    size='Toppings Count',
    palette='viridis',
    sizes=(40, 200),
    alpha=0.8
)

# Title and axis labels
plt.title('Delivery Duration vs. Distance by Traffic Level', fontsize=14)
plt.xlabel('Distance (km)')
plt.ylabel('Delivery Duration (min)')

# Show legend and plot
plt.legend(title='Traffic Level', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Delivery Duration by Traffic Level and Peak Hour: Box plot
plt.figure(figsize=(10, 5))
sns.boxplot(x='Traffic Level', y='Delivery Duration (min)', hue='Is Peak Hour', data=df)
plt.title('Delivery Duration by Traffic Level and Peak Hour')
plt.show()

# Pizza Type vs. Payment Method: Cross-tabulation
print("\nCross-tabulation of Pizza Type vs. Payment Method:")
print(pd.crosstab(df['Pizza Type'], df['Payment Method']))

# @title Temporal Analysis
# Extract year and day
df['Order Year'] = df['Order Time'].dt.year
df['Order Day'] = df['Order Time'].dt.day_name()

# Delivery Duration by Order Month
plt.figure(figsize=(10, 5))
sns.boxplot(x='Order Month', y='Delivery Duration (min)', data=df)
plt.title('Delivery Duration by Order Month')
plt.xticks(rotation=45)
plt.show()

# Delivery Duration by Order Hour
plt.figure(figsize=(10, 5))
sns.boxplot(x='Order Hour', y='Delivery Duration (min)', hue='Is Weekend', data=df)
plt.title('Delivery Duration by Order Hour and Weekend')
plt.xticks(rotation=45)
plt.show()

# Line plot for monthly trends
monthly_avg = df.groupby('Order Month')['Delivery Duration (min)'].mean().reset_index()
plt.figure(figsize=(10, 5))
sns.lineplot(x='Order Month', y='Delivery Duration (min)', data=monthly_avg)
plt.title('Average Delivery Duration by Month')
plt.xticks(rotation=45)
plt.show()

# @title Feature Engineering Exploration
# Create Delivery Speed (km/min)
df['Delivery Speed (km/min)'] = 1 / df['Delivery Efficiency (min/km)']

# Create Is Long Distance
df['Is Long Distance'] = df['Distance (km)'] >= 5
print(df['Is Long Distance'].value_counts())

# Verify new features
print("\nNew Features Summary:")
df[['Distance (km)','Delivery Speed (km/min)', 'Is Long Distance']].sample(10)

# @title Statistical Test
from scipy.stats import ttest_ind, f_oneway
# T-test: Delivery Duration by Peak Hour
peak = df[df['Is Peak Hour']]['Delivery Duration (min)']
non_peak = df[~df['Is Peak Hour']]['Delivery Duration (min)']
t_stat, p_value = ttest_ind(peak, non_peak)
print(f"\nT-test (Peak vs. Non-Peak Hour): t-stat={t_stat:.2f}, p-value={p_value:.4f}")

# ANOVA: Delivery Duration by Traffic Level
low = df[df['Traffic Level'] == 'Low']['Delivery Duration (min)']
medium = df[df['Traffic Level'] == 'Medium']['Delivery Duration (min)']
high = df[df['Traffic Level'] == 'High']['Delivery Duration (min)']
f_stat, p_value = f_oneway(low, medium, high)
print(f"ANOVA (Traffic Level): f-stat={f_stat:.2f}, p-value={p_value:.4f}")
