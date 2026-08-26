# Cleaned from the completed Break Through Tech data-preparation notebook.

import os
import pandas as pd
import scipy.stats as stats
import seaborn as sns

filename = os.path.join(os.getcwd(), "data", "censusData.csv")
df = pd.read_csv(filename)

# Winsorize extreme values in years of education.
df['education_years'] = stats.mstats.winsorize(
    df['education-num'],
    limits=[0.01, 0.01]
)

# Preserve missingness as features before imputing.
df['age_na'] = df['age'].isnull()
df['hours-per-week_na'] = df['hours-per-week'].isnull()

df['age'] = df['age'].fillna(df['age'].mean())
df['hours-per-week'] = df['hours-per-week'].fillna(df['hours-per-week'].mean())

# Correlation analysis.
exclude = ['education_years', 'education-num']
corrs = df.corr(numeric_only=True)['education_years'].drop(exclude, errors='ignore')
corrs_sorted = corrs.sort_values(ascending=False)
print(corrs_sorted.head(10))

# Visualize relationships among top numerical correlates.
top_two = list(corrs_sorted.index[:2])
plot_df = df[['education_years'] + top_two]
sns.pairplot(plot_df, kind='kde', corner=True)
