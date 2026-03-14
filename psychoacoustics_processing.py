import json
import pandas as pd
import altair as alt
from scipy.stats import mannwhitneyu, fisher_exact


def calculate_mean_val(row, col1, col2):
    """
    Calculates the mean of two columns. 
    Logic: If both > 0, return average. If only one > 0, return that value.
    Otherwise, return 0.
    """
    v1, v2 = row[col1], row[col2]
    if v1 > 0 and v2 > 0:
        return (v1 + v2) / 2
    if v1 > 0:
        return v1
    if v2 > 0:
        return v2
    return 0


def calculate_effect_size(data1, data2):
    """Calculates Hedges' g effect size."""
    n1, n2 = len(data1), len(data2)
    mean1, mean2 = data1.mean(), data2.mean()
    var1, var2 = data1.var(ddof=1), data2.var(ddof=1)

    # Pooled standard deviation
    pooled_sd = (((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)) ** 0.5

    # Cohen's d
    cohen_d = (mean1 - mean2) / pooled_sd

    # Hedges' g correction
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    hedges_g = cohen_d * correction

    return hedges_g


def fisher_test_zero_nonzero(df1, df2, column):
    """Performs Fisher's Exact Test for presence vs. absence of values."""
    zero1 = (df1[column] == 0).sum()
    nonzero1 = (df1[column] > 0).sum()
    zero2 = (df2[column] == 0).sum()
    nonzero2 = (df2[column] > 0).sum()

    contingency_table = [[zero1, nonzero1], [zero2, nonzero2]]
    odds_ratio, p_value = fisher_exact(contingency_table)
    return odds_ratio, p_value


def create_boxplot(df, title, y_label):
    """Generates an Altair boxplot with overlaid data points."""
    boxplot = alt.Chart(df).mark_boxplot(
        size=40,
        ticks=True,
        median=True
    ).encode(
        x=alt.X('group:N', title='Group',
                axis=alt.Axis(labelFontSize=14, titleFontSize=16, grid=False)),
        y=alt.Y('value:Q', title=y_label,
                axis=alt.Axis(grid=False)),
        color=alt.Color('group:N', title='Group',
                        scale=alt.Scale(range=['coral', 'skyblue']))
    )

    points = alt.Chart(df).mark_circle(size=20, color='red').encode(
        x=alt.X('group:N'),
        y=alt.Y('value:Q')
    )

    return (boxplot + points).properties(
        width=400,
        height=400,
        title=title
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_scale(bandPaddingInner=0.9)


# --- 1. Load Data ---
# Adjust filename
with open(".json", "r") as f:
    data_ctrl = json.load(f)
with open(".json", "r") as f:
    data_pt = json.load(f)

df_ctrl = pd.DataFrame(data_ctrl)
df_pt = pd.DataFrame(data_pt)

# --- 2. Data Processing ---
# Calculate means for GD and HP
for df in [df_ctrl, df_pt]:
    df['mean_gd'] = df.apply(lambda r: calculate_mean_val(r, 'GD1', 'GD2'), axis=1)
    df['mean_hp'] = df.apply(lambda r: calculate_mean_val(r, 'HP1', 'HP2'), axis=1)

# Filter for IDs and non-zero HP
# Note: Add your IDs to this list for processing
excluded_ids = []  # Placeholder: Insert IDs to exclude here

df_ctrl_filtered = df_ctrl[(df_ctrl['mean_hp'] > 0) & (~df_ctrl['ID'].isin(excluded_ids))]
df_pt_filtered = df_pt[(df_pt['mean_hp'] > 0) & (~df_pt['ID'].isin(excluded_ids))]

# --- 3. Statistical Analysis ---
# Wilcoxon tests
stat_gd, p_gd = mannwhitneyu(df_ctrl['mean_gd'], df_pt['mean_gd'], alternative='less')
stat_hp, p_hp = mannwhitneyu(df_ctrl_filtered['mean_hp'], 
                             df_pt_filtered['mean_hp'], alternative='greater')

# Effect sizes
eff_gd = calculate_effect_size(df_pt['mean_gd'], df_ctrl['mean_gd'])
eff_hp = calculate_effect_size(df_pt_filtered['mean_hp'], df_ctrl_filtered['mean_hp'])

# Fisher test
odds_hp, p_fish_hp = fisher_test_zero_nonzero(df_ctrl, df_pt, 'mean_hp')

# --- 4. Export Results ---
with open("psycho_results.txt", "w") as f:
    f.write(f"Wilcoxon GD: stat={stat_gd}, p={p_gd}\n")
    f.write(f"Wilcoxon HP: stat={stat_hp}, p={p_hp}\n")
    f.write(f"Effect Size Hedges g (GD): {eff_gd}\n")
    f.write(f"Effect Size Hedges g (HP): {eff_hp}\n")
    f.write(f"Fisher Exact Test HP: odds={odds_hp}, p={p_fish_hp}\n")

# --- 5. Visualization ---
# Prepare combined dataframes
df_ctrl['group'], df_pt['group'] = 'Controls', 'FRDA'
df_ctrl_filtered['group'], df_pt_filtered['group'] = 'Controls', 'FRDA'

df_combined_gd = pd.concat([
    df_ctrl[['mean_gd', 'group']].rename(columns={'mean_gd': 'value'}),
    df_pt[['mean_gd', 'group']].rename(columns={'mean_gd': 'value'})
])

df_combined_hp = pd.concat([
    df_ctrl_filtered[['mean_hp', 'group']].rename(columns={'mean_hp': 'value'}),
    df_pt_filtered[['mean_hp', 'group']].rename(columns={'mean_hp': 'value'})
])

# Create and save plots
boxplot_gd = create_boxplot(df_combined_gd, "Mean GD Comparison", "ms")
boxplot_hp = create_boxplot(df_combined_hp, "Mean HP Comparison", "kHz")

boxplot_gd.save("boxplot_mean_gd.html")
boxplot_hp.save("boxplot_mean_hp.html")
