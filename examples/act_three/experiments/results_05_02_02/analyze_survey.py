import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import itertools

input_file = 'survey_data.csv'
output_summary_file = 'analysis_summary.csv'

# Define conditions and map
conditions = ['unsteered', 'pe_only', 'repe_only', 'hybrid']
q_map = {'1': 'Appropriate', '2': 'Natural', '3': 'Non-Sycophantic'}

# Load the data
df = pd.read_csv(input_file)

# Extract likert columns
likert_cols = [c for c in df.columns if c.startswith('likert_')]

# Melt the dataframe
df_melt = pd.melt(df, id_vars=['participant'], value_vars=likert_cols, var_name='item', value_name='score')

def parse_item(item):
    parts = item.split('_')
    q_num = parts[-1]
    
    cond = None
    for c in conditions:
        if item.endswith(f"_{c}_{q_num}"):
            cond = c
            break
            
    return pd.Series({'condition': cond, 'question': q_num})

# Apply parsing
parsed = df_melt['item'].apply(parse_item)
df_melt = pd.concat([df_melt, parsed], axis=1)

# Clean scores
df_melt['score'] = pd.to_numeric(df_melt['score'], errors='coerce')
df_melt = df_melt.dropna(subset=['score'])
df_melt['metric'] = df_melt['question'].map(q_map)

# Extract scenario for later correlation analysis
def extract_scenario(item):
    c, q = parse_item(item)
    # the scenario ID is what's left after stripping "likert_" and the condition/q_num
    # likert_pos_e_01_unsteered_1 -> pos_e_01
    return item.replace('likert_', '').replace(f'_{c}_{q}', '')

df_melt['scenario'] = df_melt['item'].apply(extract_scenario)

# ---------------------------------------------------------
# 1. Descriptive Statistics (Mean, Median, Mode, SD, SEM)
# ---------------------------------------------------------
# Compute statistics
summary = df_melt.groupby(['condition', 'metric'])['score'].agg([
    'mean',
    ('median', 'median'),
    # for mode, we take the first mode if multiple exist
    ('mode', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
    ('std', 'std'),
    ('sem', 'sem'),
    'count'
]).reset_index()

# Sort the summary
summary['cond_cat'] = pd.Categorical(summary['condition'], categories=conditions, ordered=True)
summary = summary.sort_values(['metric', 'cond_cat']).drop('cond_cat', axis=1)

print("=========================================")
print("        DESCRIPTIVE STATISTICS           ")
print("=========================================")
print(summary.to_string(index=False))
summary.to_csv(output_summary_file, index=False)
print(f"\nDescriptive statistics saved to {output_summary_file}\n")


# ---------------------------------------------------------
# 2. Frequency Tables & Bar Charts
# ---------------------------------------------------------
print("=========================================")
print("      FREQUENCY TABLES (COUNTS)          ")
print("=========================================")
for metric in q_map.values():
    subset = df_melt[df_melt['metric'] == metric]
    
    # Frequency table
    freq = pd.crosstab(subset['condition'], subset['score'])
    # Ensure all scores 1-7 are represented as columns
    for i in range(1, 8):
        if i not in freq.columns:
            freq[i] = 0
    freq = freq.reindex(columns=range(1, 8)).reindex(conditions)
    
    print(f"--- {metric} ---")
    print(freq)
    print("\n")
    
    # Bar Chart (Percentages)
    freq_pct = freq.div(freq.sum(axis=1), axis=0) * 100
    
    ax = freq_pct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    plt.title(f'Distribution of Responses for {metric}', fontsize=14)
    plt.xlabel('Condition', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.legend(title='Likert Score (1-7)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    chart_filename = f'chart_{metric}.png'
    plt.savefig(chart_filename)
    plt.close()
    
print("Saved stacked bar charts for each metric (chart_*.png).\n")


test_results_file = 'statistical_test_results.txt'
with open(test_results_file, 'w') as f:
    def log_and_print(msg):
        print(msg)
        f.write(msg + '\n')

    # ---------------------------------------------------------
    # 3. Kruskal-Wallis H Test
    # ---------------------------------------------------------
    log_and_print("=========================================")
    log_and_print("         KRUSKAL-WALLIS H TEST           ")
    log_and_print("=========================================")
    kw_results = {}
    for metric in q_map.values():
        subset = df_melt[df_melt['metric'] == metric]
        groups = [subset[subset['condition'] == c]['score'].dropna().values for c in conditions]
        
        stat, p = stats.kruskal(*groups)
        log_and_print(f"{metric:<16}: H-stat = {stat:>8.3f}, p-value = {p:.3e}")
        kw_results[metric] = p
    log_and_print("\n")


    # ---------------------------------------------------------
    # 4. Mann-Whitney U Tests
    # ---------------------------------------------------------
    log_and_print("=========================================")
    log_and_print("          MANN-WHITNEY U TESTS           ")
    log_and_print("=========================================")
    for metric in q_map.values():
        log_and_print(f"--- {metric} ---")
        subset = df_melt[df_melt['metric'] == metric]
        pairs = list(itertools.combinations(conditions, 2))
        
        for c1, c2 in pairs:
            group1 = subset[subset['condition'] == c1]['score'].dropna().values
            group2 = subset[subset['condition'] == c2]['score'].dropna().values
            if len(group1) > 0 and len(group2) > 0:
                stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                log_and_print(f"  {c1:<10} vs {c2:<10}: U = {stat:>7.1f}, p = {p:.3e}")
            else:
                log_and_print(f"  {c1:<10} vs {c2:<10}: Not enough data")
        log_and_print("")


    # ---------------------------------------------------------
    # 5. Spearman's Rank Correlation
    # ---------------------------------------------------------
    log_and_print("=========================================")
    log_and_print("       SPEARMAN'S RANK CORRELATION       ")
    log_and_print("=========================================")
    # Pivot the dataframe to get metrics as columns so we can correlate them per response instance
    # An instance is defined by participant, scenario, and condition.
    df_pivot = df_melt.pivot_table(index=['participant', 'scenario', 'condition'], columns='metric', values='score').reset_index()

    # Drop rows that are missing any metric, as correlation requires pairwise complete data
    metrics = list(q_map.values())
    df_pivot = df_pivot.dropna(subset=metrics)

    for m1, m2 in itertools.combinations(metrics, 2):
        r, p = stats.spearmanr(df_pivot[m1], df_pivot[m2])
        log_and_print(f"{m1:<16} vs {m2:<16}: rho = {r:>6.3f}, p = {p:.3e}")

    log_and_print(f"\nStatistical test results saved to {test_results_file}")

print("\nAnalysis complete.")
