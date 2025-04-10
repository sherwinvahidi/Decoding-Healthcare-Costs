"""
Module for analyzing and visualizing borough-specific hospital data patterns.
Includes functions for operational metrics, demographics, and clinical outcomes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plot_utils import set_style

set_style()

# Constants
borough_list = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']

def analyze_top_diagnoses(df, top_n=5):
    """
    Analyze and visualize the top diagnoses by borough
    
    Args: 
        df (pd.DataFrame): processed hospital data
        top_n (int): number of diagnoses to display"""

    # Get top diagnoses overall
    top_diagnoses = df['CCS Diagnosis Description'].value_counts().nlargest(top_n).index
    
    # Filter for top diagnoses
    filtered = df[df['CCS Diagnosis Description'].isin(top_diagnoses)]
    
    # Create pivot table
    pivot = pd.pivot_table(
        data=filtered,
        index='Borough',
        columns='CCS Diagnosis Description',
        values='Total Costs', 
        aggfunc=['count', np.mean],  # Get both frequency and average metric
        fill_value=0
    )
    
    # Flatten multi-index columns
    pivot.columns = [f"{agg}_{diag}" for agg, diag in pivot.columns]

    # Extract just the diagnosis counts
    count_cols = [col for col in pivot.columns if 'count_' in col]
    count_data = pivot[count_cols]
    count_data.columns = [col.replace('count_', '') for col in count_data.columns]

    plt.figure(figsize=(12, 15))
    sns.heatmap(count_data, annot=True, cmap = "Blues")
    plt.xticks(rotation=45, ha='right')
    plt.title("Top 5 Diagnoses by Borough (Patient Counts)")
    plt.tight_layout()
    plt.savefig("figures/diagnosis_counts_heatmap.png")

    # Cost analysis pivot
    cost_pivot = pd.pivot_table(
    df[df['CCS Diagnosis Description'].isin(top_diagnoses)],
    index='Borough',
    columns='CCS Diagnosis Description',
    values='Total Costs',
    aggfunc="median"
)
    plt.figure(figsize=(12, 15))
    sns.heatmap(cost_pivot, annot=True, cmap = "Blues")
    plt.xticks(rotation=45, ha='right')
    plt.title("Median Treatment Costs by Diagnosis and Borough ($)")
    plt.savefig("figures/diagnosis_costs_heatmap.png")
    plt.close()
    

def plot_pairgrid(df):
    """ 
    Create PairGrid with borough-specific analysis
    
    Args:
        df (pd.DataFrame): processed hospital data
    """

    # Create grid
    g = sns.PairGrid(df, vars=['Length of Stay', 'Total Costs', 'APR Severity of Illness Code', "APR DRG Code"],hue='Borough')
    
    # Custom mapping
    g.map_diag(sns.kdeplot, fill=True, alpha=0.5,linewidth=1.5)
    
    g.map_offdiag(sns.scatterplot, size=df['Age Category'], alpha=0.7)

    metrics = ['Length of Stay', 'Total Costs', 'APR Severity of Illness Code']
    # Add correlation coefficients
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        borough_corrs = []
        for borough in df['Borough'].unique():
            corr = df[df['Borough']==borough][metrics].corr().iloc[i,j]
            borough_corrs.append(f"{borough[:3]}: {corr:.2f}")
        g.axes[i,j].annotate(
            "\n".join(borough_corrs),
            xy=(0.5, 0.5), 
            xycoords='axes fraction',
            ha='center',
            fontsize=9
        )
    plt.savefig('figures/enhanced_pairgrid.png')

    # Calculate disparity metrics
    disparity = calculate_borough_disparities(df)
    disparity_df = pd.DataFrame(disparity)
    
    # Calculate ratios relative to best performing borough
    best_case = disparity_df.min()
    disparity_ratios = disparity_df.div(best_case)
    
    return disparity_df, disparity_ratios

def compare_operational_metrics(df):
    """
    Compare key operational metrics across boroughs

    Args: 
        df (pd.DataFrame): processed hospital data"""

    # Calculate Pearson correlation
    corr = df[['APR DRG Code', 'Total Costs']].corr().iloc[0, 1]

    # DRG codes with sufficient frequency vs Total Cost
    top_drg = df['APR DRG Code'].value_counts().loc[lambda x: x > 100].index
    df_plot = df[df['APR DRG Code'].isin(top_drg)]

    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_plot, x='APR DRG Code', y='Total Costs', alpha=0.4)

    # Add regression line
    sns.regplot(data=df_plot, x='APR DRG Code', y='Total Costs', scatter=False, line_kws={"lw": 2})

    plt.title("Relationship Between APR DRG Code and Total Costs", fontsize=16)
    plt.xlabel("APR DRG Code")
    plt.ylabel("Total Costs ($)")
    plt.tight_layout()
    plt.savefig("figures/apr_drg_vs_costs.png", dpi=300)
    plt.close()

    # Drug code vs diagnosis
    top_drg = df['APR DRG Code'].value_counts().nlargest(10).index
    top_diag = df['CCS Diagnosis Description'].value_counts().nlargest(10).index

    # Step 2: Filter the DataFrame correctly
    df_heat = df[df['APR DRG Code'].isin(top_drg) & df['CCS Diagnosis Description'].isin(top_diag)]

    # Create pivot table
    heat_data = df_heat.pivot_table(
        index='CCS Diagnosis Description',
        columns='APR DRG Code',
        aggfunc='size',
        fill_value=0
    )

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heat_data, annot=True, fmt='d', cmap= "Blues")
    plt.title("Top Diagnoses by APR DRG Code")
    plt.xlabel("APR DRG Code")
    plt.ylabel("Diagnosis")
    plt.yticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("figures/drg_vs_diagnosis_heatmap.png", dpi=300)
    plt.close()
    
    # LOS vs Cost joint plot
    jp = sns.jointplot(
        x='Length of Stay',
        y='log_Costs',
        hue='Borough',
        data=df,
        height=10,
        ratio=3,
        joint_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'none'},
        marginal_kws={'common_norm': False}
    )
    
    # Add regression lines per borough
    boroughs = df['Borough'].unique()
    
    for borough in boroughs:
        subset = df[df['Borough'] == borough]
        m, b = np.polyfit(subset['Length of Stay'], subset['log_Costs'], 1)
        x = np.linspace(subset['Length of Stay'].min(), subset['Length of Stay'].max(), 100)
        jp.ax_joint.plot(x, m*x + b, linestyle='--', linewidth=2)
    
    x = subset['Length of Stay']
    y = subset['log_Costs']
    
    for borough in boroughs:
        slope = np.polyfit(subset['Length of Stay'], subset['log_Costs'], 1)[0]
        daily_cost = f"${np.exp(slope)*1000:,.0f}/day"  # Convert log slope to dollar estimate
        plt.gca().annotate(daily_cost, xy=(x.mean(), y.mean()), fontweight='bold')
    
    # Customize plot
    jp.ax_joint.set_xlabel('Length of Stay (Days)')
    jp.ax_joint.set_ylabel('Log(Total Costs)')
    jp.ax_joint.set_title('Hospital Stay Duration vs. Costs by Borough', y=1.1)
    
    # Adjust legend position
    jp.ax_joint.legend(bbox_to_anchor=(1.1, 1), title='Borough', frameon=False)
    
    plt.tight_layout()
    plt.savefig('figures/los_vs_costs_jointplot.png', dpi=300, bbox_inches='tight')
    plt.close()


    # Prepare data
    plot_df = df[['Borough', 'Length of Stay', 'Total Costs']].copy()
    plot_df['Cost_Per_Day'] = plot_df['Total Costs'] / plot_df['Length of Stay']
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Main scatter plot
    ax = sns.scatterplot(
        x='Length of Stay',
        y='Cost_Per_Day',
        hue='Borough',
        data=plot_df,
        alpha=0.7,
        s=100
    )
    
    # Add regression lines and annotations
    stats = []
    for borough in plot_df['Borough'].unique():
        subset = plot_df[plot_df['Borough'] == borough]
        m, b = np.polyfit(subset['Length of Stay'], subset['Cost_Per_Day'], 1)
        x = np.linspace(subset['Length of Stay'].min(), subset['Length of Stay'].max(), 100)
        plt.plot(x, m*x + b, '--', linewidth=2)
        
        # Calculate correlation
        corr = subset['Length of Stay'].corr(subset['Cost_Per_Day'])
        stats.append(f"{borough}: {corr:.2f} correlation")
    
    # Add stats box
    plt.text(
        0.95, 0.95,
        "\n".join(stats),
        transform=plt.gca().transAxes,
        ha='right', va='top',
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    # Customize
    plt.title('Daily Cost Efficiency by Length of Stay', fontsize=16)
    plt.xlabel('Length of Stay (Days)', fontsize=12)
    plt.ylabel('Cost Per Day ($)', fontsize=12)
    plt.legend(title='Borough', bbox_to_anchor=(1.05, 1))
    plt.grid(alpha=0.2)
    
    plt.savefig('figures/presentation_los_cost.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Distribution of Length of Stay
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Length of Stay'], bins=30, kde=True)
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Length of Stay')
    plt.xlabel('Days')
    plt.ylabel('Patient Count')
    plt.xlim(0, 30)  
    plt.savefig("figures/los_distribution.png")
    plt.close()

    sns.countplot(
        data=df,
        x='LOS Category',
        order=['Short', 'Medium', 'Long']
    )
    plt.savefig("figures/los_category_distribution.png")
    plt.close()

    # Average Length of Stay by Borough
    sns.barplot(data=df, x='Borough', y='Length of Stay', estimator=np.mean,
               order=['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'])
    plt.title("Average Length of Stay by Borough")
    plt.ylabel("Days")
    plt.savefig("figures/avg_los_by_borough.png")
    plt.close()

    
    # Emergency Admission Rates
    emergency_count = df[df["Emergency Department Indicator"] == "Y"].groupby('Borough')['Emergency Department Indicator'].count()
    emergency_rates = emergency_count/df.groupby('Borough').size() * 100

    # Sort for better visuals 
    boroughs_sorted = emergency_rates.index
    rates = emergency_rates.values

    plt.hlines(y=boroughs_sorted, xmin=0, xmax=rates)
    plt.plot(rates, boroughs_sorted, "o")

    # Add text labels
    for i, (rate, boro) in enumerate(zip(rates, boroughs_sorted)):
        plt.text(rate + 0.5, boro, f"{rate:.1f}%", va='center', fontsize=11)

    plt.title("Emergency Admission Rates by Borough", fontsize=16)
    plt.xlabel("Percentage of Admissions Through ED")
    plt.ylabel("")
    plt.xlim(0, max(rates) + 10)
    plt.tight_layout()
    plt.savefig("figures/emergency_rates_lollipop.png", dpi=300)
    plt.close()

    
    return emergency_rates

def analyze_patient_demographics(df):
    """
    Compare patient demographics across boroughs
    
    Args:
        df (pd.DataFrame): processed hospital data
    """
    pt_count = df['Borough'].value_counts()
    borough_pt_pct = (pt_count / pt_count.sum()) * 100 

    eth_counts = df.groupby(['Borough', 'Race']).size().unstack().fillna(0)
    eth_counts.plot(kind='bar', stacked=True, figsize=(10,6))
    plt.title("Ethnicity Distribution by Borough")
    plt.ylabel("Patient Count")
    plt.savefig("figures/ethnicity_borough_stacked.png")
    plt.close()

    sns.countplot(data=df, x='Race', hue='Mortality_Label')
    plt.title("Mortality Risk by Race")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figures/mortality_by_race.png")
    plt.close()

    borough_list = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
    plt.figure(figsize=(10, 20))
    for i, borough in enumerate(borough_list):
        plt.subplot(5, 1, i+1)
        sns.countplot(data=df[df["Borough"] == borough], x='Age Category', hue = "Age Category", order = ["Child", "Young Adult", "Adult", "Older Adult", "Senior"])
        plt.title(f"Age Distribution in {borough}")
        plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.savefig("figures/age_by_borough.png")
    plt.close()


    sns.histplot(data=df, x='LOS Category', hue='Age Category', hue_order=["Child", "Young Adult", "Adult", "Older Adult", "Senior"], multiple='stack',discrete=True, stat='count')
    plt.title('Length of Stay by Age')
    plt.xticks(rotation=45)
    plt.savefig('figures/stacked_borough_age.png')
    plt.close()
    
    # Payment Type Distribution
    payment_dist = pd.crosstab(df['Borough'], df['Payment Typology 1'], normalize='index')
    payment_dist.plot(kind='bar', stacked=True, figsize=(14,12))
    plt.xticks(rotation=45, ha='right')
    plt.title("Insurance/Payment Mix by Borough")
    plt.savefig("figures/payment_by_borough.png")
    plt.close()

def compare_clinical_outcomes(df):
    """
    Analyze clinical outcome differences
    
    Args:
        df (pd.DataFrame): processed hospital
    """
    
    mortality_rates = df.groupby('Borough')['APR Risk of Mortality'].apply(lambda x: (x == 'Extreme').mean()) * 100
    sizes = (mortality_rates.values * 10) ** 1.5

    plt.scatter(mortality_rates.values, mortality_rates.index, s=sizes, alpha=0.6)
    plt.title("Mortality Rates by Borough")
    plt.tight_layout()
    plt.savefig("figures/mortality_dotplot.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="APR Severity of Illness Description", order=["Minor", "Moderate", "Major", "Extreme"])
    plt.title("Distribution of Illness Severity")
    plt.ylabel("Number of Discharges")
    plt.tight_layout()
    plt.savefig("figures/severity_distribution.png")
    plt.close()
    
    # Cross-tab and normalize
    severity_dist = pd.crosstab(df["Borough"], df["APR Severity of Illness Description"], normalize="index") * 100
    severity_dist = severity_dist[["Minor", "Moderate", "Major", "Extreme"]] 

    # Plot
    severity_dist.plot(kind="bar", stacked=True, figsize=(10, 6))
    plt.title("Illness Severity by Borough")
    plt.ylabel("Percent of Patients")
    plt.legend(title="Severity Level", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("figures/severity_by_borough.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="APR Severity of Illness Description", y="Length of Stay", order=["Minor", "Moderate", "Major", "Extreme"], showfliers = False)
    plt.title("Length of Stay by Illness Severity")
    plt.tight_layout()
    plt.savefig("figures/los_by_severity.png")
    plt.close()

    # Create DRG code bins
    bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    labels = [f'{i+1}-{j}' for i,j in zip(bins[:-1], bins[1:])]
    df['DRG Group'] = pd.cut(df['APR DRG Code'], bins=bins, labels=labels)
    plt.figure(figsize=(14, 7))
    ax = sns.histplot(data=df, x='APR DRG Code', y='Total Costs', bins=50, cbar=True, cbar_kws={'label': 'Number of Cases'})
    ax.set_title('Drug Cost Distribution Across APR DRG Codes\n(Heatmap Density)', pad=20)
    ax.set_xlabel('APR DRG Code')
    ax.set_ylabel('Total Treatment Cost (USD)')
    ax.yaxis.set_major_formatter(lambda x, _: x/1000)
    plt.tight_layout()
    plt.savefig('figures/drg_cost_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    sns.displot(data=df, x='Length of Stay', hue='Mortality_Label', kind='kde', rug=True)
    plt.title("LOS Density with Rug Plot by Mortality")
    plt.savefig("figures/los_rug_mortality.png")
    plt.close()


def calculate_borough_disparities(df):
    """
    Calculate key disparities between boroughs
    Args: 
        df (pd.DataFrame): processed hospital data
    
    Returns: 
        pd.DataFrame: disparity metrics by borough"""

    metrics = {
        'Total Patients': df.groupby('Borough').size(),
        'Avg Length of Stay': df.groupby('Borough')['Length of Stay'].mean(),
        'Median Total Costs': df.groupby('Borough')['Total Costs'].median(),
        'Cost per Day': df.groupby('Borough').apply(
            lambda x: x['Total Costs'].sum() / x['Length of Stay'].sum()
        ),
        'Emergency Admission %': df.groupby('Borough')['Is Emergency'].mean() * 100,
        'High Severity %': df.groupby('Borough')['APR Severity of Illness Code'].apply(
            lambda x: (x >= 3).mean() * 100  # Assuming 3+ = Major/Extreme
        )
    }

    return pd.DataFrame(metrics).sort_values(by='Total Patients', ascending=False)

def plot_borough_comparisons(metrics):
    """
    Visualize borough comparison metrics
    
    Args:
        metrics (pd.DataFrame): disparity metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Patient Volume
    sns.barplot(data=metrics.reset_index(), x='Borough', y='Total Patients', ax=axes[0, 0])
    axes[0, 0].set_title('Patient Volume by Borough')
    
    # Cost Efficiency
    sns.barplot(data=metrics.reset_index(), x='Borough', y='Cost per Day', ax=axes[0, 1])
    axes[0, 1].set_title('Cost per Day by Borough ($)')
    
    # Emergency Admissions
    sns.barplot(data=metrics.reset_index(), x='Borough', y='Emergency Admission %', ax=axes[1, 0])
    axes[1, 0].set_title('Emergency Admission Rate (%)')
    
    # High Severity Cases
    sns.barplot(data=metrics.reset_index(), x='Borough', y='High Severity %', ax=axes[1, 1])
    axes[1, 1].set_title('Percentage of High-Severity Cases')
    
    plt.tight_layout()
    plt.savefig('figures/borough_operational_comparisons.png')
    plt.close()
