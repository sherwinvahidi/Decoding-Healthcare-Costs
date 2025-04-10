"""
Data loading and preprocessing functions for NY hospital data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_data(filepath):
    '''
    Read the df from file specified in main.py
    '''
    df = pd.read_csv(filepath, dtype=str, low_memory=False)

    return df

def get_info(df: pd.DataFrame) -> None:
    """
    Print basic information about the dataset.
    """
    print("\nBasic Dataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nSample Rows:")
    print(df.head())

def filter_boroughs(df):
    '''
    Extract and separate data by NYC borough
    '''
    # Convert ZIP to string and take first 3 digits
    df['Zip3'] = df['Zip Code - 3 digits'].astype(str).str[:3]

    # Create boolean masks for each borough
    df['Manhattan'] = df['Zip3'].isin(['100'])
    df['Bronx'] = df['Zip3'].isin(['104'])
    df['Brooklyn'] = df['Zip3'].isin(['112'])
    df['Queens'] = df['Zip3'].isin(['113', '114'])
    df['Staten Island'] = df['Zip3'].isin(['103'])

    # Combine into a single borough column using np.select
    conditions = [
    df['Manhattan'],
    df['Bronx'],
    df['Brooklyn'],
    df['Queens'],
    df['Staten Island']
    ]

    choices = ['Manhattan', 'Bronx', 'Brooklyn', 'Queens', 'Staten Island']

    df['Borough'] = np.select(conditions, choices, default='Outside NYC')

    # Clean up temporary columns
    df.drop(['Zip3', 'Manhattan', 'Bronx', 'Brooklyn', 'Queens', 'Staten Island'], axis=1, inplace=True)

    # Verify counts
    print("Patient Distribution by Borough:")
    pt_count = df['Borough'].value_counts()
    print(pt_count)

    df = df[(df["Borough"] != "Outside NYC")]

    return df

def prepare_features(df):
    """
    Clean and transform the dataset for analysis.
    """
    # Convert numerical columns
    num_cols = [
        'Length of Stay', 'Total Charges', 'Total Costs',
        'APR DRG Code', 'APR MDC Code', 'APR Severity of Illness Code'
    ]
    for col in num_cols:
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')

    # Filter to remove invalid entries
    df = df[df['Length of Stay'] > 0]
    df = df[df['Total Charges'] > 0]
    df = df[df['Total Costs'] > 0]

    # Drop rows with critical missing values
    df.dropna(subset=num_cols, inplace=True)

    # assign categorical age groups 
    df['Age Category'] = df['Age Group'].map({
        '0 to 17': "Child", '18 to 29': "Young Adult", '30 to 49': "Adult", 
        '50 to 69': "Older Adult", '70 or Older': "Senior"
    })

    df['Age Group Code'] = df['Age Group'].map({
    '0 to 17': 0,
    '18 to 29': 1,
    '30 to 49': 2,
    '50 to 69': 3,
    '70 or Older': 4
    })

    df['Mortality_Label'] = (df['APR Risk of Mortality'] == 'Extreme').astype(int)

    df['Is Emergency'] = df['Type of Admission'].map(lambda x: 1 if 'Emergency' in str(x) else 0)

    # Can we categorize LOS to simplify analysis of patient stay types?
    q1, q2, q3 = df['Length of Stay'].quantile([0.25, 0.5, 0.75])
    df['LOS Category'] = df['Length of Stay'].map(
        lambda x: 'Short' if x <= q1 else 'Medium' if x <= q2 else 'Long'
    )
    

    # Feature: Cost-Charge Ratio
    df['Cost_Charge_Ratio'] = df['Total Costs'] / df['Total Charges']

    # Ensure numeric types before log transform
    df['Total Costs'] = pd.to_numeric(df['Total Costs'], errors='coerce')
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

    # Feature: Log-transformed cost and charges for modeling
    df['log_Costs'] = np.log1p(df['Total Costs'])
    df['log_Charges'] = np.log1p(df['Total Charges'])

    return df

def add_borough_specific_features(df):
    """Add features that help compare boroughs"""
    # Borough-specific normalization factors
    borough_means = df.groupby('Borough')[['Total Charges', 'Total Costs']].mean()
    df = df.merge(borough_means, on='Borough', suffixes=('', '_Borough_Avg'))
    
    # Relative cost metrics
    df['Relative Charges'] = df['Total Charges'] / df['Total Charges_Borough_Avg']
    df['Relative Costs'] = df['Total Costs'] / df['Total Costs_Borough_Avg']
    
    return df