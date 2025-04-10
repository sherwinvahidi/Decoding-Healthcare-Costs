"""
Machine learning models for identifying important features
- Length of Stay
- Mortality Risk
- Treatment Costs
"""

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

feature_cols = [
    'APR Severity of Illness Code',
    'APR DRG Code',
    'APR MDC Code',
    'Age Group Code',   
    'Is Emergency',
    'Total Costs'     
]

def train_borough_cost_models(df):
    """
    Train and compare models across boroughs
    Args: 
        df (pd.DataFrame): processed hospital data
    
    Returns:
        tuple: dict of models, feature importance df
    )"""
    borough_models = {}
    feature_importances = []
    feature_cols.remove("Total Costs")
    
    for borough in df['Borough'].unique():

        borough_data = df[df['Borough'] == borough]
        X = borough_data[feature_cols]  # Use borough-specific features
        y = borough_data['Length of Stay'] #continous number requires regressor
        
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        borough_models[borough] = model
        
        ## What features should be included in our models?
        importances = pd.Series(model.feature_importances_, index=X.columns)
        importances.name = borough
        feature_importances.append(importances)
    
    # Plot comparison
    importance_df = pd.concat(feature_importances, axis=1)
    plt.figure(figsize=(10,6))
    sns.heatmap(importance_df, annot=True, cmap="Blues")
    plt.title("Feature Importance by Borough")
    plt.savefig("figures/feature_importance_by_borough.png")
    plt.close()
    
    return borough_models, importance_df

def train_mortality_models(df):
    """
    Train models for mortality risk prediction
    
    Args: 
        df (pd.DataFrame): processed hospital data
    Returns:
        tuple: trained classifer, feature importance series
    """

    X = df[feature_cols]
    y = df['Mortality_Label'] # classification requires random forest classifer

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    feat_importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=True)

    plt.figure(figsize=(8, 6))
    feat_importance.plot(kind='barh')
    plt.title("Feature Importance for Predicting Extreme Mortality Risk")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("figures/mortality_feature_importance.png", dpi=300)
    plt.close()

    return clf, feat_importance

def train_borough_los_models(df):
    """
    Train borough-specific models for LOS prediction
    
    Args:
        df (pd.DataFrame): processed hospital data
    
    Returns:
        tuple: dict of models, feature importance df
    """
    
    # Initialize storage
    borough_models = {}
    feature_importances = []
    
    for borough in df['Borough'].unique():
        # Filter borough data
        borough_data = df[df['Borough'] == borough].copy()
        
        # Prepare features/target
        X = borough_data[feature_cols]
        y = borough_data['Length of Stay']
        
        # Train model
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        
        # Store model
        borough_models[borough] = model
        
        # Get feature importances
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        importances.name = borough
        feature_importances.append(importances)

    # Plot comparison
    importance_df = pd.concat(feature_importances, axis=1)
    plt.figure(figsize=(10,6))
    sns.heatmap(importance_df, annot=True, cmap="Blues")
    plt.title("LOS Feature Importance by Borough")
    plt.savefig("figures/los_feature_importance_by_borough.png")
    plt.close()
    
    return borough_models, importance_df