"""
PROJECT OVERVIEW
This project uses the 2016 NY SPARCS inpatient discharge dataset to explore and model key aspects of hospital operations:
- Predicting Length of Stay (LOS)
- Predicting Total Charges
- Analyzing disparities in care outcomes
"""

from data_preprocessing import (load_and_clean_data,
    get_info,
    filter_boroughs,
    prepare_features,
    add_borough_specific_features)
from borough_analysis import (compare_operational_metrics,
    analyze_patient_demographics,
    compare_clinical_outcomes,
    calculate_borough_disparities, 
    plot_pairgrid, 
    analyze_top_diagnoses,
    plot_borough_comparisons)
from modeling import train_borough_cost_models, train_mortality_models, train_borough_los_models

def main():
    """Main execution function for the hospital data analysis pipeline."""

    # Step 1: Data Loading and Initial Cleaning
    print("=== Loading and Cleaning Data ===")
    raw_df = load_and_clean_data("data/nyhospital.csv")
    get_info(raw_df)  # Print initial dataset info
    
    # Step 2: Borough Filtering
    print("\n=== Filtering NYC Boroughs ===")
    nyc_df = filter_boroughs(raw_df)
    
    # Step 3: Feature Preparation
    print("\n=== Preparing Features ===")
    processed_df = prepare_features(nyc_df)
    processed_df = add_borough_specific_features(processed_df)
    
    # Step 4: Borough Analysis
    #print("\n=== Analyzing Borough Differences ===")
    analyze_top_diagnoses(processed_df, 10)
    compare_operational_metrics(processed_df)
    analyze_patient_demographics(processed_df)          # Core operational metrics
    compare_clinical_outcomes(processed_df) # Clinical patterns
    plot_pairgrid(processed_df)           
    metrics = calculate_borough_disparities(processed_df)
    plot_borough_comparisons(metrics)

    
    # # Borough-specific modeling
    print("\nTraining borough models...")
    
    # # Step 5: Borough-Specific Modeling
    # print("\n=== Training Borough-Specific Models ===")
    cost_models, cost_importance_df = train_borough_cost_models(processed_df)
    mortality_models, mortality_importance_df = train_mortality_models(processed_df)
    los_models, los_importance_df = train_borough_los_models(processed_df)
    
    print("\n=== Analysis Complete ===")
    print(f"Generated visualizations in 'figures/' directory")

if __name__ == "__main__":
    main()