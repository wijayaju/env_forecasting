#!/usr/bin/env python3
"""
Data Center Energy Impact Model v5 - WITH REAL EIA DATA

Uses actual EIA electricity consumption data to analyze correlations
between data center openings and state-level electricity changes.
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')


# Physics constants
HOURS_PER_YEAR = 8760
PUE_VALUES = {
    'hyperscale': 1.10,
    'enterprise': 1.40,
    'colocation': 1.50,
    'default': 1.40,
}
UTILIZATION = {
    'hyperscale': 0.70,
    'enterprise': 0.60,
    'colocation': 0.55,
    'default': 0.65,
}


def load_data():
    """Load real EIA data and data center specs"""
    # Load real EIA data
    eia = pd.read_csv('data/eia_state_electricity_real.csv')
    print(f"Loaded EIA data: {len(eia)} records, years {eia['year'].min()}-{eia['year'].max()}")
    
    # Load data center specs
    dc = pd.read_csv('data/datacenter_specs.csv')
    dc['year_operational'] = pd.to_numeric(dc['year_operational'], errors='coerce')
    dc['capacity_mw'] = pd.to_numeric(dc['capacity_mw'], errors='coerce')
    print(f"Loaded DC data: {len(dc)} data centers")
    
    return eia, dc


def classify_dc_type(row):
    """Classify data center type based on name/capacity"""
    name = str(row.get('data_center_name', '')).lower()
    capacity = row.get('capacity_mw', 0) or 0
    
    hyperscale_keywords = ['google', 'amazon', 'aws', 'meta', 'facebook', 
                           'microsoft', 'azure', 'apple']
    if any(kw in name for kw in hyperscale_keywords) or capacity >= 50:
        return 'hyperscale'
    
    colo_keywords = ['equinix', 'digital realty', 'cyrusone', 'coresite', 
                    'qts', 'switch', 'colocation', 'colo']
    if any(kw in name for kw in colo_keywords):
        return 'colocation'
    
    if capacity >= 10:
        return 'enterprise'
    
    return 'default'


def estimate_dc_energy(capacity_mw, dc_type='default', year=None):
    """Physics-based energy estimation"""
    pue = PUE_VALUES.get(dc_type, PUE_VALUES['default'])
    utilization = UTILIZATION.get(dc_type, UTILIZATION['default'])
    
    # Efficiency improvements over time
    if year and year > 2010:
        years_improvement = min(year - 2010, 14)
        pue *= (0.99 ** years_improvement)
    
    return capacity_mw * HOURS_PER_YEAR * utilization * pue


def prepare_training_data(eia, dc):
    """
    Create training dataset by matching DC openings with EIA electricity changes
    """
    # Impute missing capacity
    state_median = dc.groupby('state')['capacity_mw'].transform('median')
    overall_median = dc['capacity_mw'].median()
    dc['capacity_mw_est'] = dc['capacity_mw'].fillna(state_median).fillna(overall_median).fillna(10)
    
    # Classify DC type
    dc['dc_type'] = dc.apply(classify_dc_type, axis=1)
    
    # Estimate energy for each DC
    dc['estimated_energy_mwh'] = dc.apply(
        lambda row: estimate_dc_energy(
            row['capacity_mw_est'],
            row['dc_type'],
            row['year_operational'] if pd.notna(row['year_operational']) else 2015
        ),
        axis=1
    )
    
    # Aggregate DC stats by state and year
    dc_with_year = dc[dc['year_operational'].notna()].copy()
    dc_by_state_year = dc_with_year.groupby(['state', 'year_operational']).agg({
        'data_center_id': 'count',
        'capacity_mw_est': 'sum',
        'estimated_energy_mwh': 'sum',
    }).reset_index()
    dc_by_state_year.columns = ['state', 'year', 'new_dc_count', 'new_dc_capacity_mw', 'new_dc_energy_mwh']
    dc_by_state_year['year'] = dc_by_state_year['year'].astype(int)
    
    # Cumulative DC stats
    dc_by_state_year = dc_by_state_year.sort_values(['state', 'year'])
    dc_by_state_year['cumulative_dc_count'] = dc_by_state_year.groupby('state')['new_dc_count'].cumsum()
    dc_by_state_year['cumulative_dc_capacity'] = dc_by_state_year.groupby('state')['new_dc_capacity_mw'].cumsum()
    dc_by_state_year['cumulative_dc_energy'] = dc_by_state_year.groupby('state')['new_dc_energy_mwh'].cumsum()
    
    # Merge with EIA data
    merged = pd.merge(
        eia,
        dc_by_state_year,
        on=['state', 'year'],
        how='left'
    )
    
    # Fill NaN values for years with no DC openings
    for col in ['new_dc_count', 'new_dc_capacity_mw', 'new_dc_energy_mwh']:
        merged[col] = merged[col].fillna(0)
    
    # Forward fill cumulative values
    merged = merged.sort_values(['state', 'year'])
    merged['cumulative_dc_count'] = merged.groupby('state')['cumulative_dc_count'].ffill().fillna(0)
    merged['cumulative_dc_capacity'] = merged.groupby('state')['cumulative_dc_capacity'].ffill().fillna(0)
    merged['cumulative_dc_energy'] = merged.groupby('state')['cumulative_dc_energy'].ffill().fillna(0)
    
    return merged, dc


def analyze_correlation(data):
    """Analyze correlation between DC capacity and electricity consumption"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS - REAL EIA DATA")
    print("="*60)
    
    # Filter to rows with valid data
    valid = data[data['yoy_change_mwh'].notna()].copy()
    
    # Correlation between DC energy added and electricity change
    corr_new = valid['new_dc_energy_mwh'].corr(valid['yoy_change_mwh'])
    corr_cumulative = valid['cumulative_dc_energy'].corr(valid['total_consumption_mwh'])
    
    print(f"\nCorrelation: New DC Energy vs YoY Change: {corr_new:.4f}")
    print(f"Correlation: Cumulative DC Energy vs Total Consumption: {corr_cumulative:.4f}")
    
    # Top states analysis
    print("\n" + "-"*60)
    print("TOP DATA CENTER STATES - Real Electricity Impact")
    print("-"*60)
    
    latest_year = valid['year'].max()
    latest = valid[valid['year'] == latest_year].copy()
    latest = latest.sort_values('cumulative_dc_capacity', ascending=False).head(10)
    
    print(f"\nLatest Year: {latest_year}")
    print(f"{'State':<20} {'DCs':<8} {'DC Cap (MW)':<12} {'State Elec (TWh)':<18} {'DC % of State':<15}")
    print("-"*75)
    
    for _, row in latest.iterrows():
        state_twh = row['total_consumption_mwh'] / 1e6
        dc_twh = row['cumulative_dc_energy'] / 1e6
        pct = (dc_twh / state_twh * 100) if state_twh > 0 else 0
        print(f"{row['state']:<20} {int(row['cumulative_dc_count']):<8} {row['cumulative_dc_capacity']:>10,.0f} {state_twh:>16,.1f} {pct:>13.1f}%")
    
    return valid


def train_model(data):
    """Train ML model to predict electricity changes from DC additions"""
    print("\n" + "="*60)
    print("MACHINE LEARNING MODEL")
    print("="*60)
    
    # Prepare features
    features = ['new_dc_count', 'new_dc_capacity_mw', 'new_dc_energy_mwh',
                'cumulative_dc_count', 'cumulative_dc_capacity', 'cumulative_dc_energy',
                'total_consumption_mwh']
    
    # Remove rows with missing values
    model_data = data.dropna(subset=features + ['yoy_change_mwh'])
    
    if len(model_data) < 50:
        print(f"Warning: Only {len(model_data)} samples available for training")
    
    X = model_data[features]
    y = model_data['yoy_change_mwh']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:,.0f} MWh")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Feature importance
    print("\nFeature Importance:")
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance.iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='r2')
    print(f"\nCross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return model, scaler


def calculate_dc_share_of_electricity(data, dc):
    """Calculate what percentage of state electricity goes to data centers"""
    print("\n" + "="*60)
    print("DATA CENTER SHARE OF STATE ELECTRICITY")
    print("="*60)
    
    # Get latest year
    latest_year = data['year'].max()
    latest = data[data['year'] == latest_year].copy()
    
    # Calculate total DC energy by state (all DCs, not just those with known years)
    dc_by_state = dc.groupby('state').agg({
        'data_center_id': 'count',
        'capacity_mw_est': 'sum',
        'estimated_energy_mwh': 'sum',
    }).reset_index()
    dc_by_state.columns = ['state', 'total_dc_count', 'total_dc_capacity', 'total_dc_energy']
    
    # Merge with latest EIA
    analysis = pd.merge(latest[['state', 'total_consumption_mwh']], dc_by_state, on='state', how='outer')
    analysis = analysis.dropna()
    
    # Calculate percentage
    analysis['dc_pct_of_state'] = (analysis['total_dc_energy'] / analysis['total_consumption_mwh']) * 100
    analysis = analysis.sort_values('dc_pct_of_state', ascending=False)
    
    print(f"\nAll data centers (estimated) vs {latest_year} state electricity:")
    print(f"{'State':<20} {'DCs':<8} {'DC Energy (TWh)':<16} {'State Elec (TWh)':<18} {'DC % of State':<15}")
    print("-"*80)
    
    for _, row in analysis.head(15).iterrows():
        dc_twh = row['total_dc_energy'] / 1e6
        state_twh = row['total_consumption_mwh'] / 1e6
        print(f"{row['state']:<20} {int(row['total_dc_count']):<8} {dc_twh:>14,.2f} {state_twh:>16,.1f} {row['dc_pct_of_state']:>13.1f}%")
    
    # National totals
    total_dc_energy = analysis['total_dc_energy'].sum()
    total_state_energy = analysis['total_consumption_mwh'].sum()
    national_pct = (total_dc_energy / total_state_energy) * 100
    
    print("-"*80)
    print(f"{'NATIONAL TOTAL':<20} {int(analysis['total_dc_count'].sum()):<8} {total_dc_energy/1e6:>14,.2f} {total_state_energy/1e6:>16,.1f} {national_pct:>13.1f}%")
    
    return analysis


def save_results(data, dc, model, scaler):
    """Save all results"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Save training data
    data.to_csv('data/training_data_real_eia.csv', index=False)
    print(f"\nSaved training data to data/training_data_real_eia.csv")
    
    # Save DC estimates
    dc.to_csv('data/datacenter_energy_estimates_v5.csv', index=False)
    print(f"Saved DC estimates to data/datacenter_energy_estimates_v5.csv")
    
    # Save model
    joblib.dump({'model': model, 'scaler': scaler}, 'models/energy_model_v5_real.joblib')
    print(f"Saved model to models/energy_model_v5_real.joblib")


def main():
    print("="*60)
    print("DATA CENTER ENERGY MODEL v5 - REAL EIA DATA")
    print("="*60)
    
    # Load data
    eia, dc = load_data()
    
    # Prepare training data
    merged, dc = prepare_training_data(eia, dc)
    
    # Analyze correlations
    valid_data = analyze_correlation(merged)
    
    # Calculate DC share of electricity
    calculate_dc_share_of_electricity(merged, dc)
    
    # Train model
    model, scaler = train_model(valid_data)
    
    # Save results
    save_results(merged, dc, model, scaler)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
This model uses REAL EIA electricity data (2001-2024) to analyze
the relationship between data center growth and state electricity
consumption.

Key findings are shown above. The correlation analysis reveals
how much of state electricity is attributable to data centers.
""")


if __name__ == '__main__':
    main()
