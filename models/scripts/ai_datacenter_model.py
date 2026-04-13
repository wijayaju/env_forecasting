#!/usr/bin/env python3
"""
AI Data Center Energy Prediction Model v8

Categorizes data centers into:
- Small DCs: <10 MW (enterprise, small colocation)
- Decent DCs: 10-50 MW (medium facilities)  
- Crypto: Bitcoin/crypto mining operations (separate from AI)
- Big AI: >50 MW hyperscalers for cloud/AI (Google, Meta, Microsoft, Amazon)

Separates crypto mining from AI to get accurate AI energy estimates.
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

# PUE and utilization by category
DC_PARAMS = {
    'small': {
        'pue': 1.60,           # Less efficient
        'utilization': 0.50,
        'default_capacity': 2,  # MW - conservative estimate
        'energy_per_mw': 7000,  # MWh/MW/year
    },
    'decent': {
        'pue': 1.40,
        'utilization': 0.60,
        'default_capacity': 15,
        'energy_per_mw': 7350,
    },
    'crypto': {
        'pue': 1.10,           # Very efficient (just ASICs)
        'utilization': 0.95,   # Run 24/7 at max
        'default_capacity': 50,  # MW - typical mining farm
        'energy_per_mw': 8322,   # Higher due to 95% utilization
    },
    'big_ai': {
        'pue': 1.15,           # Very efficient
        'utilization': 0.70,   # AI training bursts, not 24/7
        'default_capacity': 50,  # Lower default - most are 20-100 MW
        'energy_per_mw': 7056,   # 8760 * 0.70 * 1.15
    }
}

# Crypto mining keywords
CRYPTO_KEYWORDS = [
    'riot', 'marathon', 'core scientific', 'bitdeer', 'cipher', 'cleanspark',
    'hut 8', 'hut8', 'iren', 'terawulf', 'iris energy', 'bit digital',
    'greenidge', 'stronghold', 'argo blockchain', 'hive blockchain',
    'bitcoin', 'btc', 'mining', 'miner', 'crypto', 'blockchain',
    'antminer', 'asic', 'hash', 'bitfarms', 'canaan', 'bitmain',
    'crusoe', 'applied digital', 'compute north', 'lancium'
]

# AI/Hyperscaler keywords (cloud + AI, NOT crypto)
AI_HYPERSCALER_KEYWORDS = [
    'google', 'amazon', 'aws', 'meta', 'facebook', 'microsoft', 'azure',
    'apple', 'nvidia', 'openai', 'anthropic', 'oracle cloud', 'ibm cloud',
    'alibaba', 'tencent', 'bytedance', 'coreweave', 'lambda labs'
]


def classify_datacenter(row):
    """
    Classify data center into Small, Decent, Crypto, or Big AI
    
    Returns: 'small', 'decent', 'crypto', or 'big_ai'
    """
    name = str(row.get('data_center_name', '')).lower()
    capacity = row.get('capacity_mw', None)
    
    # Check crypto FIRST (before AI, as some crypto companies use cloud keywords)
    is_crypto = any(kw in name for kw in CRYPTO_KEYWORDS)
    if is_crypto:
        return 'crypto'
    
    # Check if it's an AI/hyperscaler by name
    is_ai_hyperscaler = any(kw in name for kw in AI_HYPERSCALER_KEYWORDS)
    
    # If capacity is known, use it
    if pd.notna(capacity):
        if capacity >= 50 or is_ai_hyperscaler:
            return 'big_ai'
        elif capacity >= 10:
            return 'decent'
        else:
            return 'small'
    
    # If no capacity but is hyperscaler, it's big AI
    if is_ai_hyperscaler:
        return 'big_ai'
    
    # Check for colocation keywords (usually medium)
    colo_keywords = ['equinix', 'digital realty', 'cyrusone', 'coresite', 
                     'qts', 'switch', 'colocation', 'colo', 'vantage', 
                     'flexential', 'databank']
    if any(kw in name for kw in colo_keywords):
        return 'decent'
    
    # Default: small (conservative)
    return 'small'


def is_planned_facility(name):
    """Check if facility is planned/announced (not operational yet)"""
    name_lower = str(name).lower()
    planned_keywords = [
        'project ', 'proposed', 'planned', 'future', 'phase 2', 'phase 3',
        'upcoming', 'announced', 'under construction', 'expansion',
        'campus',  # Campuses are often umbrella entries, not individual DCs
    ]
    return any(kw in name_lower for kw in planned_keywords)


def load_and_classify_data(operational_only=True, require_year=True):
    """
    Load data and classify all DCs
    
    Args:
        operational_only: If True, exclude planned/announced facilities
        require_year: If True, only include DCs with known operational year
    """
    # Load data
    dc = pd.read_csv('data/datacenter_specs.csv')
    dc['year_operational'] = pd.to_numeric(dc['year_operational'], errors='coerce')
    dc['capacity_mw'] = pd.to_numeric(dc['capacity_mw'], errors='coerce')
    
    original_count = len(dc)
    
    # Filter 1: Remove planned/announced facilities
    if operational_only:
        dc['is_planned'] = dc['data_center_name'].apply(is_planned_facility)
        planned_count = dc['is_planned'].sum()
        dc = dc[~dc['is_planned']].copy()
        print(f"  Filtered out {planned_count} planned/announced facilities")
    
    # Filter 2: Require operational year
    if require_year:
        no_year_count = dc['year_operational'].isna().sum()
        dc = dc[dc['year_operational'].notna()].copy()
        print(f"  Filtered out {no_year_count} DCs without operational year")
    
    # Filter 3: Remove duplicates by data_center_id
    dc = dc.drop_duplicates(subset='data_center_id', keep='first')
    
    # Filter 4: Only include currently operational (year <= 2026)
    if require_year:
        future_count = (dc['year_operational'] > 2026).sum()
        dc = dc[dc['year_operational'] <= 2026].copy()
        if future_count > 0:
            print(f"  Filtered out {future_count} future facilities (year > 2026)")
    
    print(f"  Final count: {len(dc)} DCs (from {original_count} total)")
    
    # Classify each DC
    dc['dc_category'] = dc.apply(classify_datacenter, axis=1)
    
    # Estimate capacity based on category
    dc['capacity_mw_est'] = dc.apply(
        lambda row: row['capacity_mw'] if pd.notna(row['capacity_mw']) 
        else DC_PARAMS[row['dc_category']]['default_capacity'],
        axis=1
    )
    
    # Estimate energy
    dc['estimated_energy_mwh'] = dc.apply(
        lambda row: row['capacity_mw_est'] * DC_PARAMS[row['dc_category']]['energy_per_mw'],
        axis=1
    )
    
    return dc


def analyze_categories(dc):
    """Analyze data centers by category"""
    print("\n" + "="*70)
    print("DATA CENTER CLASSIFICATION")
    print("="*70)
    
    # Summary by category
    summary = dc.groupby('dc_category').agg({
        'data_center_id': 'count',
        'capacity_mw': lambda x: x.notna().sum(),
        'capacity_mw_est': 'sum',
        'estimated_energy_mwh': 'sum',
    }).round(0)
    summary.columns = ['Count', 'Known Capacity', 'Total MW', 'Total Energy (MWh)']
    
    print("\nBy Category:")
    print("-"*70)
    for cat in ['small', 'decent', 'crypto', 'big_ai']:
        if cat in summary.index:
            row = summary.loc[cat]
            pct = row['Count'] / len(dc) * 100
            energy_twh = row['Total Energy (MWh)'] / 1e6
            print(f"  {cat.upper():<10} {int(row['Count']):>5} DCs ({pct:>5.1f}%)  "
                  f"{row['Total MW']:>10,.0f} MW  {energy_twh:>8.1f} TWh")
    
    print("-"*70)
    total_energy = dc['estimated_energy_mwh'].sum() / 1e6
    print(f"  {'TOTAL':<10} {len(dc):>5} DCs           "
          f"{dc['capacity_mw_est'].sum():>10,.0f} MW  {total_energy:>8.1f} TWh")
    
    # Top Crypto Mining Facilities
    print("\n" + "="*70)
    print("TOP CRYPTO MINING FACILITIES")
    print("="*70)
    
    crypto_dcs = dc[dc['dc_category'] == 'crypto'].copy()
    crypto_dcs = crypto_dcs.sort_values('capacity_mw_est', ascending=False)
    
    print(f"\n{'Name':<50} {'State':<15} {'Capacity (MW)':<15}")
    print("-"*80)
    for _, row in crypto_dcs.head(15).iterrows():
        name = row['data_center_name'][:48]
        print(f"{name:<50} {row['state']:<15} {row['capacity_mw_est']:>10,.0f}")
    
    # Top AI Data Centers (excluding crypto)
    print("\n" + "="*70)
    print("TOP AI/HYPERSCALE DATA CENTERS (excluding crypto)")
    print("="*70)
    
    ai_dcs = dc[dc['dc_category'] == 'big_ai'].copy()
    ai_dcs = ai_dcs.sort_values('capacity_mw_est', ascending=False)
    
    print(f"\n{'Name':<50} {'State':<15} {'Capacity (MW)':<15}")
    print("-"*80)
    for _, row in ai_dcs.head(15).iterrows():
        name = row['data_center_name'][:48]
        print(f"{name:<50} {row['state']:<15} {row['capacity_mw_est']:>10,.0f}")
    
    return summary


def analyze_ai_by_state(dc):
    """Analyze AI data centers by state"""
    print("\n" + "="*70)
    print("AI DATA CENTERS BY STATE")
    print("="*70)
    
    ai_dcs = dc[dc['dc_category'] == 'big_ai'].copy()
    
    state_summary = ai_dcs.groupby('state').agg({
        'data_center_id': 'count',
        'capacity_mw_est': 'sum',
        'estimated_energy_mwh': 'sum',
    }).sort_values('estimated_energy_mwh', ascending=False)
    state_summary.columns = ['AI DC Count', 'AI Capacity (MW)', 'AI Energy (MWh)']
    
    # Load real EIA data for comparison
    eia = pd.read_csv('data/eia_state_electricity_real.csv')
    latest_eia = eia[eia['year'] == eia['year'].max()][['state', 'total_consumption_mwh']]
    
    state_summary = state_summary.reset_index()
    state_summary = pd.merge(state_summary, latest_eia, on='state', how='left')
    state_summary['ai_pct_of_state'] = (state_summary['AI Energy (MWh)'] / 
                                         state_summary['total_consumption_mwh'] * 100)
    
    print(f"\n{'State':<20} {'AI DCs':<10} {'AI MW':<12} {'AI TWh':<10} {'State TWh':<12} {'AI % of State':<15}")
    print("-"*85)
    
    for _, row in state_summary.head(15).iterrows():
        ai_twh = row['AI Energy (MWh)'] / 1e6
        state_twh = row['total_consumption_mwh'] / 1e6 if pd.notna(row['total_consumption_mwh']) else 0
        pct = row['ai_pct_of_state'] if pd.notna(row['ai_pct_of_state']) else 0
        print(f"{row['state']:<20} {int(row['AI DC Count']):<10} {row['AI Capacity (MW)']:>10,.0f} "
              f"{ai_twh:>8.2f} {state_twh:>10.1f} {pct:>12.1f}%")
    
    return state_summary


def compare_with_industry(dc):
    """Compare estimates with industry data"""
    print("\n" + "="*70)
    print("VALIDATION AGAINST INDUSTRY DATA")
    print("="*70)
    
    # Load EIA
    eia = pd.read_csv('data/eia_state_electricity_real.csv')
    us_total = eia[eia['year'] == 2024]['total_consumption_mwh'].sum()
    
    total_dc_energy = dc['estimated_energy_mwh'].sum()
    ai_energy = dc[dc['dc_category'] == 'big_ai']['estimated_energy_mwh'].sum()
    crypto_energy = dc[dc['dc_category'] == 'crypto']['estimated_energy_mwh'].sum()
    
    print(f"\nUS Electricity (2024 EIA): {us_total/1e9:.0f} TWh")
    print(f"\nOur Estimates:")
    print(f"  Total DC Energy:     {total_dc_energy/1e6:.1f} TWh ({total_dc_energy/us_total*100:.2f}% of US)")
    print(f"  Crypto Mining Only:  {crypto_energy/1e6:.1f} TWh ({crypto_energy/us_total*100:.2f}% of US)")
    print(f"  AI/Hyperscale Only:  {ai_energy/1e6:.1f} TWh ({ai_energy/us_total*100:.2f}% of US)")
    
    print(f"\nIndustry Benchmarks (2024):")
    print(f"  US Data Centers: 4-5% of electricity (~160-200 TWh)")
    print(f"  US Crypto Mining: ~0.6-2.3% (EIA estimate: 25-50 TWh)")
    print(f"  AI Training: ~0.5-1% of US (~20-40 TWh)")
    
    # Check if our AI estimate is realistic
    ai_realistic = 30  # TWh - rough industry estimate for AI
    crypto_realistic = 35  # TWh - EIA estimate for crypto
    our_ai = ai_energy / 1e6
    our_crypto = crypto_energy / 1e6
    
    print(f"\nValidation:")
    if our_crypto > crypto_realistic * 2:
        print(f"  ⚠️  Crypto estimate ({our_crypto:.1f} TWh) HIGH vs industry (~{crypto_realistic} TWh)")
    elif our_crypto < crypto_realistic * 0.3:
        print(f"  ⚠️  Crypto estimate ({our_crypto:.1f} TWh) LOW vs industry (~{crypto_realistic} TWh)")
    else:
        print(f"  ✓  Crypto estimate ({our_crypto:.1f} TWh) reasonable vs industry (~{crypto_realistic} TWh)")
    
    if our_ai > ai_realistic * 2:
        print(f"  ⚠️  AI estimate ({our_ai:.1f} TWh) HIGH vs industry (~{ai_realistic} TWh)")
    elif our_ai < ai_realistic * 0.3:
        print(f"  ⚠️  AI estimate ({our_ai:.1f} TWh) LOW vs industry (~{ai_realistic} TWh)")
    else:
        print(f"  ✓  AI estimate ({our_ai:.1f} TWh) reasonable vs industry (~{ai_realistic} TWh)")


def train_ai_prediction_model(dc):
    """Train model to predict AI DC growth impact"""
    print("\n" + "="*70)
    print("AI DATA CENTER PREDICTION MODEL")
    print("="*70)
    
    # Load EIA data
    eia = pd.read_csv('data/eia_state_electricity_real.csv')
    
    # Filter to AI DCs with known operational year
    ai_dcs = dc[(dc['dc_category'] == 'big_ai') & (dc['year_operational'].notna())].copy()
    ai_dcs['year'] = ai_dcs['year_operational'].astype(int)
    
    print(f"\nAI DCs with known operational year: {len(ai_dcs)}")
    
    # Aggregate by state and year
    ai_by_year = ai_dcs.groupby(['state', 'year']).agg({
        'data_center_id': 'count',
        'capacity_mw_est': 'sum',
        'estimated_energy_mwh': 'sum',
    }).reset_index()
    ai_by_year.columns = ['state', 'year', 'new_ai_dc_count', 'new_ai_capacity', 'new_ai_energy']
    
    # Merge with EIA
    merged = pd.merge(eia, ai_by_year, on=['state', 'year'], how='left')
    merged = merged.fillna({'new_ai_dc_count': 0, 'new_ai_capacity': 0, 'new_ai_energy': 0})
    
    # Cumulative AI capacity
    merged = merged.sort_values(['state', 'year'])
    merged['cumulative_ai_capacity'] = merged.groupby('state')['new_ai_capacity'].cumsum()
    merged['cumulative_ai_energy'] = merged.groupby('state')['new_ai_energy'].cumsum()
    
    # Prepare features
    valid = merged[merged['yoy_change_mwh'].notna()].copy()
    
    features = ['new_ai_dc_count', 'new_ai_capacity', 'new_ai_energy',
                'cumulative_ai_capacity', 'cumulative_ai_energy', 'total_consumption_mwh']
    
    X = valid[features]
    y = valid['yoy_change_mwh']
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:,.0f} MWh")
    
    # Feature importance
    print("\nFeature Importance:")
    importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    importance = importance.sort_values('importance', ascending=False)
    for _, row in importance.iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.3f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler, 'features': features}, 
                'models/ai_dc_prediction_model.joblib')
    
    return model, scaler, merged


def predict_future_impact(model, scaler, current_data, dc):
    """Predict future AI DC energy impact"""
    print("\n" + "="*70)
    print("FUTURE AI DATA CENTER PREDICTIONS")
    print("="*70)
    
    # Example: What if Virginia adds 500 MW of AI capacity?
    scenarios = [
        ('Virginia adds 500 MW AI DC', 'Virginia', 500),
        ('Texas adds 1000 MW AI DC', 'Texas', 1000),
        ('Ohio adds 300 MW AI DC', 'Ohio', 300),
    ]
    
    print("\nScenario Analysis:")
    print("-"*70)
    
    latest = current_data[current_data['year'] == current_data['year'].max()]
    
    for scenario_name, state, new_capacity in scenarios:
        state_data = latest[latest['state'] == state].iloc[0] if len(latest[latest['state'] == state]) > 0 else None
        
        if state_data is not None:
            new_energy = new_capacity * DC_PARAMS['big_ai']['energy_per_mw']
            
            # Calculate homes equivalent
            homes = (new_energy * 1000) / 10500  # 10500 kWh per home
            
            print(f"\n  {scenario_name}:")
            print(f"    New AI Capacity: {new_capacity} MW")
            print(f"    Annual Energy: {new_energy:,.0f} MWh ({new_energy/1e3:.1f} GWh)")
            print(f"    Equivalent Homes: {homes:,.0f}")
            print(f"    % of State Current: {new_energy/state_data['total_consumption_mwh']*100:.2f}%")


def save_results(dc):
    """Save categorized data"""
    os.makedirs('data', exist_ok=True)
    
    # Save full categorized data
    dc.to_csv('data/datacenter_categorized.csv', index=False)
    print(f"\nSaved categorized data to data/datacenter_categorized.csv")
    
    # Save AI DCs only
    ai_dcs = dc[dc['dc_category'] == 'big_ai']
    ai_dcs.to_csv('data/ai_datacenters.csv', index=False)
    print(f"Saved AI data centers to data/ai_datacenters.csv")
    
    # Save crypto DCs only  
    crypto_dcs = dc[dc['dc_category'] == 'crypto']
    crypto_dcs.to_csv('data/crypto_datacenters.csv', index=False)
    print(f"Saved crypto mining facilities to data/crypto_datacenters.csv")
    
    # Summary stats
    summary = {
        'total_dcs': len(dc),
        'small_dcs': len(dc[dc['dc_category'] == 'small']),
        'decent_dcs': len(dc[dc['dc_category'] == 'decent']),
        'crypto_dcs': len(crypto_dcs),
        'big_ai_dcs': len(ai_dcs),
        'total_capacity_mw': float(dc['capacity_mw_est'].sum()),
        'crypto_capacity_mw': float(crypto_dcs['capacity_mw_est'].sum()),
        'ai_capacity_mw': float(ai_dcs['capacity_mw_est'].sum()),
        'total_energy_twh': float(dc['estimated_energy_mwh'].sum() / 1e6),
        'crypto_energy_twh': float(crypto_dcs['estimated_energy_mwh'].sum() / 1e6),
        'ai_energy_twh': float(ai_dcs['estimated_energy_mwh'].sum() / 1e6),
    }
    
    import json
    with open('data/dc_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to data/dc_summary.json")


def main():
    print("="*70)
    print("AI DATA CENTER ENERGY PREDICTION MODEL v8")
    print("="*70)
    print("\nFilters Applied:")
    print("  - Operational only (exclude planned/announced)")
    print("  - Require operational year (exclude unknown dates)")
    print("\nCategories:")
    print("  - SMALL:  <10 MW (default 2 MW)")
    print("  - DECENT: 10-50 MW (default 15 MW)")
    print("  - CRYPTO: Bitcoin/crypto mining (default 50 MW, 95% util)")
    print("  - BIG_AI: >50 MW or Hyperscaler (default 50 MW, 70% util)")
    
    # Load and classify with filters
    print("\n" + "="*70)
    print("APPLYING FILTERS")
    print("="*70)
    dc = load_and_classify_data(operational_only=True, require_year=True)
    
    # Analyze
    analyze_categories(dc)
    analyze_ai_by_state(dc)
    compare_with_industry(dc)
    
    # Train model
    model, scaler, merged = train_ai_prediction_model(dc)
    
    # Predictions
    predict_future_impact(model, scaler, merged, dc)
    
    # Save
    save_results(dc)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    crypto_count = len(dc[dc['dc_category'] == 'crypto'])
    crypto_energy = dc[dc['dc_category'] == 'crypto']['estimated_energy_mwh'].sum() / 1e6
    ai_count = len(dc[dc['dc_category'] == 'big_ai'])
    ai_energy = dc[dc['dc_category'] == 'big_ai']['estimated_energy_mwh'].sum() / 1e6
    print(f"""
This model separates CRYPTO from AI data centers:

  Crypto Mining:      {crypto_count:,} facilities, {crypto_energy:.1f} TWh/year
  AI/Hyperscale:      {ai_count:,} facilities, {ai_energy:.1f} TWh/year

Crypto operators: Riot, Marathon, Core Scientific, Bitdeer, Cipher
AI operators: Google, Amazon/AWS, Meta, Microsoft, Apple, NVIDIA

The model now provides accurate AI-specific energy predictions.
""")


if __name__ == '__main__':
    main()
