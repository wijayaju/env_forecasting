#!/usr/bin/env python3
"""
BA Multi-Year Electricity Predictor
Enhanced model using 2019-2024 EIA-861 data for panel analysis.

Key Improvement: 6 years of data gives us ~300+ observations (54 BAs × 6 years)
instead of just 54. This enables:
- Time-series analysis of DC impact growth
- More robust feature importance estimates
- Year-over-year trend detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
EIA_DIR = DATA_DIR / "eia"
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Years to process
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

# State -> BA mapping
STATE_TO_BA = {
    'VA': 'PJM', 'MD': 'PJM', 'DE': 'PJM', 'NJ': 'PJM', 'PA': 'PJM',
    'DC': 'PJM', 'WV': 'PJM', 'OH': 'PJM',
    'TX': 'ERCO', 'CA': 'CISO',
    'IL': 'MISO', 'IN': 'MISO', 'IA': 'MISO', 'MI': 'MISO', 'MN': 'MISO',
    'MO': 'MISO', 'WI': 'MISO', 'ND': 'MISO', 'SD': 'MISO', 'MT': 'MISO',
    'LA': 'MISO', 'MS': 'MISO', 'AR': 'MISO',
    'NY': 'NYIS',
    'CT': 'ISNE', 'MA': 'ISNE', 'ME': 'ISNE', 'NH': 'ISNE', 'RI': 'ISNE', 'VT': 'ISNE',
    'GA': 'SOCO', 'AL': 'SOCO',
    'FL': 'FPL',
    'NC': 'DUK', 'SC': 'DUK',
    'KS': 'SWPP', 'NE': 'SWPP', 'OK': 'SWPP', 'NM': 'SWPP',
    'TN': 'TVA', 'KY': 'TVA',
    'WA': 'BPAT', 'OR': 'BPAT', 'ID': 'BPAT',
    'NV': 'NEVP', 'AZ': 'APS', 'UT': 'PACE', 'WY': 'PACE', 'CO': 'PSCO',
    'HI': 'HECO', 'AK': 'OTHER'
}

STATE_ABBREV = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
    'District of Columbia': 'DC'
}


def load_year_data(year):
    """Load EIA-861 data for a specific year."""
    sales_file = EIA_DIR / str(year) / f"Sales_Ult_Cust_{year}.xlsx"
    
    if not sales_file.exists():
        print(f"   ⚠️ Missing data for {year}")
        return None
    
    df = pd.read_excel(sales_file, sheet_name='States', header=1)
    n_cols = len(df.columns)
    
    # Column mapping varies by year - use positional mapping
    # Key columns we need: year (0), state (6), ba_code (8), sales data (near end)
    col_mapping = {}
    
    # First columns are consistent
    col_mapping[0] = 'year'
    col_mapping[1] = 'utility_id'
    col_mapping[2] = 'utility_name'
    col_mapping[6] = 'state'
    col_mapping[8] = 'ba_code'
    
    # Last columns contain totals (work backwards)
    col_mapping[n_cols - 1] = 'total_customers'
    col_mapping[n_cols - 2] = 'total_sales_mwh'
    col_mapping[n_cols - 3] = 'total_revenue'
    
    # Mid columns for sector data
    col_mapping[9] = 'residential_revenue'
    col_mapping[10] = 'residential_sales_mwh'
    col_mapping[11] = 'residential_customers'
    col_mapping[12] = 'commercial_revenue'
    col_mapping[13] = 'commercial_sales_mwh'
    col_mapping[14] = 'commercial_customers'
    col_mapping[15] = 'industrial_revenue'
    col_mapping[16] = 'industrial_sales_mwh'
    col_mapping[17] = 'industrial_customers'
    
    # Rename only mapped columns
    new_names = {}
    for idx, name in col_mapping.items():
        if idx < n_cols:
            new_names[df.columns[idx]] = name
    
    df = df.rename(columns=new_names)
    
    # Filter valid data
    df = df[pd.to_numeric(df.get('year', pd.Series()), errors='coerce').notna()]
    df['year'] = year  # Ensure year is set correctly
    
    # Convert numeric columns
    for col in ['total_sales_mwh', 'industrial_sales_mwh', 'commercial_sales_mwh', 
                'residential_sales_mwh', 'total_customers']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def load_all_years():
    """Load and combine EIA data from all years."""
    print("\n📊 Loading Multi-Year EIA-861 Data...")
    
    all_data = []
    for year in YEARS:
        df = load_year_data(year)
        if df is not None:
            all_data.append(df)
            print(f"   {year}: {len(df):,} utility-state records")
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n   Total: {len(combined):,} records across {len(YEARS)} years")
    
    return combined


def aggregate_by_ba_year(df):
    """Aggregate utility data by BA and year."""
    print("\n📈 Aggregating by Balancing Authority and Year...")
    
    ba_year = df.groupby(['ba_code', 'year']).agg({
        'total_sales_mwh': 'sum',
        'industrial_sales_mwh': 'sum',
        'commercial_sales_mwh': 'sum',
        'residential_sales_mwh': 'sum',
        'utility_id': 'nunique',
        'total_customers': 'sum'
    }).reset_index()
    
    ba_year.columns = ['ba_code', 'year', 'total_mwh', 'industrial_mwh', 
                       'commercial_mwh', 'residential_mwh', 'utility_count', 'customers']
    
    # Filter out empty/invalid BAs
    ba_year = ba_year[ba_year['ba_code'].notna() & (ba_year['ba_code'] != '')]
    ba_year = ba_year[ba_year['total_mwh'] > 1e6]  # > 1 TWh
    
    print(f"   Created {len(ba_year)} BA-year observations")
    print(f"   Unique BAs: {ba_year['ba_code'].nunique()}")
    print(f"   Years covered: {sorted(ba_year['year'].unique())}")
    
    # Calculate derived features
    ba_year['ci_ratio'] = (ba_year['industrial_mwh'] + ba_year['commercial_mwh']) / ba_year['total_mwh']
    ba_year['ci_ratio'] = ba_year['ci_ratio'].fillna(0).clip(0, 1)
    
    return ba_year


def load_dc_data():
    """Load data center information."""
    print("\n🏢 Loading Data Center Data...")
    
    json_file = BASE_DIR / "website" / "datacenters.json"
    with open(json_file) as f:
        dcs = json.load(f)
    
    df = pd.DataFrame(dcs)
    
    # Map to BA
    def get_abbrev(state):
        if pd.isna(state):
            return None
        state = str(state).strip()
        if len(state) == 2:
            return state.upper()
        return STATE_ABBREV.get(state, None)
    
    df['state_abbrev'] = df['state'].apply(get_abbrev)
    df['ba_code'] = df['state_abbrev'].map(STATE_TO_BA).fillna('OTHER')
    
    # Clean numeric fields
    df['capacity_mw'] = pd.to_numeric(df['capacity_mw'], errors='coerce')
    df['energy_mwh'] = pd.to_numeric(df['energy_mwh'], errors='coerce')
    df['year_built'] = pd.to_numeric(df.get('year', df.get('year_built', None)), errors='coerce')
    
    print(f"   Loaded {len(df)} data centers")
    print(f"   With capacity: {df['capacity_mw'].notna().sum()}")
    print(f"   With year: {df['year_built'].notna().sum()}")
    
    return df


def create_dc_features_by_ba_year(dc_df, years):
    """Create DC features by BA and year (cumulative up to each year)."""
    print("\n📈 Creating DC Features by BA-Year...")
    
    records = []
    
    for ba in dc_df['ba_code'].unique():
        ba_dcs = dc_df[dc_df['ba_code'] == ba]
        
        for year in years:
            # For DCs with year data, count cumulative DCs up to this year
            # For DCs without year, assume they exist in all years
            dcs_with_year = ba_dcs[ba_dcs['year_built'].notna()]
            dcs_without_year = ba_dcs[ba_dcs['year_built'].isna()]
            
            # Cumulative DCs built by this year
            cumulative = dcs_with_year[dcs_with_year['year_built'] <= year]
            
            # Total = cumulative + unknown (assume existing)
            total_dcs = pd.concat([cumulative, dcs_without_year])
            
            if len(total_dcs) == 0:
                continue
            
            record = {
                'ba_code': ba,
                'year': year,
                'dc_count': len(total_dcs),
                'total_capacity_mw': total_dcs['capacity_mw'].sum(),
                'avg_capacity_mw': total_dcs['capacity_mw'].mean(),
                'max_capacity_mw': total_dcs['capacity_mw'].max(),
                'crypto_count': (total_dcs['category'] == 'crypto').sum(),
                'big_ai_count': (total_dcs['category'] == 'big_ai').sum(),
                'hyperscale_count': ((total_dcs['category'] == 'big_ai') | 
                                     (total_dcs['capacity_mw'] >= 50)).sum(),
                'total_energy_mwh': total_dcs['energy_mwh'].sum()
            }
            records.append(record)
    
    features = pd.DataFrame(records)
    features = features.fillna(0)
    
    # Calculate ratios
    features['crypto_ratio'] = features['crypto_count'] / features['dc_count'].clip(lower=1)
    features['hyperscale_ratio'] = features['hyperscale_count'] / features['dc_count'].clip(lower=1)
    
    print(f"   Created {len(features)} BA-year DC feature records")
    
    return features


def merge_datasets(ba_elec, ba_dc):
    """Merge electricity and DC data."""
    print("\n🔗 Merging Datasets...")
    
    merged = ba_elec.merge(ba_dc, on=['ba_code', 'year'], how='left')
    merged = merged.fillna(0)
    
    # Calculate DC share
    merged['dc_energy_share'] = merged['total_energy_mwh'] / merged['total_mwh']
    merged['dc_energy_share'] = merged['dc_energy_share'].clip(0, 1)
    
    # Log transforms
    merged['log_total_mwh'] = np.log1p(merged['total_mwh'])
    merged['log_capacity'] = np.log1p(merged['total_capacity_mw'])
    
    # Year as numeric (relative to 2019)
    merged['year_rel'] = merged['year'] - 2019
    
    print(f"\n   Final dataset: {len(merged)} observations")
    print(f"   BAs: {merged['ba_code'].nunique()}")
    print(f"   Years: {sorted(merged['year'].unique())}")
    
    return merged


def train_multiyear_model(data):
    """Train ML models on multi-year panel data."""
    print("\n🤖 Training Multi-Year BA-Level Models...")
    
    # Feature columns
    feature_cols = [
        'dc_count', 'total_capacity_mw', 'avg_capacity_mw', 'max_capacity_mw',
        'crypto_count', 'big_ai_count', 'hyperscale_count',
        'crypto_ratio', 'hyperscale_ratio',
        'utility_count', 'ci_ratio',
        'year_rel'  # Time trend
    ]
    
    feature_cols = [c for c in feature_cols if c in data.columns]
    target_col = 'total_mwh'
    
    X = data[feature_cols].values
    y = data[target_col].values
    
    print(f"\n   Features ({len(feature_cols)}): {feature_cols}")
    print(f"   Observations: {len(X)}")
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split (using most recent year as test)
    test_year = data['year'].max()
    train_mask = data['year'] < test_year
    test_mask = data['year'] == test_year
    
    X_train, y_train = X_scaled[train_mask], y[train_mask]
    X_test, y_test = X_scaled[test_mask], y[test_mask]
    
    print(f"\n   Train: {len(X_train)} obs (years < {test_year})")
    print(f"   Test: {len(X_test)} obs (year = {test_year})")
    
    # Models
    models = {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            min_child_weight=5, random_state=42,
            objective='reg:squarederror'
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=150, max_depth=6, min_samples_split=5, random_state=42
        ),
        'Neural Network': MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=1000, 
            early_stopping=True, random_state=42
        )
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Time series CV
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2')
        
        results[name] = {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'model': model
        }
        
        print(f"\n   {name}:")
        print(f"   Test R² = {r2:.4f}")
        print(f"   TS-CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"   MAE = {mae/1e6:.1f} TWh, RMSE = {rmse/1e6:.1f} TWh")
    
    # Feature importance (XGBoost)
    best_model = results['XGBoost']['model']
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n   Feature Importance (XGBoost):")
    dc_importance = 0
    for _, row in importance.iterrows():
        bar = '█' * int(row['importance'] * 50)
        is_dc = 'DC' if any(x in row['feature'] for x in ['dc_', 'capacity', 'crypto', 'hyperscale', 'big_ai', 'energy']) else ''
        if is_dc:
            dc_importance += row['importance']
        print(f"   {row['feature']:25s} {row['importance']:.4f} {bar} {is_dc}")
    
    print(f"\n   Total DC Feature Contribution: {dc_importance:.1%}")
    
    return results, importance


def analyze_temporal_trends(data):
    """Analyze year-over-year trends in DC impact."""
    print("\n📈 Temporal Trend Analysis...")
    
    # Yearly aggregates
    yearly = data.groupby('year').agg({
        'total_mwh': 'sum',
        'dc_count': 'sum',
        'total_capacity_mw': 'sum',
        'total_energy_mwh': 'sum',
        'ba_code': 'nunique'
    }).reset_index()
    
    yearly['dc_share'] = yearly['total_energy_mwh'] / yearly['total_mwh'] * 100
    
    print("\n   Year-over-Year Summary:")
    print("   Year    Electricity     DC Count    DC Capacity    DC Share")
    print("   " + "-" * 60)
    
    for _, row in yearly.iterrows():
        elec_twh = row['total_mwh'] / 1e6
        print(f"   {int(row['year'])}    {elec_twh:8.1f} TWh    {int(row['dc_count']):6,}      {row['total_capacity_mw']:8,.0f} MW    {row['dc_share']:.2f}%")
    
    # Growth rates
    if len(yearly) > 1:
        first = yearly.iloc[0]
        last = yearly.iloc[-1]
        years_span = last['year'] - first['year']
        
        elec_cagr = ((last['total_mwh'] / first['total_mwh']) ** (1/years_span) - 1) * 100
        dc_count_cagr = ((last['dc_count'] / max(first['dc_count'], 1)) ** (1/years_span) - 1) * 100
        cap_cagr = ((last['total_capacity_mw'] / max(first['total_capacity_mw'], 1)) ** (1/years_span) - 1) * 100
        
        print(f"\n   CAGR ({int(first['year'])}-{int(last['year'])}):")
        print(f"   - Electricity: {elec_cagr:+.1f}%/year")
        print(f"   - DC Count: {dc_count_cagr:+.1f}%/year")
        print(f"   - DC Capacity: {cap_cagr:+.1f}%/year")
    
    return yearly


def analyze_top_bas(data):
    """Analyze top BAs by DC concentration over time."""
    print("\n🏆 Top BAs by DC Concentration (2024):")
    
    latest = data[data['year'] == data['year'].max()]
    top = latest.nlargest(10, 'dc_energy_share')
    
    for _, row in top.iterrows():
        share = row['dc_energy_share'] * 100
        signal = "🔥" if share > 5 else "⚡" if share > 2 else "📊"
        print(f"   {signal} {row['ba_code']:8s}: {share:5.2f}% DC share ({int(row['dc_count']):4} DCs, {row['total_capacity_mw']:6,.0f} MW)")


def save_results(data, results, importance, yearly):
    """Save comprehensive results."""
    print("\n💾 Saving Results...")
    
    best_result = results['XGBoost']
    
    output = {
        'model_type': 'BA Multi-Year Panel Model',
        'data_source': 'EIA-861 2019-2024',
        'years': list(map(int, sorted(data['year'].unique()))),
        'n_observations': len(data),
        'n_balancing_authorities': int(data['ba_code'].nunique()),
        'models': {},
        'feature_importance': importance.to_dict('records'),
        'dc_feature_contribution': float(importance[
            importance['feature'].str.contains('dc_|capacity|crypto|hyperscale|big_ai|energy', na=False)
        ]['importance'].sum()),
        'yearly_trends': yearly.to_dict('records'),
        'top_bas_2024': []
    }
    
    for name, res in results.items():
        output['models'][name] = {
            'test_r2': round(res['r2'], 4),
            'cv_r2': round(res['cv_r2_mean'], 4),
            'cv_r2_std': round(res['cv_r2_std'], 4),
            'mae_twh': round(res['mae'] / 1e6, 2),
            'rmse_twh': round(res['rmse'] / 1e6, 2)
        }
    
    # Top BAs
    latest = data[data['year'] == data['year'].max()]
    for _, row in latest.nlargest(15, 'dc_energy_share').iterrows():
        output['top_bas_2024'].append({
            'ba_code': row['ba_code'],
            'dc_count': int(row['dc_count']),
            'capacity_mw': round(row['total_capacity_mw'], 1),
            'electricity_twh': round(row['total_mwh'] / 1e6, 2),
            'dc_share_pct': round(row['dc_energy_share'] * 100, 2)
        })
    
    output_file = OUTPUT_DIR / "ba_multiyear_model_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"   Saved to {output_file}")
    
    return output


def main():
    """Main execution."""
    print("=" * 70)
    print("⚡ BA MULTI-YEAR ELECTRICITY PREDICTOR")
    print("   Panel Analysis with 2019-2024 EIA-861 Data")
    print("=" * 70)
    
    # 1. Load multi-year EIA data
    eia_df = load_all_years()
    
    # 2. Aggregate by BA-year
    ba_elec = aggregate_by_ba_year(eia_df)
    
    # 3. Load DC data
    dc_df = load_dc_data()
    
    # 4. Create DC features by BA-year
    ba_dc = create_dc_features_by_ba_year(dc_df, YEARS)
    
    # 5. Merge datasets
    combined = merge_datasets(ba_elec, ba_dc)
    
    # 6. Train models
    results, importance = train_multiyear_model(combined)
    
    # 7. Analyze temporal trends
    yearly = analyze_temporal_trends(combined)
    
    # 8. Analyze top BAs
    analyze_top_bas(combined)
    
    # 9. Save results
    output = save_results(combined, results, importance, yearly)
    
    # Summary
    best = results['XGBoost']
    dc_contrib = output['dc_feature_contribution']
    
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    print(f"""
   Data: {len(combined)} observations ({ba_elec['ba_code'].nunique()} BAs × {len(YEARS)} years)
   
   Model Performance (XGBoost):
   - Test R² (holdout 2024): {best['r2']:.4f}
   - Time-Series CV R²: {best['cv_r2_mean']:.4f} ± {best['cv_r2_std']:.4f}
   - MAE: {best['mae']/1e6:.1f} TWh
   
   DC Feature Contribution: {dc_contrib:.1%} of predictive signal
   
   Key Features:
   1. {importance.iloc[0]['feature']}: {importance.iloc[0]['importance']:.1%}
   2. {importance.iloc[1]['feature']}: {importance.iloc[1]['importance']:.1%}
   3. {importance.iloc[2]['feature']}: {importance.iloc[2]['importance']:.1%}
   
   Improvement over state-level: {dc_contrib*100/3:.0f}× stronger DC signal!
""")


if __name__ == "__main__":
    main()
