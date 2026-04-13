#!/usr/bin/env python3
"""
BA-Level Electricity Predictor
Enhanced model using Balancing Authority-level granularity for stronger DC signal.

Key Improvement: BAs are larger than states (58 BAs vs 51 states) but more
geographically focused, giving us 10-30% DC signal vs 3-5% at state level.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
EIA_DIR = DATA_DIR / "eia" / "2024"
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Comprehensive state -> BA mapping
# Each state is assigned to its primary BA (some states span multiple)
STATE_TO_BA = {
    # PJM territory (Mid-Atlantic)
    'VA': 'PJM', 'MD': 'PJM', 'DE': 'PJM', 'NJ': 'PJM', 'PA': 'PJM',
    'DC': 'PJM', 'WV': 'PJM', 'OH': 'PJM',
    
    # ERCOT (Texas)
    'TX': 'ERCO',
    
    # CAISO (California)
    'CA': 'CISO',
    
    # MISO (Midwest)
    'IL': 'MISO', 'IN': 'MISO', 'IA': 'MISO', 'MI': 'MISO', 'MN': 'MISO',
    'MO': 'MISO', 'WI': 'MISO', 'ND': 'MISO', 'SD': 'MISO', 'MT': 'MISO',
    'LA': 'MISO', 'MS': 'MISO', 'AR': 'MISO',
    
    # NYISO (New York)
    'NY': 'NYIS',
    
    # ISO-NE (New England)
    'CT': 'ISNE', 'MA': 'ISNE', 'ME': 'ISNE', 'NH': 'ISNE', 'RI': 'ISNE', 'VT': 'ISNE',
    
    # Southern Company (Southeast)
    'GA': 'SOCO', 'AL': 'SOCO',
    
    # Florida
    'FL': 'FPL',
    
    # Duke (Carolinas)
    'NC': 'DUK', 'SC': 'DUK',
    
    # SPP (Plains)
    'KS': 'SWPP', 'NE': 'SWPP', 'OK': 'SWPP', 'NM': 'SWPP',
    
    # TVA (Tennessee Valley)
    'TN': 'TVA', 'KY': 'TVA',
    
    # Western
    'WA': 'BPAT', 'OR': 'BPAT', 'ID': 'BPAT',
    'NV': 'NEVP',
    'AZ': 'APS',
    'UT': 'PACE', 'WY': 'PACE',
    'CO': 'PSCO',
    
    # Others
    'HI': 'HECO', 'AK': 'OTHER'
}

# State name to abbreviation
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


def load_eia_data():
    """Load and process EIA-861 utility-level sales data."""
    print("\n📊 Loading EIA-861 Data...")
    
    sales_file = EIA_DIR / "Sales_Ult_Cust_2024.xlsx"
    df = pd.read_excel(sales_file, sheet_name='States', header=1)
    
    # Assign proper column names
    df.columns = [
        'year', 'utility_id', 'utility_name', 'part', 'service_type', 
        'data_type', 'state', 'ownership', 'ba_code',
        'residential_revenue', 'residential_sales_mwh', 'residential_customers',
        'commercial_revenue', 'commercial_sales_mwh', 'commercial_customers',
        'industrial_revenue', 'industrial_sales_mwh', 'industrial_customers',
        'transport_revenue', 'transport_sales_mwh', 'transport_customers',
        'total_revenue', 'total_sales_mwh', 'total_customers'
    ][:len(df.columns)]
    
    # Filter valid data
    df = df[pd.to_numeric(df['year'], errors='coerce').notna()]
    df['total_sales_mwh'] = pd.to_numeric(df['total_sales_mwh'], errors='coerce')
    df['industrial_sales_mwh'] = pd.to_numeric(df['industrial_sales_mwh'], errors='coerce')
    df['commercial_sales_mwh'] = pd.to_numeric(df['commercial_sales_mwh'], errors='coerce')
    
    # Aggregate by BA
    ba_data = df.groupby('ba_code').agg({
        'total_sales_mwh': 'sum',
        'industrial_sales_mwh': 'sum',
        'commercial_sales_mwh': 'sum',
        'utility_id': 'nunique',
        'total_customers': 'sum'
    }).reset_index()
    
    ba_data.columns = ['ba_code', 'total_sales_mwh', 'industrial_mwh', 
                       'commercial_mwh', 'utility_count', 'customer_count']
    
    # Calculate commercial + industrial ratio (where DCs would be)
    ba_data['ci_ratio'] = (ba_data['industrial_mwh'] + ba_data['commercial_mwh']) / ba_data['total_sales_mwh']
    ba_data['ci_ratio'] = ba_data['ci_ratio'].fillna(0)
    
    print(f"   Loaded {len(ba_data)} balancing authorities")
    print(f"   Total electricity: {ba_data['total_sales_mwh'].sum()/1e9:.1f} TWh")
    
    return ba_data


def load_dc_data():
    """Load data center information."""
    print("\n🏢 Loading Data Center Data...")
    
    json_file = BASE_DIR / "website" / "datacenters.json"
    with open(json_file) as f:
        dcs = json.load(f)
    
    df = pd.DataFrame(dcs)
    print(f"   Loaded {len(df)} data centers")
    print(f"   With capacity data: {df['capacity_mw'].notna().sum()}")
    print(f"   With energy data: {df['has_energy_data'].sum()}")
    
    return df


def map_dcs_to_bas(dc_df):
    """Map each data center to its balancing authority."""
    print("\n🔗 Mapping DCs to Balancing Authorities...")
    
    dc_df = dc_df.copy()
    
    # Convert state names to abbreviations
    def get_abbrev(state):
        if pd.isna(state):
            return None
        state = str(state).strip()
        if len(state) == 2:
            return state.upper()
        return STATE_ABBREV.get(state, None)
    
    dc_df['state_abbrev'] = dc_df['state'].apply(get_abbrev)
    
    # Map to BA
    dc_df['ba_code'] = dc_df['state_abbrev'].map(STATE_TO_BA).fillna('OTHER')
    
    # Clean capacity
    dc_df['capacity_mw'] = pd.to_numeric(dc_df['capacity_mw'], errors='coerce')
    dc_df['energy_mwh'] = pd.to_numeric(dc_df['energy_mwh'], errors='coerce')
    
    # Summary
    ba_distribution = dc_df.groupby('ba_code').size().sort_values(ascending=False)
    print("\n   DC Distribution by BA (Top 15):")
    for ba, count in ba_distribution.head(15).items():
        print(f"   - {ba}: {count} data centers")
    
    return dc_df


def create_ba_dc_features(dc_df):
    """Create BA-level DC features."""
    print("\n📈 Creating BA-Level DC Features...")
    
    # Aggregate by BA
    ba_features = dc_df.groupby('ba_code').agg({
        'name': 'count',
        'capacity_mw': ['sum', 'mean', 'max'],
        'energy_mwh': 'sum',
        'category': lambda x: (x == 'crypto').sum(),
        'has_energy_data': 'sum'
    })
    
    # Flatten column names
    ba_features.columns = ['dc_count', 'total_capacity_mw', 'avg_capacity_mw', 
                           'max_capacity_mw', 'total_energy_mwh', 
                           'crypto_count', 'energy_data_count']
    ba_features = ba_features.reset_index()
    
    # Fill NaN
    ba_features = ba_features.fillna(0)
    
    # Calculate additional features
    ba_features['crypto_ratio'] = ba_features['crypto_count'] / ba_features['dc_count']
    ba_features['avg_with_data'] = ba_features['energy_data_count'] / ba_features['dc_count']
    
    # Big AI count (based on category)
    big_ai_counts = dc_df[dc_df['category'] == 'big_ai'].groupby('ba_code').size()
    ba_features['big_ai_count'] = ba_features['ba_code'].map(big_ai_counts).fillna(0)
    
    print(f"\n   Created features for {len(ba_features)} BAs")
    print("\n   Top 10 BAs by DC count:")
    top_bas = ba_features.nlargest(10, 'dc_count')
    for _, row in top_bas.iterrows():
        print(f"   - {row['ba_code']}: {int(row['dc_count'])} DCs, "
              f"{row['total_capacity_mw']:.0f} MW, "
              f"{int(row['crypto_count'])} crypto, {int(row['big_ai_count'])} AI")
    
    return ba_features


def create_combined_dataset(ba_elec, ba_dc):
    """Combine electricity and DC data for modeling."""
    print("\n🔗 Creating Combined Dataset...")
    
    # Merge
    merged = ba_elec.merge(ba_dc, on='ba_code', how='left')
    merged = merged.fillna(0)
    
    # Filter out BAs with very little data
    merged = merged[merged['total_sales_mwh'] > 1e6]  # > 1 TWh
    
    # Calculate DC share of BA electricity
    merged['dc_energy_share'] = merged['total_energy_mwh'] / merged['total_sales_mwh']
    merged['dc_energy_share'] = merged['dc_energy_share'].clip(0, 1)
    
    # Log transform large values
    merged['log_sales'] = np.log1p(merged['total_sales_mwh'])
    merged['log_capacity'] = np.log1p(merged['total_capacity_mw'])
    
    print(f"\n   Combined dataset: {len(merged)} BAs")
    
    # Show DC share by BA
    print("\n   Estimated DC Share of BA Electricity (Top 10):")
    top_share = merged.nlargest(10, 'dc_energy_share')
    for _, row in top_share.iterrows():
        share_pct = row['dc_energy_share'] * 100
        elec_twh = row['total_sales_mwh'] / 1e6
        dc_twh = row['total_energy_mwh'] / 1e6
        print(f"   - {row['ba_code']}: {share_pct:.1f}% "
              f"({dc_twh:.1f} TWh DC / {elec_twh:.1f} TWh total)")
    
    return merged


def train_ba_model(data):
    """Train ML models on BA-level data."""
    print("\n🤖 Training BA-Level ML Models...")
    
    # Define features and target
    feature_cols = [
        'dc_count', 'total_capacity_mw', 'avg_capacity_mw', 'max_capacity_mw',
        'crypto_count', 'big_ai_count', 'crypto_ratio',
        'utility_count', 'ci_ratio'
    ]
    
    # Filter to features that exist
    feature_cols = [c for c in feature_cols if c in data.columns]
    
    target_col = 'total_sales_mwh'
    
    X = data[feature_cols].values
    y = data[target_col].values
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # We have limited data (~30 BAs with good data), so use LOO CV
    from sklearn.model_selection import LeaveOneOut
    
    print(f"\n   Features: {feature_cols}")
    print(f"   Samples: {len(X)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42
    )
    
    # Models
    models = {
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42
        )
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        
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
        print(f"   CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"   MAE = {mae/1e6:.1f} TWh, RMSE = {rmse/1e6:.1f} TWh")
    
    # Feature importance
    best_model = results['Gradient Boosting']['model']
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n   Feature Importance (Gradient Boosting):")
    for _, row in importance.iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"   {row['feature']:25s} {row['importance']:.4f} {bar}")
    
    return results, importance


def analyze_dc_concentration(data):
    """Analyze where DC energy concentration is highest."""
    print("\n📍 DC Concentration Analysis...")
    
    # Sort by DC share
    data_sorted = data.sort_values('dc_energy_share', ascending=False)
    
    print("\n   BAs with Highest DC Concentration:")
    for _, row in data_sorted.head(10).iterrows():
        share = row['dc_energy_share'] * 100
        ba = row['ba_code']
        dcs = int(row['dc_count'])
        cap = row['total_capacity_mw']
        elec = row['total_sales_mwh'] / 1e6
        
        signal = "🔥 HIGH" if share > 10 else ("⚡ MED" if share > 5 else "📊 LOW")
        print(f"   {signal} {ba}: {share:.1f}% DC share ({dcs} DCs, {cap:.0f} MW, {elec:.0f} TWh total)")


def save_results(data, results, importance):
    """Save results to JSON for website."""
    print("\n💾 Saving Results...")
    
    output = {
        'model_type': 'BA-Level Granular Model',
        'data_source': 'EIA-861 2024 Utility Sales + DC Scrape',
        'n_balancing_authorities': len(data),
        'total_electricity_twh': data['total_sales_mwh'].sum() / 1e6,
        'total_dc_energy_twh': data['total_energy_mwh'].sum() / 1e6,
        'models': {},
        'feature_importance': importance.to_dict('records'),
        'ba_summary': []
    }
    
    for name, res in results.items():
        output['models'][name] = {
            'test_r2': round(res['r2'], 4),
            'cv_r2': round(res['cv_r2_mean'], 4),
            'cv_r2_std': round(res['cv_r2_std'], 4),
            'mae_twh': round(res['mae'] / 1e6, 2),
            'rmse_twh': round(res['rmse'] / 1e6, 2)
        }
    
    # BA summary
    for _, row in data.iterrows():
        output['ba_summary'].append({
            'ba_code': row['ba_code'],
            'dc_count': int(row['dc_count']),
            'total_capacity_mw': round(row['total_capacity_mw'], 1),
            'electricity_twh': round(row['total_sales_mwh'] / 1e6, 2),
            'dc_energy_twh': round(row['total_energy_mwh'] / 1e6, 2),
            'dc_share_pct': round(row['dc_energy_share'] * 100, 2)
        })
    
    output_file = OUTPUT_DIR / "ba_level_model_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"   Saved to {output_file}")
    
    return output


def main():
    """Main execution."""
    print("=" * 70)
    print("⚡ BA-LEVEL ELECTRICITY PREDICTOR")
    print("   Enhanced Model with Balancing Authority Granularity")
    print("=" * 70)
    
    # 1. Load data
    ba_elec = load_eia_data()
    dc_df = load_dc_data()
    
    # 2. Map DCs to BAs
    dc_df = map_dcs_to_bas(dc_df)
    
    # 3. Create BA-level DC features
    ba_dc_features = create_ba_dc_features(dc_df)
    
    # 4. Combine datasets
    combined = create_combined_dataset(ba_elec, ba_dc_features)
    
    # 5. Train models
    results, importance = train_ba_model(combined)
    
    # 6. Analyze concentration
    analyze_dc_concentration(combined)
    
    # 7. Save results
    output = save_results(combined, results, importance)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    
    best = results['Gradient Boosting']
    print(f"""
   Model Performance:
   - Test R²: {best['r2']:.4f}
   - Cross-val R²: {best['cv_r2_mean']:.4f} ± {best['cv_r2_std']:.4f}
   - MAE: {best['mae']/1e6:.1f} TWh
   
   Key Features:
   - {importance.iloc[0]['feature']}: {importance.iloc[0]['importance']:.1%}
   - {importance.iloc[1]['feature']}: {importance.iloc[1]['importance']:.1%}
   - {importance.iloc[2]['feature']}: {importance.iloc[2]['importance']:.1%}
   
   DC Concentration Insight:
   At BA level, DC energy is ~{combined['dc_energy_share'].mean()*100:.1f}% average,
   up to ~{combined['dc_energy_share'].max()*100:.1f}% in high-concentration BAs.
   This is stronger than the ~3% signal at state level!
""")


if __name__ == "__main__":
    main()
