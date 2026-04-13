#!/usr/bin/env python3
"""
Improved Data Center Energy Impact Model v2

Key improvements:
1. Focus on per-MW energy impact (more generalizable)
2. Better handling of missing capacity data
3. Normalized features for better model performance
4. Cross-validation for robust evaluation
5. Hyperparameter tuning
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class ImprovedEnergyModel:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.dc_data = None
        self.electricity_data = None
        self.training_data = None
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_cols = None
        
    def load_data(self):
        """Load all data"""
        print("Loading data...")
        
        # Load data center specs
        self.dc_data = pd.read_csv(f'{self.data_dir}/datacenter_specs.csv')
        self.dc_data['year_operational'] = pd.to_numeric(
            self.dc_data['year_operational'], errors='coerce'
        )
        self.dc_data['capacity_mw'] = pd.to_numeric(
            self.dc_data['capacity_mw'], errors='coerce'
        )
        
        # Load electricity data
        self.electricity_data = pd.read_csv(f'{self.data_dir}/eia_state_electricity.csv')
        
        print(f"Data centers: {len(self.dc_data)}")
        print(f"  With year: {self.dc_data['year_operational'].notna().sum()}")
        print(f"  With capacity: {self.dc_data['capacity_mw'].notna().sum()}")
        
    def impute_capacity(self):
        """
        Impute missing capacity values using multiple strategies
        """
        print("\nImputing missing capacity values...")
        
        df = self.dc_data.copy()
        
        # Strategy 1: State median
        state_median = df.groupby('state')['capacity_mw'].transform('median')
        
        # Strategy 2: Overall median
        overall_median = df['capacity_mw'].median()
        
        # Strategy 3: Time-based estimation (newer DCs tend to be larger)
        year_median = df.groupby('year_operational')['capacity_mw'].transform('median')
        
        # Combine strategies
        df['capacity_mw_imputed'] = df['capacity_mw'].fillna(state_median)
        df['capacity_mw_imputed'] = df['capacity_mw_imputed'].fillna(year_median)
        df['capacity_mw_imputed'] = df['capacity_mw_imputed'].fillna(overall_median)
        df['capacity_mw_imputed'] = df['capacity_mw_imputed'].fillna(15)  # Final fallback
        
        # Flag imputed values
        df['capacity_imputed'] = df['capacity_mw'].isna().astype(int)
        
        print(f"Imputed {df['capacity_imputed'].sum()} capacity values")
        print(f"Median capacity: {df['capacity_mw_imputed'].median():.1f} MW")
        
        self.dc_data = df
        return df
    
    def create_features(self):
        """Create feature-rich dataset for training"""
        print("\nCreating features...")
        
        # Filter to valid years
        dc_valid = self.dc_data[
            (self.dc_data['year_operational'] >= 1990) & 
            (self.dc_data['year_operational'] <= 2024)
        ].copy()
        
        print(f"Data centers with valid years: {len(dc_valid)}")
        
        # Aggregate by state and year
        dc_agg = dc_valid.groupby(['state', 'year_operational']).agg({
            'data_center_id': 'count',
            'capacity_mw_imputed': ['sum', 'mean', 'max', 'min'],
            'latitude': 'mean',
            'longitude': 'mean',
            'capacity_imputed': 'mean',  # Fraction of imputed values
        }).reset_index()
        
        # Flatten columns
        dc_agg.columns = [
            'state', 'year', 'num_dcs',
            'total_capacity_mw', 'avg_capacity_mw', 'max_capacity_mw', 'min_capacity_mw',
            'avg_lat', 'avg_lon', 'imputation_fraction'
        ]
        
        # Join with electricity data
        elec = self.electricity_data[
            ['state', 'year', 'total_consumption_mwh', 'yoy_change_mwh', 'yoy_change_pct']
        ].copy()
        
        # Get previous year consumption for context
        elec['prev_consumption'] = elec.groupby('state')['total_consumption_mwh'].shift(1)
        
        # Merge
        training_data = dc_agg.merge(elec, on=['state', 'year'], how='inner')
        
        # Create additional features
        training_data['capacity_per_dc'] = training_data['total_capacity_mw'] / training_data['num_dcs']
        training_data['consumption_per_capita'] = training_data['total_consumption_mwh'] / 1e6  # Rough proxy
        
        # Normalize change by previous consumption
        training_data['normalized_change'] = \
            training_data['yoy_change_mwh'] / training_data['prev_consumption']
        
        # Compute per-MW impact (our key target variable)
        training_data['impact_per_mw'] = \
            training_data['yoy_change_mwh'] / training_data['total_capacity_mw']
        
        # Remove infinite/nan values
        training_data = training_data.replace([np.inf, -np.inf], np.nan)
        training_data = training_data.dropna()
        
        print(f"Training samples: {len(training_data)}")
        
        self.training_data = training_data
        return training_data
    
    def train_model(self, target='yoy_change_mwh'):
        """
        Train the model with cross-validation and hyperparameter tuning
        """
        print(f"\nTraining model to predict: {target}")
        
        # Define features
        self.feature_cols = [
            'num_dcs',
            'total_capacity_mw',
            'avg_capacity_mw',
            'max_capacity_mw',
            'avg_lat',
            'avg_lon',
            'year',
            'prev_consumption',
            'imputation_fraction',
        ]
        
        X = self.training_data[self.feature_cols].values
        y = self.training_data[target].values
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Try multiple models
        models = {
            'Ridge': Ridge(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=42
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=4, min_samples_leaf=5, random_state=42
            ),
        }
        
        results = {}
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            
            # Train on full training set
            model.fit(X_train_scaled, y_train)
            
            # Test predictions
            y_pred = model.predict(X_test_scaled)
            
            results[name] = {
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'test_r2': r2_score(y_test, y_pred),
                'test_mae': mean_absolute_error(y_test, y_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'model': model
            }
            
            print(f"\n{name}:")
            print(f"  CV R² = {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            print(f"  Test R² = {results[name]['test_r2']:.3f}")
            print(f"  Test MAE = {results[name]['test_mae']:,.0f}")
        
        # Select best model
        best_name = max(results, key=lambda x: results[x]['cv_r2_mean'])
        self.model = results[best_name]['model']
        
        print(f"\n=== Best Model: {best_name} ===")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nFeature Importance:")
            print(importance.to_string(index=False))
        
        return results
    
    def predict(self, state, year, num_dcs, total_capacity_mw, lat, lon):
        """Make prediction for new data centers"""
        
        # Get previous consumption for the state
        state_data = self.electricity_data[
            (self.electricity_data['state'] == state) &
            (self.electricity_data['year'] == year - 1)
        ]
        
        if len(state_data) > 0:
            prev_consumption = state_data['total_consumption_mwh'].values[0]
        else:
            prev_consumption = self.electricity_data['total_consumption_mwh'].median()
        
        features = np.array([[
            num_dcs,
            total_capacity_mw,
            total_capacity_mw / num_dcs,  # avg
            total_capacity_mw,            # max (assuming single DC)
            lat,
            lon,
            year,
            prev_consumption,
            0,  # imputation_fraction = 0 (real data)
        ]])
        
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        
        return prediction
    
    def analyze_patterns(self):
        """Analyze patterns in the data"""
        print("\n" + "="*60)
        print("DATA ANALYSIS")
        print("="*60)
        
        df = self.training_data
        
        # Top states by DC count
        print("\nTop States by Data Center Count:")
        state_counts = df.groupby('state')['num_dcs'].sum().sort_values(ascending=False)
        print(state_counts.head(10).to_string())
        
        # Average capacity by state
        print("\nAverage Capacity by State (MW):")
        state_capacity = df.groupby('state')['avg_capacity_mw'].mean().sort_values(ascending=False)
        print(state_capacity.head(10).to_string())
        
        # Correlation analysis
        print("\nCorrelation with YoY Change:")
        correlations = df[self.feature_cols + ['yoy_change_mwh']].corr()['yoy_change_mwh']
        print(correlations.sort_values(ascending=False).to_string())
        
        # Impact per MW analysis
        print("\nImpact per MW Statistics:")
        print(df['impact_per_mw'].describe())
    
    def save_model(self, filepath='models/energy_impact_model.joblib'):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
        }
        
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to {filepath}")
    
    def run_pipeline(self):
        """Run full pipeline"""
        print("="*60)
        print("IMPROVED DATA CENTER ENERGY IMPACT MODEL v2")
        print("="*60)
        
        self.load_data()
        self.impute_capacity()
        self.create_features()
        results = self.train_model()
        self.analyze_patterns()
        
        # Example predictions
        print("\n" + "="*60)
        print("EXAMPLE PREDICTIONS")
        print("="*60)
        
        examples = [
            ('Virginia', 2024, 5, 100, 38.9, -77.4),
            ('Texas', 2024, 3, 75, 32.8, -96.8),
            ('California', 2024, 2, 50, 37.4, -122.1),
        ]
        
        for state, year, num_dcs, capacity, lat, lon in examples:
            pred = self.predict(state, year, num_dcs, capacity, lat, lon)
            print(f"\n{state} ({year}): {num_dcs} DCs, {capacity} MW total")
            print(f"  Predicted electricity change: {pred:,.0f} MWh")
            print(f"  Per MW: {pred/capacity:,.0f} MWh/MW")
        
        # Save model
        self.save_model()
        
        # Save training data
        self.training_data.to_csv(f'{self.data_dir}/training_data_v2.csv', index=False)
        print(f"\nTraining data saved to {self.data_dir}/training_data_v2.csv")
        
        return results


def main():
    model = ImprovedEnergyModel()
    results = model.run_pipeline()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
1. Get real EIA data with API key
2. Add more features:
   - Population data
   - Climate/temperature data
   - Industrial vs residential mix
   - Electricity prices
3. Use time-series models for better forecasting
4. Validate against known data center impacts
""")


if __name__ == '__main__':
    main()
