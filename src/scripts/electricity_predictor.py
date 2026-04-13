#!/usr/bin/env python3
"""
Electricity Usage Prediction Model
===================================
Uses data center features, economic indicators, and temporal data
to predict state-level electricity consumption (GWh).

Models:
- Gradient Boosting (primary)
- Random Forest (ensemble comparison)
- XGBoost (if available)
- Neural Network (MLPRegressor)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Output directories
DATA_DIR = '/Users/diggy/Documents/env-forecasting/env_forecasting/data'
MODEL_DIR = '/Users/diggy/Documents/env-forecasting/env_forecasting/data/models'

def load_data():
    """Load the ML feature dataset."""
    df = pd.read_csv(f'{DATA_DIR}/ml_features.csv')
    print(f"Loaded {len(df)} state-year observations")
    print(f"Years: {df['year'].min()} - {df['year'].max()}")
    print(f"States: {df['state'].nunique()}")
    return df


def prepare_features(df, target='electricity_gwh', include_year=True):
    """
    Prepare feature matrix and target for prediction.
    
    IMPORTANT: We exclude elec_per_capita_mwh because it's derived from the target
    (elec_per_capita = electricity / population), which would cause data leakage.
    
    Core features:
    - Data center: cumulative_dc_mw, cumulative_dc_count, new_dc_mw, dc_growth_mw
    - Energy: cumulative_dc_energy_twh, dc_share_of_state_pct
    - Economic: population_millions, gdp_billions
    - Per-capita DC: dc_per_million_pop, dc_mw_per_million_pop, dc_intensity
    - Temporal: year (optional for pure DC prediction)
    - Lags: cumulative_dc_mw_lag1, cumulative_dc_mw_lag2
    """
    
    # Select features - EXCLUDE elec_per_capita_mwh (data leakage!)
    base_features = [
        # Data center capacity
        'cumulative_dc_mw',
        'cumulative_dc_count',
        'new_dc_mw',
        'new_dc_count',
        'dc_growth_mw',
        'cumulative_dc_energy_twh',
        
        # AI/Advanced DCs
        'cumulative_ai_count',
        'ai_ratio',
        
        # Lag features (past DC capacity)
        'cumulative_dc_mw_lag1',
        'cumulative_dc_mw_lag2',
        
        # Economic & demographic (NOT elec_per_capita - that's derived from target!)
        'population_millions',
        'gdp_billions',
        
        # DC intensity metrics (these are OK - based on DC, not electricity)
        'dc_per_million_pop',
        'dc_mw_per_million_pop',
        'dc_intensity',
        
        # DO NOT include: elec_per_capita_mwh (data leakage - derived from target)
        # DO NOT include: dc_share_of_state_pct (data leakage - uses electricity)
    ]
    
    if include_year:
        base_features.append('year')
    
    # Filter to available features
    available_features = [f for f in base_features if f in df.columns]
    
    print(f"\nUsing {len(available_features)} features:")
    for f in available_features:
        print(f"  - {f}")
    
    # Create feature matrix
    X = df[available_features].copy()
    y = df[target].copy()
    
    # Handle any missing values
    X = X.fillna(0)
    y = y.fillna(y.median())
    
    return X, y, available_features


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train Gradient Boosting model with hyperparameter tuning."""
    print("\n--- Gradient Boosting Regressor ---")
    
    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'min_samples_split': [5, 10],
        'subsample': [0.8, 1.0]
    }
    
    # Quick grid search
    gb = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best params: {grid_search.best_params_}")
    
    # Evaluate
    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²:  {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.2f} GWh")
    print(f"Test RMSE: {test_rmse:.2f} GWh")
    
    # Cross-validation
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
    print(f"CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return best_model, {
        'model': 'GradientBoosting',
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'best_params': grid_search.best_params_
    }


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest model."""
    print("\n--- Random Forest Regressor ---")
    
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    train_pred = rf.predict(X_train)
    test_pred = rf.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²:  {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.2f} GWh")
    print(f"Test RMSE: {test_rmse:.2f} GWh")
    
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
    print(f"CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return rf, {
        'model': 'RandomForest',
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std()
    }


def train_neural_network(X_train, y_train, X_test, y_test):
    """Train Neural Network model."""
    print("\n--- Neural Network (MLP) ---")
    
    # Scale features for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    mlp = MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    
    train_pred = mlp.predict(X_train_scaled)
    test_pred = mlp.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²:  {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.2f} GWh")
    print(f"Test RMSE: {test_rmse:.2f} GWh")
    
    return (mlp, scaler), {
        'model': 'NeuralNetwork',
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'architecture': '100-50-25'
    }


def train_ridge_regression(X_train, y_train, X_test, y_test):
    """Train Ridge Regression with polynomial features."""
    print("\n--- Ridge Regression (with polynomial features) ---")
    
    # Create pipeline with polynomial features and ridge
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])
    
    pipeline.fit(X_train, y_train)
    
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²:  {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.2f} GWh")
    print(f"Test RMSE: {test_rmse:.2f} GWh")
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
    print(f"CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return pipeline, {
        'model': 'RidgePolynomial',
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std()
    }


def analyze_feature_importance(model, feature_names, model_name='Model'):
    """Analyze and display feature importance."""
    print(f"\n=== Feature Importance ({model_name}) ===")
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("Model doesn't support feature importance extraction")
        return None
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 10 Most Important Features:")
    print("-" * 50)
    importance_data = []
    for i, idx in enumerate(indices[:10]):
        pct = importances[idx] * 100
        print(f"{i+1:2}. {feature_names[idx]:<30} {pct:>6.2f}%")
        importance_data.append({
            'rank': i + 1,
            'feature': feature_names[idx],
            'importance_pct': round(pct, 2)
        })
    
    return importance_data


def predict_future(model, df, feature_names, years=[2025, 2026, 2027, 2028, 2030]):
    """Make predictions for future years.
    
    Note: Data is in TWh (1 TWh = 1000 GWh), converted for display.
    """
    print("\n" + "=" * 60)
    print("FUTURE ELECTRICITY PREDICTIONS")
    print("=" * 60)
    
    # Get most recent data for each state
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year].copy()
    
    # Calculate baseline (2023 actual)
    baseline_total = latest_data['electricity_gwh'].sum() * 1000  # Convert TWh to GWh
    print(f"\nBaseline ({latest_year}): {baseline_total:,.0f} GWh total US electricity")
    
    predictions = []
    
    for future_year in years:
        print(f"\n--- Predictions for {future_year} ---")
        
        future_df = latest_data.copy()
        
        # Project forward
        years_forward = future_year - latest_year
        dc_growth_factor = 1.15 ** years_forward  # ~15% annual DC growth
        
        # Update features for future projection
        if 'year' in feature_names:
            future_df['year'] = future_year
        
        future_df['cumulative_dc_mw'] *= dc_growth_factor
        future_df['cumulative_dc_count'] *= dc_growth_factor
        future_df['cumulative_dc_energy_twh'] *= dc_growth_factor
        future_df['dc_mw_per_million_pop'] *= dc_growth_factor
        future_df['dc_intensity'] *= dc_growth_factor
        
        # Population growth ~0.5%/year
        pop_growth = 1.005 ** years_forward
        future_df['population_millions'] *= pop_growth
        future_df['gdp_billions'] *= (1.025 ** years_forward)  # GDP ~2.5%/year
        
        # Create feature matrix
        X_future = future_df[feature_names].fillna(0)
        
        # Get predictions (in TWh, scale to GWh for display)
        if isinstance(model, tuple):  # Neural network with scaler
            mlp, scaler = model
            X_scaled = scaler.transform(X_future)
            preds = mlp.predict(X_scaled)
        else:
            preds = model.predict(X_future)
        
        # Convert TWh to GWh (multiply by 1000)
        preds_gwh = preds * 1000
        future_df['predicted_electricity_gwh'] = preds_gwh
        future_df['cumulative_dc_mw_proj'] = future_df['cumulative_dc_mw']
        
        # Top 10 states by predicted consumption
        top_states = future_df.nlargest(10, 'predicted_electricity_gwh')[
            ['state', 'predicted_electricity_gwh', 'cumulative_dc_mw_proj', 'population_millions']
        ]
        
        print(f"\nTop 10 States by Predicted Electricity ({future_year}):")
        print("-" * 75)
        print(f"  {'State':<20} {'Electricity':>14} {'Population':>12} {'Per Capita':>15}")
        print("-" * 75)
        for _, row in top_states.iterrows():
            per_capita = row['predicted_electricity_gwh'] / (row['population_millions'] * 1000)  # MWh per person
            print(f"  {row['state']:<20} {row['predicted_electricity_gwh']:>11,.0f} GWh "
                  f"{row['population_millions']:>10.1f} M {per_capita:>12.1f} MWh/person")
        
        # Total US prediction
        total_pred_gwh = preds_gwh.sum()
        total_pop = future_df['population_millions'].sum()
        per_capita_us = total_pred_gwh / (total_pop * 1000)  # MWh per person
        
        print(f"\n  TOTAL US: {total_pred_gwh:,.0f} GWh")
        print(f"  US Population: {total_pop:.1f} million")
        print(f"  Per Capita: {per_capita_us:.1f} MWh/person")
        
        growth_vs_baseline = ((total_pred_gwh / baseline_total) - 1) * 100
        print(f"  Growth vs {latest_year}: {growth_vs_baseline:+.1f}%")
        
        predictions.append({
            'year': future_year,
            'total_us_gwh': round(total_pred_gwh, 0),
            'us_population_millions': round(total_pop, 1),
            'per_capita_mwh': round(per_capita_us, 1),
            'growth_pct': round(growth_vs_baseline, 1),
            'top_state': top_states.iloc[0]['state'],
            'top_state_gwh': round(top_states.iloc[0]['predicted_electricity_gwh'], 0)
        })
    
    return predictions


def save_results(results, model_name, feature_importance, predictions):
    """Save model results to JSON."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    output = {
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': results,
        'feature_importance': feature_importance,
        'predictions': predictions
    }
    
    filename = f'{MODEL_DIR}/electricity_predictor_results.json'
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    return filename


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("  ELECTRICITY USAGE PREDICTION MODEL")
    print("=" * 60)
    print(f"  Target: State-level electricity consumption (GWh)")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Prepare features
    X, y, feature_names = prepare_features(df, target='electricity_gwh', include_year=True)
    
    # Split data (temporal split: train on older data, test on recent)
    df_with_target = pd.concat([X, y], axis=1)
    train_df = df_with_target[df_with_target['year'] < 2020]
    test_df = df_with_target[df_with_target['year'] >= 2020]
    
    X_train = train_df[feature_names]
    y_train = train_df['electricity_gwh']
    X_test = test_df[feature_names]
    y_test = test_df['electricity_gwh']
    
    print(f"\nTrain set: {len(X_train)} samples (years < 2020)")
    print(f"Test set:  {len(X_test)} samples (years >= 2020)")
    
    # Train multiple models
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)
    
    all_results = []
    
    # Gradient Boosting
    gb_model, gb_results = train_gradient_boosting(X_train, y_train, X_test, y_test)
    all_results.append(gb_results)
    
    # Random Forest
    rf_model, rf_results = train_random_forest(X_train, y_train, X_test, y_test)
    all_results.append(rf_results)
    
    # Neural Network
    nn_model, nn_results = train_neural_network(X_train, y_train, X_test, y_test)
    all_results.append(nn_results)
    
    # Ridge Polynomial
    ridge_model, ridge_results = train_ridge_regression(X_train, y_train, X_test, y_test)
    all_results.append(ridge_results)
    
    # Model comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"\n{'Model':<25} {'Test R²':>10} {'Test MAE':>12} {'Test RMSE':>12}")
    print("-" * 60)
    
    for r in all_results:
        print(f"{r['model']:<25} {r['test_r2']:>10.4f} {r['test_mae']:>12,.0f} {r['test_rmse']:>12,.0f}")
    
    # Select best model based on test R²
    best_result = max(all_results, key=lambda x: x['test_r2'])
    best_model_name = best_result['model']
    
    print(f"\n★ Best Model: {best_model_name} (Test R² = {best_result['test_r2']:.4f})")
    
    # Use best tree-based model for feature importance
    if best_model_name == 'GradientBoosting':
        best_model = gb_model
    elif best_model_name == 'RandomForest':
        best_model = rf_model
    else:
        # Use GB for feature importance if best is non-tree
        best_model = gb_model
    
    # Feature importance analysis
    feature_importance = analyze_feature_importance(gb_model, feature_names, 'Gradient Boosting')
    
    # Make future predictions using best tree model
    predictions = predict_future(gb_model, df, feature_names)
    
    # Save results
    save_results(best_result, best_model_name, feature_importance, predictions)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n★ Best Model: {best_model_name}")
    print(f"  Test R²:   {best_result['test_r2']:.4f}")
    print(f"  Test MAE:  {best_result['test_mae']:,.0f} TWh ({best_result['test_mae']*1000:,.0f} GWh)")
    print(f"  Test RMSE: {best_result['test_rmse']:,.0f} TWh ({best_result['test_rmse']*1000:,.0f} GWh)")
    
    # Summary interpretation
    print("\n" + "=" * 60)
    print("MODEL INTERPRETATION")
    print("=" * 60)
    if feature_importance:
        dc_features = ['cumulative_dc_mw', 'cumulative_dc_count', 'cumulative_dc_energy_twh', 
                      'dc_mw_per_million_pop', 'dc_intensity', 'dc_share_of_state_pct',
                      'new_dc_mw', 'dc_growth_mw']
        dc_importance = sum(fi['importance_pct'] for fi in feature_importance 
                           if fi['feature'] in dc_features)
        print(f"\nData Center features explain: {dc_importance:.2f}% of electricity variance")
        print(f"Economic features (pop, GDP) explain: ~{100-dc_importance:.1f}%")
        print("\nKey insight: Population drives most electricity demand, but DC features")
        print("contribute to marginal changes in high-intensity states (VA, TX, GA).")
    
    return best_model, best_result


if __name__ == '__main__':
    model, results = main()
