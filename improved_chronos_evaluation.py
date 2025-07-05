#!/usr/bin/env python3
"""
Improved Chronos Evaluation Script
Addresses performance issues and implements best practices
"""

import pandas as pd
import numpy as np
import torch
from chronos import BaseChronosPipeline
import warnings
warnings.filterwarnings('ignore')

# Import our improvements
from improved_data_preprocessing import improved_data_preprocessing, create_enhanced_fev_dataset
from enhanced_chronos_wrapper import EnhancedChronosWrapper, optimize_context_length
from enhanced_evaluation import EnhancedEvaluationFramework
from model_optimization import ChronosModelOptimizer, create_ensemble_model

def main():
    print("🚀 Starting Improved Chronos Evaluation")
    print("="*60)
    
    # 1. Load and preprocess data with improvements
    print("📊 Loading and preprocessing data...")
    try:
        df = pd.read_csv('GCUSD_MAX_FROM_PERPLEXITY.csv')
        print(f"✅ Loaded {len(df)} data points")
    except FileNotFoundError:
        print("❌ Data file not found. Please ensure GCUSD_MAX_FROM_PERPLEXITY.csv is available.")
        return
    
    # Enhanced preprocessing
    data, target_col, scaler = improved_data_preprocessing(
        df, 
        use_returns=True,  # Use returns instead of raw prices
        remove_outliers=True,
        scaling_method='robust'
    )
    
    print(f"✅ Preprocessed data: {len(data)} points")
    print(f"✅ Target column: {target_col}")
    print(f"✅ Data range: {data['Date'].min()} to {data['Date'].max()}")
    
    # 2. Model optimization and selection
    print("\n🔍 Optimizing model selection...")
    optimizer = ChronosModelOptimizer()
    
    # Compare different models (comment out if you want to skip this step)
    print("Comparing different Chronos variants...")
    model_comparison = optimizer.compare_models(data, target_col, n_samples=30)
    
    if model_comparison:
        recommended_model = optimizer.get_recommended_model(model_comparison)
        print(f"✅ Recommended model: {recommended_model}")
        
        # Print comparison results
        print("\nModel Comparison Results:")
        for model_name, metrics in model_comparison.items():
            print(f"  {model_name}: Error={metrics['avg_error']:.4f}, Success={metrics['success_rate']:.2f}")
    else:
        recommended_model = "amazon/chronos-bolt-base"
        print(f"⚠️ Using default model: {recommended_model}")
    
    # 3. Load optimal model
    print(f"\n🤖 Loading {recommended_model}...")
    pipeline = optimizer.load_model_with_optimal_config(recommended_model)
    
    if pipeline is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # 4. Create enhanced wrapper
    enhanced_chronos = EnhancedChronosWrapper(pipeline, f"Enhanced-{recommended_model}")
    print(f"✅ Created enhanced wrapper: {enhanced_chronos.name}")
    
    # 5. Optimize context length
    print("\n📏 Optimizing context length...")
    optimal_context, best_score = optimize_context_length(
        data, target_col, enhanced_chronos, context_lengths=[30, 60, 90, 120]
    )
    
    if optimal_context:
        print(f"✅ Optimal context length: {optimal_context} days (error: {best_score:.4f})")
    else:
        optimal_context = 60  # Default
        print(f"⚠️ Using default context length: {optimal_context} days")
    
    # 6. Create baseline models (simplified)
    class SimpleBaseline:
        def __init__(self, name, func):
            self.name = name
            self.func = func
        
        def predict_point(self, context, prediction_length=1):
            return self.func(context, prediction_length)
    
    def naive_forecast(context, prediction_length=1):
        return np.full(prediction_length, context[-1])
    
    def ma_forecast(context, prediction_length=1):
        return np.full(prediction_length, np.mean(context[-5:]))
    
    baseline_models = {
        'Naive': SimpleBaseline('Naive', naive_forecast),
        'Moving_Average': SimpleBaseline('Moving Average', ma_forecast)
    }
    
    # 7. Combine all models
    all_models = {**baseline_models, 'Enhanced_Chronos': enhanced_chronos}
    
    # 8. Enhanced evaluation
    print(f"\n📈 Running enhanced evaluation...")
    evaluator = EnhancedEvaluationFramework(data, target_col, scaler)
    
    # Walk-forward validation
    results = evaluator.walk_forward_validation(
        models=all_models,
        initial_train_size=200,
        step_size=10,
        max_predictions=100
    )
    
    print(f"✅ Generated predictions for {len(results['Enhanced_Chronos']['predictions'])} time points")
    
    # 9. Calculate enhanced metrics
    print("\n📊 Calculating enhanced metrics...")
    metrics = evaluator.calculate_enhanced_metrics(results)
    
    # Display results
    print("\n" + "="*80)
    print("🏆 ENHANCED EVALUATION RESULTS")
    print("="*80)
    
    # Create results DataFrame
    metrics_df = pd.DataFrame(metrics).T
    
    # Sort by multiple criteria
    metrics_df['Combined_Score'] = (
        metrics_df['MAE'] * 0.3 + 
        metrics_df['RMSE'] * 0.3 + 
        (100 - metrics_df['Hit_Rate']) * 0.2 +  # Invert hit rate for minimization
        abs(metrics_df['Sharpe_Ratio']) * -0.2  # Favor higher Sharpe ratio
    )
    
    ranking = metrics_df.sort_values('Combined_Score')
    
    print("\n📊 MODEL RANKING:")
    print("-" * 60)
    for i, (model, row) in enumerate(ranking.iterrows(), 1):
        print(f"{i}. {model:20s}")
        print(f"   MAE: {row['MAE']:8.4f} | RMSE: {row['RMSE']:8.4f}")
        print(f"   Hit Rate: {row['Hit_Rate']:6.1f}% | Sharpe: {row['Sharpe_Ratio']:6.3f}")
        print(f"   Total Return: {row['Total_Return']:6.1f}% | Max DD: {row['Max_Drawdown']:6.1f}%")
        print("-" * 60)
    
    # Enhanced Chronos specific analysis
    if 'Enhanced_Chronos' in metrics:
        chronos_metrics = metrics['Enhanced_Chronos']
        naive_metrics = metrics.get('Naive', {})
        
        print(f"\n🤖 ENHANCED CHRONOS ANALYSIS:")
        print("-" * 60)
        print(f"Prediction Accuracy (MAE): {chronos_metrics['MAE']:.4f}")
        print(f"Hit Rate (Direction): {chronos_metrics['Hit_Rate']:.1f}%")
        print(f"Trading Performance: {chronos_metrics['Total_Return']:.1f}% return")
        print(f"Risk-Adjusted Return: {chronos_metrics['Sharpe_Ratio']:.3f} Sharpe ratio")
        
        if naive_metrics:
            mae_improvement = ((naive_metrics['MAE'] - chronos_metrics['MAE']) / naive_metrics['MAE']) * 100
            print(f"Improvement over Naive: {mae_improvement:+.1f}%")
    
    # 10. Market regime analysis
    print(f"\n🏛️ Market Regime Analysis...")
    regime_analysis = evaluator.market_regime_analysis(results)
    
    for model_name, regimes in regime_analysis.items():
        if model_name == 'Enhanced_Chronos':
            print(f"\n{model_name} performance by market regime:")
            for regime, regime_metrics in regimes.items():
                print(f"  {regime:20s}: MAE={regime_metrics['MAE']:.4f}, Hit Rate={regime_metrics['Hit_Rate']:.1f}%")
    
    # 11. Export results
    print(f"\n💾 Exporting results...")
    
    # Export metrics
    metrics_df.to_csv('enhanced_chronos_metrics.csv')
    print("✅ Metrics saved to enhanced_chronos_metrics.csv")
    
    # Export detailed predictions
    all_predictions = []
    for model_name, model_results in results.items():
        for i, date in enumerate(model_results['dates']):
            all_predictions.append({
                'Date': date,
                'Model': model_name,
                'Actual': model_results['actuals'][i],
                'Predicted': model_results['predictions'][i],
                'Error': model_results['errors'][i]
            })
    
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv('enhanced_chronos_predictions.csv', index=False)
    print("✅ Predictions saved to enhanced_chronos_predictions.csv")
    
    # 12. Final recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("="*60)
    
    best_model = ranking.index[0]
    if best_model == 'Enhanced_Chronos':
        print("✅ Enhanced Chronos shows superior performance!")
        print("✅ Consider deploying for production forecasting")
        print(f"✅ Optimal context length: {optimal_context} days")
        print(f"✅ Key improvement: {mae_improvement:+.1f}% MAE reduction vs baseline")
    else:
        print(f"⚠️  {best_model} outperforms Enhanced Chronos")
        print("💡 Consider ensemble methods combining Chronos with traditional models")
        print("💡 Investigate different preprocessing approaches")
        print("💡 Try fine-tuning Chronos on domain-specific data")
    
    print("\n📋 Next steps for further improvement:")
    print("   1. Experiment with different data transformations")
    print("   2. Try ensemble methods")
    print("   3. Validate on out-of-sample periods")
    print("   4. Consider fine-tuning on gold-specific data")
    print("   5. Integrate external features (macro indicators, etc.)")
    
    print(f"\n🎉 Enhanced evaluation completed!")
    return results, metrics, ranking

if __name__ == "__main__":
    results, metrics, ranking = main()