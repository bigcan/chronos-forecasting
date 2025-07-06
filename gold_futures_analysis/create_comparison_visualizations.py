#!/usr/bin/env python3
"""
Create comparative visualizations for 2016-2019 vs 2020-2021 analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("📊 Creating Market Regime Comparison Visualizations")
print("="*60)

# Load data
def load_and_prepare_data():
    """Load and prepare data for visualization"""
    
    # Load full dataset
    df = pd.read_csv('GCUSD_MAX_FROM_PERPLEXITY.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Split periods
    mask_2016_2019 = (df['Date'] >= '2016-01-01') & (df['Date'] <= '2019-12-31')
    mask_2020_2021 = (df['Date'] >= '2020-01-01') & (df['Date'] <= '2021-12-31')
    
    data_2016_2019 = df[mask_2016_2019].reset_index(drop=True)
    data_2020_2021 = df[mask_2020_2021].reset_index(drop=True)
    
    # Load comparison results
    comparison_df = pd.read_csv('baseline_comparison_2016_2019_vs_2020_2021.csv')
    
    return data_2016_2019, data_2020_2021, comparison_df

# Create visualizations
def create_visualizations():
    """Create comprehensive comparison visualizations"""
    
    data_2016_2019, data_2020_2021, comparison_df = load_and_prepare_data()
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Price comparison over time
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(data_2016_2019['Date'], data_2016_2019['Close'], 
             label='2016-2019', linewidth=2, color='blue', alpha=0.8)
    ax1.set_title('2016-2019 Price Evolution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(data_2020_2021['Date'], data_2020_2021['Close'], 
             label='2020-2021', linewidth=2, color='red', alpha=0.8)
    ax2.set_title('2020-2021 Price Evolution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Price ($)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 2. Volatility comparison
    ax3 = plt.subplot(3, 3, 3)
    returns_2016_2019 = data_2016_2019['Close'].pct_change().dropna()
    returns_2020_2021 = data_2020_2021['Close'].pct_change().dropna()
    
    ax3.hist(returns_2016_2019, bins=50, alpha=0.7, label='2016-2019', color='blue', density=True)
    ax3.hist(returns_2020_2021, bins=50, alpha=0.7, label='2020-2021', color='red', density=True)
    ax3.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Daily Return', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 3. Rolling volatility
    ax4 = plt.subplot(3, 3, 4)
    rolling_vol_2016_2019 = returns_2016_2019.rolling(window=30).std() * np.sqrt(252) * 100
    rolling_vol_2020_2021 = returns_2020_2021.rolling(window=30).std() * np.sqrt(252) * 100
    
    ax4.plot(data_2016_2019['Date'][1:], rolling_vol_2016_2019, 
             label='2016-2019', linewidth=2, color='blue', alpha=0.8)
    ax4.set_title('2016-2019 Rolling Volatility (30-day)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Annualized Volatility (%)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(data_2020_2021['Date'][1:], rolling_vol_2020_2021, 
             label='2020-2021', linewidth=2, color='red', alpha=0.8)
    ax5.set_title('2020-2021 Rolling Volatility (30-day)', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Annualized Volatility (%)', fontsize=12)
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 4. MASE comparison
    ax6 = plt.subplot(3, 3, 6)
    models = comparison_df['Model']
    x_pos = np.arange(len(models))
    
    bars1 = ax6.bar(x_pos - 0.2, comparison_df['MASE_2016_2019'], 
                    0.4, label='2016-2019', color='blue', alpha=0.7)
    bars2 = ax6.bar(x_pos + 0.2, comparison_df['MASE_2020_2021'], 
                    0.4, label='2020-2021', color='red', alpha=0.7)
    
    ax6.set_title('MASE Comparison by Model', fontsize=14, fontweight='bold')
    ax6.set_ylabel('MASE', fontsize=12)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(models, rotation=45, ha='right')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 5. Directional accuracy comparison
    ax7 = plt.subplot(3, 3, 7)
    bars3 = ax7.bar(x_pos - 0.2, comparison_df['Dir_Acc_2016_2019'], 
                    0.4, label='2016-2019', color='blue', alpha=0.7)
    bars4 = ax7.bar(x_pos + 0.2, comparison_df['Dir_Acc_2020_2021'], 
                    0.4, label='2020-2021', color='red', alpha=0.7)
    
    ax7.set_title('Directional Accuracy Comparison', fontsize=14, fontweight='bold')
    ax7.set_ylabel('Directional Accuracy (%)', fontsize=12)
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(models, rotation=45, ha='right')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 6. Performance change
    ax8 = plt.subplot(3, 3, 8)
    performance_change = comparison_df['MASE_Change']
    colors = ['green' if x < 0 else 'red' for x in performance_change]
    
    bars5 = ax8.bar(models, performance_change, color=colors, alpha=0.7)
    ax8.set_title('MASE Performance Change (%)', fontsize=14, fontweight='bold')
    ax8.set_ylabel('Change (%)', fontsize=12)
    ax8.set_xticklabels(models, rotation=45, ha='right')
    ax8.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax8.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars5:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -1),
                f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
    
    # 7. Market regime summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Create summary table
    summary_data = [
        ['Metric', '2016-2019', '2020-2021', 'Difference'],
        ['Trading Days', '1,031', '517', '-514'],
        ['Total Return', '41.7%', '19.7%', '-22.0pp'],
        ['Volatility', '12.27%', '18.39%', '+6.12pp'],
        ['Max Drawdown', '-17.7%', '-18.9%', '-1.2pp'],
        ['Sharpe Ratio', '0.76', '0.57', '-0.19'],
        ['', '', '', ''],
        ['Best Model', 'Naive', 'Naive', 'Consistent'],
        ['Naive MASE', '0.9995', '1.0014', '+0.19%'],
        ['Chronos MASE*', 'N/A', '1.4259', '43.3% gap']
    ]
    
    table = ax9.table(cellText=summary_data[1:], colLabels=summary_data[0],
                      cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_data)):
        for j in range(len(summary_data[0])):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            elif i == 7:  # Empty row
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('#f9f9f9')
    
    ax9.set_title('Market Regime Summary', fontsize=14, fontweight='bold', pad=20)
    
    # Overall title
    fig.suptitle('Gold Futures Market Regime Analysis: 2016-2019 vs 2020-2021', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # Save the plot
    plt.savefig('market_regime_comparison_2016_2019_vs_2020_2021.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig('market_regime_comparison_2016_2019_vs_2020_2021.pdf', 
                bbox_inches='tight')
    
    print("✅ Comprehensive visualization created and saved")
    print("   - PNG: market_regime_comparison_2016_2019_vs_2020_2021.png")
    print("   - PDF: market_regime_comparison_2016_2019_vs_2020_2021.pdf")
    
    return fig

# Create individual focused charts
def create_focused_charts():
    """Create individual focused charts for key insights"""
    
    data_2016_2019, data_2020_2021, comparison_df = load_and_prepare_data()
    
    # 1. Naive vs Chronos Performance Chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Data for comparison
    models = ['Naive 2016-2019', 'Naive 2020-2021', 'Chronos 2020-2021']
    mase_values = [0.9995, 1.0014, 1.4259]  # Chronos from 2020-2021 analysis
    colors = ['blue', 'red', 'orange']
    
    bars = ax.bar(models, mase_values, color=colors, alpha=0.7)
    ax.set_title('MASE Performance: Market Regime Impact on Forecasting', 
                 fontsize=16, fontweight='bold')
    ax.set_ylabel('MASE (Lower is Better)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at MASE = 1.0
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, 
               label='MASE = 1.0 (Naive Baseline)')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add annotations
    ax.annotate('Nearly Optimal\nPerformance', xy=(0, 0.9995), xytext=(0, 0.8),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=12, ha='center', color='blue', fontweight='bold')
    
    ax.annotate('Slight Degradation\nin Volatile Period', xy=(1, 1.0014), xytext=(1, 1.2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, ha='center', color='red', fontweight='bold')
    
    ax.annotate('43.3% Gap\nfrom Naive', xy=(2, 1.4259), xytext=(2, 1.6),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2),
                fontsize=12, ha='center', color='orange', fontweight='bold')
    
    ax.legend()
    plt.tight_layout()
    plt.savefig('naive_vs_chronos_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Volatility Impact Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Market characteristics
    periods = ['2016-2019', '2020-2021']
    volatility = [12.27, 18.39]
    returns = [41.7, 19.7]
    
    bars1 = ax1.bar(periods, volatility, color=['blue', 'red'], alpha=0.7)
    ax1.set_title('Annualized Volatility by Period', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Volatility (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    bars2 = ax2.bar(periods, returns, color=['blue', 'red'], alpha=0.7)
    ax2.set_title('Total Return by Period', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Return (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.8,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    fig.suptitle('Market Regime Characteristics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('market_regime_characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Focused charts created:")
    print("   - naive_vs_chronos_performance.png")
    print("   - market_regime_characteristics.png")

if __name__ == "__main__":
    try:
        create_visualizations()
        create_focused_charts()
        print("\n🎉 All visualizations created successfully!")
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()