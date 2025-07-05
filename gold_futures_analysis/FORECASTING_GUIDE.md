# FORECASTING_GUIDE.md

This file provides specialized guidance for time series forecasting projects using the Chronos framework, with focus on financial time series analysis.

## Data Specifications

### Gold Futures Data Format
- **Source**: GCUSD_MAX_FROM_PERPLEXITY.csv
- **Columns**: Date, Open, High, Low, Close, Volume
- **Format**: Daily OHLCV data, reverse chronological order (newest first)
- **Baseline Period**: 2020-2021 (517 trading days)
- **Date Range**: Full historical data available from 2020 to present

### Data Preprocessing Standards
- **Sorting**: Convert to chronological order (oldest first)
- **Target Variable**: Next-day Close price
- **Missing Values**: Forward-fill method for gaps
- **Context Window**: 63 trading days (≈3 months)
- **Prediction Horizon**: 1 day ahead (extendable)

## Model Performance Baselines

### Expected Performance Ranges (2020-2021 Gold Futures)
- **Naive Baseline**: MASE ≈ 1.0 (by definition)
- **Moving Average**: MASE ≈ 0.8-1.2
- **Chronos-Bolt-Base**: MASE ≈ 0.6-0.9 (target performance)
- **Directional Accuracy**: 45-65% (above 50% indicates skill)

### Performance Indicators
- **Excellent**: MASE < 0.7, Directional Accuracy > 60%
- **Good**: MASE 0.7-0.9, Directional Accuracy 55-60%
- **Acceptable**: MASE 0.9-1.1, Directional Accuracy 50-55%
- **Poor**: MASE > 1.1, Directional Accuracy < 50%

### Computational Expectations
- **Chronos Loading**: 2-5 minutes on CPU, 30-60 seconds on GPU
- **Rolling Evaluation**: ~0.5-2 seconds per prediction
- **Full 2020-2021 Evaluation**: 15-45 minutes depending on hardware

## Proven Evaluation Framework

### FEV Integration Pattern
```python
# Standard setup for financial time series
class FinancialForecastingTask:
    def __init__(self, dataset, context_length=63, prediction_length=1):
        self.dataset = dataset
        self.context_length = context_length  # 3 months
        self.prediction_length = prediction_length  # 1 day
```

### Rolling Window Configuration
- **Window Size**: 63 trading days (proven optimal for gold futures)
- **Step Size**: 1 day (daily rolling)
- **Minimum History**: 63 days before first prediction
- **Evaluation Period**: Start from day 64 to end of dataset

### Statistical Significance Testing
- **Method**: Diebold-Mariano test for forecast accuracy comparison
- **Significance Level**: p < 0.05
- **Multiple Comparisons**: Use Bonferroni correction when testing multiple models

## Interactive Visualization Standards

### Dashboard Components
1. **Time Series Plot**: Actual vs predicted with zoom capabilities
2. **Error Analysis**: Temporal error patterns with hover details
3. **Performance Comparison**: Bar charts with metric selection
4. **Distribution Analysis**: Error distribution histograms
5. **Directional Accuracy**: Model comparison visualization
6. **Rolling Performance**: Moving window performance metrics

### Plotly Configuration
```python
# Standard zoom-enabled configuration
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(buttons=[
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(step="all")
        ]),
        rangeslider=dict(visible=True),
        type="date"
    )
)
```

### Widget Integration
- **Model Selection**: SelectMultiple widget for comparison
- **Metric Selection**: Dropdown for different performance metrics
- **Date Range**: Interactive date selectors for period analysis
- **Export Options**: HTML and CSV export capabilities

## Model Integration Patterns

### Chronos Model Wrapper
```python
class ChronosWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.name = "Chronos-Bolt-Base"
    
    def predict_point(self, context, prediction_length=1):
        context_tensor = torch.tensor(context, dtype=torch.float32)
        quantiles, mean = self.pipeline.predict_quantiles(
            context=context_tensor,
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.5, 0.9]
        )
        return mean[0].cpu().numpy()
```

### Baseline Model Collection
- **Naive**: Last value carried forward
- **Seasonal Naive**: Same weekday value (5-day seasonality)
- **Moving Average**: 5-day simple moving average
- **Linear Trend**: Linear regression on context window

## Extension Guidelines

### Adding New Models
1. **Wrapper Class**: Implement consistent `predict_point()` interface
2. **Error Handling**: Include fallback to naive forecast on failures
3. **Performance Tracking**: Add to evaluation loop with same metrics
4. **Visualization**: Include in dashboard with unique color/style

### Expanding Time Periods
1. **Data Validation**: Ensure consistent format across periods
2. **Market Regime Analysis**: Consider different market conditions
3. **Rolling Recalibration**: Update baseline performance expectations
4. **Comparative Analysis**: Maintain consistent evaluation methodology

### Multi-Horizon Forecasting
1. **Prediction Length**: Extend from 1 to N days
2. **Metric Adaptation**: Adjust MASE calculation for longer horizons
3. **Visualization Updates**: Multi-step confidence intervals
4. **Performance Degradation**: Expect accuracy decline with horizon

## Domain-Specific Considerations

### Gold Futures Market Characteristics
- **Trading Hours**: 24-hour electronic trading
- **Volatility Patterns**: Higher during economic uncertainty
- **Seasonal Effects**: Limited but consider year-end effects
- **Economic Sensitivity**: Responds to inflation, currency, geopolitical events

### Market Regime Awareness
- **Bull Markets**: Trend-following models may perform better
- **Bear Markets**: Mean-reversion strategies could excel
- **High Volatility**: Wider prediction intervals needed
- **Low Volatility**: Directional accuracy becomes more important

### Risk Management Integration
- **Position Sizing**: Use prediction confidence for position allocation
- **Stop Losses**: Combine with traditional risk management
- **Portfolio Context**: Consider correlation with other assets
- **Transaction Costs**: Include realistic spread and commission assumptions

## Troubleshooting Common Issues

### Model Loading Problems
- **GPU Memory**: Use `torch.float32` instead of `torch.bfloat16` on limited GPU
- **Device Issues**: Fallback to CPU if CUDA unavailable
- **Model Download**: Ensure internet connection for HuggingFace model loading

### Performance Issues
- **Slow Evaluation**: Reduce batch size or use subset for testing
- **Memory Usage**: Process data in chunks for large datasets
- **Visualization Lag**: Limit data points in interactive plots

### Data Quality Issues
- **Missing Values**: Use forward-fill, avoid interpolation for financial data
- **Outliers**: Investigate but don't automatically remove (may be real events)
- **Date Alignment**: Ensure proper handling of weekends and holidays

## Best Practices

### Code Organization
- **Modular Functions**: Separate data loading, preprocessing, evaluation
- **Configuration**: Use dictionaries for model parameters
- **Logging**: Track evaluation progress and model performance
- **Documentation**: Comment complex financial calculations

### Reproducibility
- **Random Seeds**: Set for all random number generators
- **Version Tracking**: Record library versions in requirements
- **Data Snapshots**: Save preprocessed data for consistent evaluation
- **Model Checkpoints**: Save trained model states

### Performance Optimization
- **Vectorization**: Use numpy operations over loops
- **Batch Processing**: Group predictions when possible
- **Caching**: Store expensive computations
- **Profiling**: Identify bottlenecks in evaluation pipeline

## Future Development Roadmap

### Short-term Enhancements
- **Multi-asset Support**: Extend to other commodities (silver, oil, etc.)
- **Ensemble Methods**: Combine multiple forecasting approaches
- **Real-time Updates**: Stream new data and update predictions
- **Advanced Metrics**: Add Sharpe ratio, maximum drawdown analysis

### Medium-term Goals
- **Feature Engineering**: Technical indicators, sentiment data
- **Model Fine-tuning**: Optimize Chronos for financial data
- **Cross-validation**: Time series specific validation schemes
- **Automated Retraining**: Periodic model updates

### Long-term Vision
- **Multi-modal Models**: Combine price data with news, sentiment
- **Explainable AI**: Understand model decision-making process
- **Portfolio Optimization**: Integrate forecasts with allocation
- **Production Deployment**: Real-time trading system integration

## References and Resources

### Academic Papers
- Diebold-Mariano test for forecast accuracy comparison
- Time series cross-validation methodologies
- Financial forecasting evaluation frameworks

### Technical Documentation
- FEV library documentation and examples
- Chronos model architecture and training details
- Plotly/Bokeh interactive visualization guides

### Domain Knowledge
- Gold futures market microstructure
- Commodity trading risk management
- Financial time series characteristics and stylized facts