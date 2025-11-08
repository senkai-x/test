# BTC-trade

## Time Series Forecasting Example

This repository now includes a self-contained PyTorch example for
training a neural network on a synthetic sine wave time series.  To run
it, first ensure PyTorch is installed, then execute:

```bash
pip install torch
python timeseries_model.py
```

The script will train a small LSTM-based model and print periodic loss
updates along with the final evaluation mean-squared error.
