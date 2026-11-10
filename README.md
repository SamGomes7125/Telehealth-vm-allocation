# Telehealth Demand Forecasting & VM Allocation

Forecasts weekly telehealth session demand and simulates how many virtual machines to provision, trading infrastructure cost against the risk of running out of capacity.

**Live demo:** [Hugging Face Space](https://huggingface.co/spaces/Sam-Gomes7125/Telehealth-vm-allocation)

## What it does

- **Forecasting:** a `RidgeCV` regression model trained on 104 weeks of synthetic, Poisson-distributed session demand (30-minute slots, Mon–Fri, 08:00–18:00), using one-week lag and day/time seasonal features.
- **Provisioning simulation:** turns the forecast into VM allocations and compares three strategies:
  - `delay_unaware`: provisions for predicted demand only
  - `delay_aware`: accounts for VM start-up delay
  - `delay_uncertainty_aware`: also adds a safety buffer for forecast error
- **Metrics:** forecast MAE/RMSE, SLA %, response time, queue wait, VM savings vs a flat allocation, patients served/unserved, and overload slots for each strategy, so the cost/risk trade-off is visible.

## Stack

Python · scikit-learn · NumPy · pandas · Flask (REST API) · vanilla JS dashboard · Docker (deployment on Hugging Face Spaces)

## Project structure

```
app.py           Flask API: data synthesis, model training, /predict endpoint
dashboard.html   Interactive admin dashboard served at /
```

## Run locally

```bash
pip install flask flask-cors scikit-learn numpy pandas
python app.py
# open http://localhost:8080
```
