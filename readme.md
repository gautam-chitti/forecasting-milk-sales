# Milk Demand Forecasting

Accurate demand forecasting for a perishable supply chain is critical to minimize waste and optimize inventory, production, and logistics. This project builds an end-to-end pipeline to forecast monthly demand at the Customer–Item level, with rollups to quarterly and yearly horizons. It includes exploratory data analysis (EDA), modeling, validation, forecasts, and a polished report notebook aligned with the assignment brief.

## Objectives

- Predict future demand at the Customer–Item level on monthly, quarterly, and yearly baselines.
- Provide a clear EDA with trends, seasonality, and entity-level insights.
- Achieve strong predictive performance with R² ≥ 0.5 on a time-based validation split.
- Deliver actionable business recommendations and charts for presentation.

## Project Structure

| File Name                              | Description                                                                                                        |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `forecast_pipeline.py`                 | End-to-end training and forecasting pipeline. Run this first OR use the notebook version once to generate outputs. |
| `forecast_pipelineBook.ipynb`          | End-to-end pipeline in notebook form (same logic as the script). Use this first if working fully in notebooks.     |
| `Milk_Demand_Forecasting_Report.ipynb` | Reporting notebook with EDA, charts, validation, tables, and recommendations after pipeline outputs exist.         |
| `eda_summary.csv`                      | EDA aggregates (date range, counts, totals).                                                                       |
| `validation_metrics.csv`               | Overall validation metrics including R².                                                                           |
| `validation_r2_by_series.csv`          | Per-series R² for Customer–Item pairs (diagnostics).                                                               |
| `run_summary.csv`                      | Run metadata and overall R².                                                                                       |
| `forecast_monthly.csv`                 | Monthly forecasts by Customer–Item with month and quarter tags.                                                    |
| `forecast_quarterly.csv`               | Quarterly forecasts by Customer–Item.                                                                              |
| `forecast_yearly.csv`                  | Yearly forecasts by Customer–Item.                                                                                 |
| `viz_total_history.csv`                | Aggregated historical monthly net demand for plotting.                                                             |
| `viz_total_forecast.csv`               | Aggregated total monthly forecast for plotting.                                                                    |

## Data

**Input dataset**: `milk_sales_datav1.csv`

**Columns**:

- `Date`
- `Item Code`
- `Route Code`
- `Customer Code`
- `Sales Quantity`
- `Stales Quantity`

**Note**: Net Quantity is computed as `Sales Quantity − Stales Quantity` (clipped at 0).

### Example Forecast Outputs

#### `forecast_monthly.csv`

**Columns**: `Customer Code`, `Item Code`, `MonthStart`, `yhat_monthly`, `quarter`, `year`

```csv
Customer Code,Item Code,MonthStart,yhat_monthly,quarter,year
CUST001,ITEM01,2025-01-01,428.0044,2025Q1,2025
CUST001,ITEM01,2025-02-01,509.6899,2025Q1,2025
CUST001,ITEM01,2025-03-01,1103.2808,2025Q1,2025
...
```

#### `forecast_quarterly.csv`

**Columns**: `Customer Code`, `Item Code`, `quarter`, `yhat_quarterly`

```csv
Customer Code,Item Code,quarter,yhat_quarterly
CUST001,ITEM01,2025Q1,2040.9751
CUST001,ITEM01,2025Q2,2002.7522
...
```

#### `forecast_yearly.csv`

**Columns**: `Customer Code`, `Item Code`, `year`, `yhat_yearly`

```csv
Customer Code,Item Code,year,yhat_yearly
CUST001,ITEM01,2025,8036.4055
CUST001,ITEM04,2025,12151.3014
...
```

**Note**: Large-scale examples in the attached files show multiple customers and items across 2024–2025, consistent with the pipeline’s rollups.

## Requirements

- **Python**: 3.9+ (tested with 3.12)
- **Recommended libraries**:
  - `pandas`, `numpy`
  - `scikit-learn`
  - `lightgbm` (preferred) or `xgboost` (fallback) or `scikit-learn`’s `RandomForest`
  - `matplotlib`, `seaborn` (for the report notebook)

Install via:

```bash

```

```bash
pip install -r requirements.txt
```

Or quickly:

```bash
pip install pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn
```

**Note**: If `lightgbm`/`xgboost` are not available, the pipeline falls back to `RandomForest` automatically.

## How to Run

**Important**: Run either the script OR the pipeline notebook first to generate all outputs, then open the report notebook.

### Option A — Script (recommended for one-click run)

1. Place `milk_sales_datav1.csv` in the same directory.
2. Run:
   ```bash
   python forecast_pipeline.py
   ```
3. Confirm outputs exist:
   - `eda_summary.csv`, `validation_metrics.csv`, `run_summary.csv`, `validation_r2_by_series.csv`
   - `forecast_monthly.csv`, `forecast_quarterly.csv`, `forecast_yearly.csv`
   - `viz_total_history.csv`, `viz_total_forecast.csv`

### Option B — Pipeline Notebook

1. Place `milk_sales_datav1.csv` in the same directory.
2. Open `forecast_pipelineBook.ipynb` and Run All.
3. Confirm the same outputs are generated as above.

### After Running A or B

Open the report:

- `Milk_Demand_Forecasting_Report.ipynb` and Run All to render charts, tables, and recommendations.

## What the Pipeline Does

### Preprocessing

- Parses `Date` as day-first (e.g., `01-01-2022`).
- Computes `Net Quantity = Sales Quantity − Stales Quantity`, clipped at 0.

### Aggregation

- Aggregates to monthly per `Customer Code` × `Item Code`.

### Features

- **Lags**: 1, 2, 3, 6, 12 months
- **Rolling means**: 3-month, 6-month
- **Calendar features**: year, month, quarter
- **Frequency encodings**: `Customer Code` and `Item Code`

### Model and Validation

- Gradient boosting regressor (`LightGBM` preferred) with strict time-based split.
- Last 3 months reserved as validation.
- If overall R² < 0.5, blends with `Ridge` regression to lift performance.

### Forecasting

- Recursive 12-month monthly forecast per Customer–Item.
- Rollups to quarterly and yearly forecasts.
- Outputs for visualization and reporting.

## EDA and Validation

- **`eda_summary.csv`**: Date range, counts of rows, unique customers/items/routes, and totals for sales, stales, and net.
- **`validation_metrics.csv` and `run_summary.csv`**: Overall R² and model used, with row counts for train/valid/total.
- **`validation_r2_by_series.csv`**: R² per Customer–Item series on the last 3 months (diagnostics).

### Example Quality from a Run

- Overall R² ≈ 0.992 on a time-based split, indicating strong fit and signal capture.
- Aggregated monthly totals show a reasonable match from historical totals to near-term forecasts.

## Using the Outputs

### Forecasts for Planning

- **Monthly** (`forecast_monthly.csv`): Use for monthly production scheduling and dispatch sizing.
- **Quarterly** (`forecast_quarterly.csv`): Set procurement contracts and route capacity.
- **Yearly** (`forecast_yearly.csv`): Strategic planning and budget allocation.

### Reporting

Run `Milk_Demand_Forecasting_Report.ipynb` after generating outputs to get:

- EDA tables and figures
- History vs. forecast overlay plot
- Seasonality by month-of-year
- Validation metrics and R² distribution
- Top Customer–Item yearly forecasted demand
- Inventory and operations recommendations

## Inventory and Operations Recommendations

- Use monthly forecasts for production planning; quarterly rollups for procurement and route capacity alignment.
- For Customer–Item pairs with weak validation (low per-series R²), reduce dispatch frequency or initial loads and revisit MOQs.
- Apply safety stock factors for pairs with higher stale rates and higher forecast variance; re-estimate after 4 weeks with new data.
- Monitor routes historically associated with higher stales and adjust delivery windows or cold-chain handling.
- Retrain monthly with a rolling last-3-month validation window for continuous improvement.

## Reproducibility

- Random seed fixed (`SEED = 42`) for repeatable splits and model training.
- Time-based split ensures leakage-free validation aligned with real-world forecasting.
