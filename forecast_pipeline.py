import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = 'milk_sales_datav1.csv'
SEED = 42
np.random.seed(SEED)

def parse_date(s):
    
    return pd.to_datetime(s, dayfirst=True, errors='coerce')

def main():
    
    df = pd.read_csv(INPUT_FILE)
    
    df.columns = [c.strip().title().replace('_',' ') for c in df.columns]

    # Validate essential columns
    required_cols = ['Date','Item Code','Route Code','Customer Code','Sales Quantity','Stales Quantity']
    miss = [c for c in required_cols if c not in df.columns]
    if miss:
        raise ValueError(f'Missing columns: {miss}')

  
    df['Date'] = parse_date(df['Date'])
    df['Sales Quantity'] = pd.to_numeric(df['Sales Quantity'], errors='coerce')
    df['Stales Quantity'] = pd.to_numeric(df['Stales Quantity'], errors='coerce').fillna(0)

    df = df.dropna(subset=['Date','Item Code','Customer Code','Sales Quantity']).copy()

    # Net quantity
    df['Net Quantity'] = (df['Sales Quantity'] - df['Stales Quantity']).clip(lower=0)

    #  EDA
    eda = {
        'date_min': df['Date'].min(),
        'date_max': df['Date'].max(),
        'n_rows': len(df),
        'n_customers': df['Customer Code'].nunique(),
        'n_items': df['Item Code'].nunique(),
        'n_routes': df['Route Code'].nunique(),
        'total_sales_qty': df['Sales Quantity'].sum(),
        'total_stales_qty': df['Stales Quantity'].sum(),
        'total_net_qty': df['Net Quantity'].sum(),
    }
    pd.DataFrame({k:[v] for k,v in eda.items()}).to_csv('eda_summary.csv', index=False)

    # Monthly aggregation
    daily = (df.groupby(['Customer Code','Item Code','Date'], as_index=False)['Net Quantity']
               .sum())
    monthly = daily.copy()
    monthly['MonthStart'] = monthly['Date'].values.astype('datetime64[M]')
    monthly = (monthly.groupby(['Customer Code','Item Code','MonthStart'], as_index=False)['Net Quantity']
                      .sum())

    # Supervised frame
    m = monthly.rename(columns={'MonthStart':'ds','Net Quantity':'y'})
    m['year'] = m['ds'].dt.year
    m['month'] = m['ds'].dt.month
    m['quarter'] = m['ds'].dt.quarter

    # Lag features 
    lag_k = [1,2,3,6,12]
    chunks = []
    for (cust,item), g in m.sort_values('ds').groupby(['Customer Code','Item Code']):
        g = g.copy()
        for k in lag_k:
            g[f'lag_{k}'] = g['y'].shift(k)
        g['rolling_3'] = g['y'].rolling(3).mean()
        g['rolling_6'] = g['y'].rolling(6).mean()
        chunks.append(g)
    X = pd.concat(chunks, ignore_index=True)
    lag_cols = [f'lag_{k}' for k in lag_k] + ['rolling_3','rolling_6']
    X = X.dropna(subset=lag_cols).copy()

    for col in ['Customer Code','Item Code']:
        freq = X[col].value_counts(normalize=True)
        X[f'{col}_freq'] = X[col].map(freq).astype(float)

   
    cutoff = X['ds'].max() - pd.offsets.MonthBegin(4)
    train = X[X['ds'] < cutoff].copy()
    valid = X[X['ds'] >= cutoff].copy()

    features = lag_cols + ['year','month','quarter','Customer Code_freq','Item Code_freq']

    model_name = None
    model = None
    try:
        import lightgbm as lgb
        model_name = 'lightgbm'
        model = lgb.LGBMRegressor(
            random_state=SEED, n_estimators=1200, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, num_leaves=63
        )
    except Exception:
        try:
            from xgboost import XGBRegressor
            model_name = 'xgboost'
            model = XGBRegressor(
                random_state=SEED, n_estimators=1200, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                max_depth=8, tree_method='hist'
            )
        except Exception:
            from sklearn.ensemble import RandomForestRegressor
            model_name = 'randomforest'
            model = RandomForestRegressor(
                random_state=SEED, n_estimators=600, n_jobs=-1, max_depth=None
            )

    from sklearn.metrics import r2_score
    from sklearn.linear_model import Ridge

    model.fit(train[features], train['y'])
    valid_pred = model.predict(valid[features])
    r2 = r2_score(valid['y'], valid_pred)

    metrics = pd.DataFrame({'metric':['r2'], 'value':[r2], 'model':[model_name]})

    if r2 < 0.5:
        ridge = Ridge(alpha=1.0, random_state=SEED)
        ridge.fit(train[features], train['y'])
        valid_pred2 = ridge.predict(valid[features])
        valid_blend = 0.7*valid_pred + 0.3*valid_pred2
        r2_blend = r2_score(valid['y'], valid_blend)
        metrics = pd.concat([metrics, pd.DataFrame({'metric':['r2_blend'], 'value':[r2_blend], 'model':[model_name + '+ridge']})], ignore_index=True)
        if r2_blend > r2:
            valid_pred = valid_blend
            r2 = r2_blend

    metrics.to_csv('validation_metrics.csv', index=False)

    
    series_scores = []
    for (cust,item), g in X.groupby(['Customer Code','Item Code']):
        g = g.sort_values('ds').copy()
        cut = g['ds'].max() - pd.offsets.MonthBegin(4)
        tr = g[g['ds'] < cut]
        va = g[g['ds'] >= cut]
        if len(tr) == 0 or len(va) == 0:
            continue
        yhat = model.predict(va[features])
        s = r2_score(va['y'], yhat)
        series_scores.append({'Customer Code':cust,'Item Code':item,'r2':s})
    pd.DataFrame(series_scores).to_csv('validation_r2_by_series.csv', index=False)

    
    full = X.sort_values('ds').copy()
    final_model = model
    final_model.fit(full[features], full['y'])

   
    horizon = 12
    future_rows = []

    cust_freq_all = full['Customer Code'].value_counts(normalize=True)
    item_freq_all = full['Item Code'].value_counts(normalize=True)

    eng_hist = full[['Customer Code','Item Code','ds','y'] + lag_cols + ['year','month','quarter','Customer Code_freq','Item Code_freq']].copy()

    for (cust,item), g in m.groupby(['Customer Code','Item Code']):
        g = g.sort_values('ds').copy()
        last_ds = g['ds'].max()

        hist = eng_hist[(eng_hist['Customer Code']==cust) & (eng_hist['Item Code']==item)].sort_values('ds').copy()
        recent = hist[['ds','y']].tail(12).copy()

        for h in range(1, horizon+1):
            next_ds = (last_ds + pd.offsets.MonthBegin(h))
            feat = {}
            feat['ds'] = pd.Timestamp(next_ds)
            feat['year'] = feat['ds'].year
            feat['month'] = feat['ds'].month
            feat['quarter'] = feat['ds'].quarter

            y_hist = recent.sort_values('ds')['y']
            def last_k(k):
                return y_hist.iloc[-k] if len(y_hist) >= k else np.nan

            for k in [1,2,3,6,12]:
                feat[f'lag_{k}'] = last_k(k)
            feat['rolling_3'] = y_hist.iloc[-3:].mean() if len(y_hist) >= 3 else np.nan
            feat['rolling_6'] = y_hist.iloc[-6:].mean() if len(y_hist) >= 6 else np.nan

            feat['Customer Code_freq'] = float(cust_freq_all.get(cust, 0.0))
            feat['Item Code_freq'] = float(item_freq_all.get(item, 0.0))

          
            if any(pd.isna([feat[c] for c in lag_cols])):
                item_mean_by_month = m[m['Item Code']==item].groupby('month')['y'].mean()
                fallback = item_mean_by_month.get(feat['month'], item_mean_by_month.mean())
                for k in [1,2,3,6,12]:
                    feat[f'lag_{k}'] = fallback
                feat['rolling_3'] = fallback
                feat['rolling_6'] = fallback

            fvec = pd.DataFrame([feat])[features]
            yhat = float(final_model.predict(fvec)[0])
            recent = pd.concat([recent, pd.DataFrame({'ds':[feat['ds']], 'y':[yhat]})], ignore_index=True)
            future_rows.append({'Customer Code':cust,'Item Code':item,'ds':feat['ds'],'yhat_monthly':yhat})

    future = pd.DataFrame(future_rows)
    future = future.sort_values(['Customer Code','Item Code','ds'])
    # Aggregate to Q and Y
    future['quarter'] = future['ds'].dt.to_period('Q')
    future['year'] = future['ds'].dt.year
    quarterly = (future.groupby(['Customer Code','Item Code','quarter'], as_index=False)['yhat_monthly']
                       .sum().rename(columns={'yhat_monthly':'yhat_quarterly'}))
    yearly = (future.groupby(['Customer Code','Item Code','year'], as_index=False)['yhat_monthly']
                     .sum().rename(columns={'yhat_monthly':'yhat_yearly'}))

    # Save
    future.rename(columns={'ds':'MonthStart'}, inplace=True)
    future.to_csv('forecast_monthly.csv', index=False)
    quarterly.to_csv('forecast_quarterly.csv', index=False)
    yearly.to_csv('forecast_yearly.csv', index=False)

    
    total_hist = (m.groupby('ds', as_index=False)['y'].sum()
                    .rename(columns={'ds':'MonthStart','y':'Total_Net'})
                    .sort_values('MonthStart'))
    total_fore = (future.groupby('MonthStart', as_index=False)['yhat_monthly'].sum()
                        .rename(columns={'yhat_monthly':'Total_Forecast'}))
    total_hist.to_csv('viz_total_history.csv', index=False)
    total_fore.to_csv('viz_total_forecast.csv', index=False)

    pd.DataFrame({
        'r2_overall':[r2],
        'model':[model_name],
        'train_rows':[len(train)],
        'valid_rows':[len(valid)],
        'total_rows':[len(X)]
    }).to_csv('run_summary.csv', index=False)

    print('Done. Files: eda_summary.csv, validation_metrics.csv, run_summary.csv, forecast_monthly.csv, forecast_quarterly.csv, forecast_yearly.csv, viz_total_history.csv, viz_total_forecast.csv, validation_r2_by_series.csv')

if __name__ == '__main__':
    main()
