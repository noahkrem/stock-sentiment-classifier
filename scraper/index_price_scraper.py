import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

def clean_local_csv(file_path, date_col, value_col, index_name):
    df = pd.read_csv(file_path, parse_dates=[date_col])
    
    # just making sure columns are named consistently
    df = df.rename(columns={date_col: 'Date', value_col: index_name})
    
    # Drop rows where the value is missing or '.'
    df[index_name] = pd.to_numeric(df[index_name], errors='coerce')
    df = df.dropna(subset=['Date', index_name])
    
    # set date as index to make merging easier
    return df[['Date', index_name]].set_index('Date')

def main():
    nasdaq_file = 'NASDAQNDXT_2006_2026.csv'
    vix_file = 'VIXCLS_1990_2026.csv'
    vxazn_file = 'VXAZN_History.csv'
    
    df_nasdaq = clean_local_csv(nasdaq_file, 'observation_date', 'NASDAQNDXT', 'NASDAQNDXT')
    df_vix = clean_local_csv(vix_file, 'observation_date', 'VIXCLS', 'VIXCLS')
    df_vxazn = clean_local_csv(vxazn_file, 'DATE', 'CLOSE', 'VXAZN')
    
    # start and end date to match eriks
    start_date = df_vxazn.index.min()
    end_date = df_vxazn.index.max()
    
    live_tickers = ['^VXN', '^VIX3M']
    live_data_list = []
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    for ticker in live_tickers:
        raw_live = yf.download(ticker, start=start_str, end=end_str, progress=False)
        
        if raw_live.empty:
            print(f"Warning: No data found for {ticker}. Check symbol or internet connection.")
            continue
                      
        live_series = raw_live['Close'].copy()
        live_series = live_series.squeeze()
        live_df = pd.DataFrame({ticker: live_series})
        live_data_list.append(live_df)
    
    # Start the final dataframe with a complete business day index to catch gaps
    all_business_days = pd.date_range(start=start_date, end=end_date, freq='B')
    total_df = pd.DataFrame(index=all_business_days)
    total_df.index.name = 'Date'
    
    # Chronologically join every feature group
    datasets = [df_nasdaq, df_vix, df_vxazn] + live_data_list
    for dataset in datasets:
        total_df = total_df.join(dataset, how='left')
        
    # Handle missing data 
    # ffill carries the last known price across holidays/weekends
    # bfill handles any structural gaps at the very start of the timeline
    total_df = total_df.ffill().bfill()
    
    # Output Result in Repo Structure layout
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / 'data' / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'indexes_raw.csv'
    
    total_df.reset_index().to_csv(output_file, index=False)
    print(f"Successfully compiled {len(total_df)} market rows.")
    print(total_df.head(10))

if __name__ == "__main__":
    main()