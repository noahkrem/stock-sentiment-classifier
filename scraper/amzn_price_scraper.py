# If you want to run the scraper on your local machine you will need to install yfinance for python 
import pandas as pd
import yfinance as yf 
import sys
from datetime import timedelta
from pathlib import Path

def main():
    # Load the existing index CSV
    # Accept a file path as a command-line argument
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'VXAZN_History.csv'

    repo_root = Path(__file__).resolve().parent.parent
    scraper_dir = Path(__file__).resolve().parent
    candidate_paths = []

    input_path = Path(csv_file)
    if input_path.is_absolute():
        candidate_paths.append(input_path)
    else:
        candidate_paths.extend([
            Path.cwd() / input_path,
            scraper_dir / input_path,
            repo_root / input_path,
        ])

    resolved_path = None
    for path in candidate_paths:
        if path.exists():
            resolved_path = path
            break

    if resolved_path is None:
        print(f"Error: File '{csv_file}' not found.")
        sys.exit(1)

    try:
        df_index = pd.read_csv(resolved_path, dayfirst=False)
    except FileNotFoundError:
        print(f"Error: File '{resolved_path}' not found.")
        sys.exit(1)

    df_index.columns = [col.strip() for col in df_index.columns]
    date_col = next((col for col in ['DATE', 'Date', 'date'] if col in df_index.columns), None)
    if date_col is None:
        print("Error: Could not find a date column in the input CSV. Expected 'DATE' or 'Date'.")
        sys.exit(1)

    df_index[date_col] = pd.to_datetime(df_index[date_col], errors='coerce')
    df_index = df_index.dropna(subset=[date_col]).copy()

    # Determine the full date span of your index data
    start_date = df_index[date_col].min()
    end_date = df_index[date_col].max()
    print(f"Index data range: {start_date.date()} to {end_date.date()}")

    # Download AMZN weekly data for the SAME period
    # Add a few days to end_date to include the last week
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = (end_date + timedelta(days=7)).strftime('%Y-%m-%d')

    print("Downloading AMZN weekly data...")
    amzn = yf.download('AMZN', start=start_str, end=end_str,
                        interval='1wk', progress=False)

    if amzn.empty:
        print("No data downloaded. Check your internet connection or ticker symbol.")
        sys.exit(1)

    # Keep only the adjusted close (column 'Close' with auto_adjust=True)
    close = amzn['Close'].copy()

    # Compute weekly percentage return
    weekly_return = close.pct_change() * 100

    # Ensure both are 1-dimensional (Series, not DataFrame)
    close_1d = close.squeeze()
    returns_1d = weekly_return.squeeze()

    # Combine into a clean DataFrame
    result = pd.DataFrame({
        'Date': close_1d.index,
        'AMZN_Close': close_1d,
        'Weekly_Return_Pct': returns_1d
    })

    # Drop the first row (NaN return) and keep only weeks within original range
    result = result.dropna(subset=['Weekly_Return_Pct'])
    result = result[result['Date'] <= pd.Timestamp(end_date)]

    # Save the label source under the repository's data/raw directory
    output_dir = repo_root / 'data' / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'amzn_weekly_returns.csv'
    result.to_csv(output_file, index=False)
    print(f"Saved {len(result)} weeks of AMZN returns to '{output_file}'")
    print(result.head(10))

if __name__ == "__main__":
    main()