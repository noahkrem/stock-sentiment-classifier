#!/usr/bin/env python3
"""
paper_trade.py

Automates paper trading simulation starting from Jan 1, 2026.
Runs weekly predictions using model.pkl, simulates Buy/Sell/Hold actions,
and compares total portfolio return against a passive Buy & Hold benchmark.

Usage:
    python paper_trade.py --data data/processed/labeled.csv --model models/model.pkl --initial-cash 1000 --trade-amount 1000
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd


def load_model_bundle(model_path: Path) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Run src/train.py first.")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def run_paper_trading(
    data_path: Path,
    model_path: Path,
    start_date: str = "2026-01-01",
    initial_cash: float = 1000.0,
    trade_amount: float = 1000.0,
):
    # 1. Load trained model bundle
    bundle = load_model_bundle(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_columns"] # Expected: ['nasdaqndxt', 'vix', 'vxazn', 'vxn', 'vix3m']
    classes = list(bundle["classes"])
    threshold = bundle.get("confidence_threshold", 0.35)

    # 2. Load dataset
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Filter data from start_date onwards
    df_sim = df[df["date"] >= pd.to_datetime(start_date)].copy().reset_index(drop=True)

    if df_sim.empty:
        print(f"No data found on or after {start_date}. Check your CSV dates.")
        return

    # Verify required feature columns exist
    missing_cols = [c for c in feature_cols if c not in df_sim.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required feature columns: {missing_cols}")

    # Check for AMZN price column for trade calculation
    price_col = None
    for col in ["amzn_close", "close", "amzn"]:
        if col in df_sim.columns:
            price_col = col
            break

    # 3. Paper Trading Variables
    cash = initial_cash
    shares = 0.0
    
    # Benchmark tracking (Buy & Hold with initial cash)
    initial_price = df_sim.iloc[0][price_col] if price_col else None
    benchmark_shares = (initial_cash / initial_price) if initial_price else 0.0

    trade_log = []

    print(f"\n{'='*75}")
    print(f" PAPER TRADING SIMULATION (Starting {start_date})")
    print(f" Model: {bundle.get('model_name', 'Trained Model')} | Confidence Threshold: {threshold:.2f}")
    print(f" Initial Cash: ${initial_cash:,.2f} | Trade Allocation: ${trade_amount:,.2f}")
    print(f"{'='*75}\n")

    # 4. Simulation Loop
    for idx, row in df_sim.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")
        current_price = row[price_col] if price_col else 1.0 # fallback if no price column

        # Extract the 5 feature values
        features_raw = row[feature_cols].values.reshape(1, -1)
        
        # Scale & predict
        features_scaled = scaler.transform(features_raw)
        probas = model.predict_proba(features_scaled)[0]
        
        raw_pred = classes[int(np.argmax(probas))]
        confidence = float(probas.max())

        # Apply confidence gating
        if raw_pred in ("Negative", "Positive") and confidence < threshold:
            signal = "Neutral"
        else:
            signal = raw_pred

        # Execute Trading Logic
        action = "HOLD"
        shares_bought_sold = 0.0

        if signal == "Positive":
            # BUY $trade_amount worth of shares (or remaining cash if less)
            buy_capital = min(cash, trade_amount)
            if buy_capital > 0:
                shares_bought_sold = buy_capital / current_price
                shares += shares_bought_sold
                cash -= buy_capital
                action = f"BUY ({shares_bought_sold:.2f} shrs)"
            else:
                action = "BUY (No Cash Left)"

        elif signal == "Negative":
            # SELL all currently held shares
            if shares > 0:
                cash_gained = shares * current_price
                cash += cash_gained
                shares_bought_sold = shares
                shares = 0.0
                action = f"SELL ({shares_bought_sold:.2f} shrs)"
            else:
                action = "SELL (No Shares)"

        elif signal == "Neutral":
            action = "HOLD"

        portfolio_value = cash + (shares * current_price)

        trade_log.append({
            "date": date_str,
            "signal": signal,
            "confidence": confidence,
            "action": action,
            "price": current_price,
            "cash": cash,
            "shares": shares,
            "total_value": portfolio_value
        })

        print(f"[{date_str}] Signal: {signal:8s} (Conf: {confidence:.2f}) | Price: ${current_price:7.2f} | Action: {action:20s} | Value: ${portfolio_value:,.2f}")

    # 5. Performance Summary
    final_row = trade_log[-1]
    final_value = final_row["total_value"]
    strategy_return = ((final_value - initial_cash) / initial_cash) * 100

    print(f"\n{'='*75}")
    print(" SIMULATION RESULTS SUMMARY")
    print(f"{'='*75}")
    print(f"Initial Balance        : ${initial_cash:,.2f}")
    print(f"Final Strategy Value   : ${final_value:,.2f} ({strategy_return:+.2f}%)")

    if price_col:
        final_price = df_sim.iloc[-1][price_col]
        benchmark_value = benchmark_shares * final_price
        benchmark_return = ((benchmark_value - initial_cash) / initial_cash) * 100
        outperformance = strategy_return - benchmark_return

        print(f"Buy & Hold Value       : ${benchmark_value:,.2f} ({benchmark_return:+.2f}%)")
        print(f"Strategy Alpha         : {outperformance:+.2f}%")
    
    print(f"{'='*75}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run paper trading backtest using trained model.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/labeled.csv"), help="Path to labeled data CSV")
    parser.add_argument("--model", type=Path, default=Path("models/model.pkl"), help="Path to model bundle pkl")
    parser.add_argument("--start-date", type=str, default="2026-01-01", help="Start date for simulation (YYYY-MM-DD)")
    parser.add_argument("--initial-cash", type=float, default=1000.0, help="Starting cash amount")
    parser.add_argument("--trade-amount", type=float, default=1000.0, help="Amount to allocate per BUY signal")

    args = parser.parse_args()
    run_paper_trading(
        data_path=args.data,
        model_path=args.model,
        start_date=args.start_date,
        initial_cash=args.initial_cash,
        trade_amount=args.trade_amount,
    )
