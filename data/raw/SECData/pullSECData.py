import time
import requests
import pandas as pd

# Identify yourself. The SEC fair-access policy requires a descriptive
# User-Agent with a contact. Use your real app name + email.
HEADERS = {"User-Agent": "StockClassifier/1.0 (jsp29@sfu.ca)"}

def get_ticker_cik_map():
    """Download the official ticker -> CIK map."""
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    # Keys are arbitrary indices; each value has cik_str, ticker, title.
    return {row["ticker"].upper(): row["cik_str"] for row in resp.json().values()}

def cik_padded(cik_int):
    """EDGAR requires the CIK zero-padded to 10 digits."""
    return f"CIK{int(cik_int):010d}"

def get_company_facts(cik_int):
        """Fetch all XBRL concepts for a company."""
        url = f"https://data.sec.gov/api/xbrl/companyfacts/{cik_padded(cik_int)}.json"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()
        

if __name__ == "__main__":

    # 1. Map AMZN Ticker to CIK
    tickers = get_ticker_cik_map()
    cik = tickers["AMZN"]
    print(f"AMZN CIK: {cik} -> {cik_padded(cik)}")

    time.sleep(0.1)  # stay under 10 req/s

    # 2. Download Company Facts JSON
    facts_data = get_company_facts(cik)
    us_gaap_facts = facts_data.get("facts", {}).get("us-gaap", {})

    # Target tags to extract
    target_tags = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "ContractWithCustomerLiabilityRevenueRecognized",
        "OperatingIncomeLoss",
    ]

    target_fields = ["start", "end", "val", "form", "frame"]
    extracted_records = []

    # 3. Iterate through target concepts and extract values
    for tag in target_tags:
        if tag in us_gaap_facts:
            units = us_gaap_facts[tag].get("units", {})

            # Financial figures are typically categorized under 'USD'
            for currency, entries in units.items():
                for entry in entries:
                    record = {"Tag": tag}

                    # Extract requested keys; set to "None" if key is missing/labeled
                    for field in target_fields:
                        val = entry.get(field)
                        record[field] = "None" if val is None else val

                    extracted_records.append(record)
        else:
            print(f"Warning: Tag '{tag}' not found in AMZN facts database.")

    # 4. Process and Export to CSV
    df = pd.DataFrame(extracted_records)

    # Ensure column ordering: Tag, start, end, val, form, frame
    columns_order = ["Tag"] + target_fields
    df = df[columns_order]

    output_filename = "AMZN_Financial_Metrics.csv"
    df.to_csv(output_filename, index=False)

    print(
        f"Extraction complete! Successfully saved {len(df)} records to '{output_filename}'."
    )
    print("\nPreview of extracted data:")
    print(df.head(15).to_string(index=False))
