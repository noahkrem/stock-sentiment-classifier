import requests
import time
import pandas as pd

HEADERS = {"User-Agent": "StockClassifier/1.0 (jsp29@sfu.ca)"}


def get_ticker_cik_map():
    """Download official ticker -> CIK map."""
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return {
        row["ticker"].upper(): row["cik_str"] for row in resp.json().values()
    }

def cik_padded(cik_int):
    """EDGAR requires CIK zero-padded to 10 digits."""
    return f"CIK{int(cik_int):010d}"


def get_company_facts(cik_int):
    """Fetch all XBRL concepts, labels, and descriptions for a company."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/{cik_padded(cik_int)}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    # 1. Map AMZN to CIK
    tickers = get_ticker_cik_map()
    cik = tickers["AMZN"]
    print(f"AMZN CIK: {cik} -> {cik_padded(cik)}\n")

    time.sleep(0.1)  # Respect SEC rate limit (< 10 req/s)

    # 2. Fetch company facts
    facts_data = get_company_facts(cik)

    # 3. Extract US-GAAP concepts along with their Label and Description
    us_gaap_facts = facts_data.get("facts", {}).get("us-gaap", {})

    concept_list = []
    for tag_name, details in us_gaap_facts.items():
        label = details.get("label") or "N/A"
        description = details.get("description")

        # Handle None values safely before running string methods
        if description:
            clean_description = (
                description.replace("\n", " ").replace("\r", " ").strip()
            )
        else:
            clean_description = "No description available."

        concept_list.append(
            {
                "Tag": tag_name,
                "Label": label,
                "Description": clean_description,
            }
        )

    # Convert to pandas DataFrame for pretty display
    df = pd.DataFrame(concept_list)

    print(f"Total US-GAAP tags found for AMZN: {len(df)}\n")

    # Display first 20 concepts as a preview
    pd.set_option("display.max_colwidth", 80)
    print(df.head(20).to_string(index=False))

    # Optional: Save complete tag directory to CSV
    df.to_csv("amzn_xbri_tags_directory.csv", index=False)
    
