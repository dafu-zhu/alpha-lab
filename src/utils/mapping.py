import requests


def symbol_cik_mapping() -> dict:
        # Download the official ticker-CIK mapping file
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {'User-Agent': 'YourName/YourEmail@domain.com'} # SEC requires a proper User-Agent
        response = requests.get(url, headers=headers)

        symbol_cik = {}
        if response.status_code == 200:
            data = response.json()
            for key, value in data.items():
                symbol = value.get('ticker')
                if symbol:
                    symbol_cik[symbol] = value.get('cik_str')
        
        return symbol_cik

if __name__ == "__main__":
    mp = symbol_cik_mapping()
    print(mp.keys())