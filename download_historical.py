from retrieve_ned import get_historical_data
from config import DOWNLOADED_DIR, HISTORICAL_DIR

def download_historical_data():
    """Download historical data using the NED API."""
    print("Starting download of historical data...")
    
    try:
        # Download historical data
        df = get_historical_data()
        print(f"Successfully downloaded data to {HISTORICAL_DIR}")
        print(f"Data range: {df.index.min()} to {df.index.max()}")
    except Exception as e:
        print(f"Error downloading data: {str(e)}")
        print("Make sure you've set your NED API key with:")
        print("export ned_api_key=your-key-here")

if __name__ == "__main__":
    download_historical_data() 