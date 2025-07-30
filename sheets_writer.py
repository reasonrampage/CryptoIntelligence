"""
Free Google Sheets Integration
Uses Google Sheets API (free with usage limits)
- 100 requests per 100 seconds per user
- 300 requests per 100 seconds per project
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import time

# Google Sheets API (free)
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    print("Google API libraries not available. Install with: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")

class FreeGoogleSheetsWriter:
    """Writes data to Google Sheets using free API."""
    
    def __init__(self, credentials_file: str, spreadsheet_id: str):
        if not GOOGLE_API_AVAILABLE:
            raise ImportError("Google API libraries not installed")
        
        self.spreadsheet_id = spreadsheet_id
        self.credentials_file = credentials_file
        self.service = None
        
        # API rate limiting for free tier
        self.requests_per_100_seconds = 90  # Stay under 100 limit
        self.request_times = []
        
        # Scopes needed for Sheets API
        self.scopes = ['https://www.googleapis.com/auth/spreadsheets']
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize the service
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Sheets API."""
        creds = None
        
        try:
            # Try service account authentication (recommended for production)
            if self.credentials_file.endswith('.json'):
                creds = ServiceAccountCredentials.from_service_account_file(
                    self.credentials_file, scopes=self.scopes)
                self.logger.info("Authenticated using service account")
            else:
                self.logger.error("Please provide a service account JSON file")
                return
        
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            return
        
        try:
            self.service = build('sheets', 'v4', credentials=creds)
            self.logger.info("Google Sheets service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to build Sheets service: {e}")
    
    def _rate_limit(self):
        """Implement rate limiting for free API tier."""
        current_time = time.time()
        
        # Remove requests older than 100 seconds
        self.request_times = [t for t in self.request_times if current_time - t < 100]
        
        # If we've made too many requests, wait
        if len(self.request_times) >= self.requests_per_100_seconds:
            sleep_time = 100 - (current_time - self.request_times[0])
            if sleep_time > 0:
                self.logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        self.request_times.append(current_time)
    
    def _make_request(self, request_func, *args, **kwargs):
        """Make a rate-limited request to Google Sheets API."""
        if not self.service:
            self.logger.error("Google Sheets service not initialized")
            return None
        
        self._rate_limit()
        
        try:
            return request_func(*args, **kwargs)
        except HttpError as error:
            self.logger.error(f"Google Sheets API error: {error}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return None
    
    def create_worksheet(self, sheet_name: str) -> bool:
        """Create a new worksheet in the spreadsheet."""
        self.logger.info(f"Creating worksheet: {sheet_name}")
        
        requests = [{
            'addSheet': {
                'properties': {
                    'title': sheet_name
                }
            }
        }]
        
        body = {'requests': requests}
        
        result = self._make_request(
            self.service.spreadsheets().batchUpdate,
            spreadsheetId=self.spreadsheet_id,
            body=body
        )
        
        return result is not None
    
    def get_worksheet_names(self) -> List[str]:
        """Get list of all worksheet names in the spreadsheet."""
        result = self._make_request(
            self.service.spreadsheets().get,
            spreadsheetId=self.spreadsheet_id
        )
        
        if not result:
            return []
        
        sheet_names = []
        for sheet in result.get('sheets', []):
            sheet_names.append(sheet['properties']['title'])
        
        return sheet_names
    
    def clear_worksheet(self, sheet_name: str) -> bool:
        """Clear all data from a worksheet."""
        self.logger.info(f"Clearing worksheet: {sheet_name}")
        
        result = self._make_request(
            self.service.spreadsheets().values().clear,
            spreadsheetId=self.spreadsheet_id,
            range=f"{sheet_name}!A:ZZ",
            body={}
        )
        
        return result is not None
    
    def write_dataframe(self, df: pd.DataFrame, sheet_name: str, 
                       start_cell: str = 'A1', include_index: bool = False) -> bool:
        """Write a pandas DataFrame to Google Sheets."""
        if df.empty:
            self.logger.warning(f"DataFrame is empty, skipping write to {sheet_name}")
            return False
        
        self.logger.info(f"Writing {len(df)} rows to {sheet_name}")
        
        # Prepare data for Google Sheets
        if include_index:
            df_to_write = df.reset_index()
        else:
            df_to_write = df.copy()
        
        # Convert DataFrame to list of lists
        values = []
        
        # Add headers
        headers = df_to_write.columns.tolist()
        values.append(headers)
        
        # Add data rows
        for _, row in df_to_write.iterrows():
            row_values = []
            for value in row:
                if pd.isna(value):
                    row_values.append('')
                elif isinstance(value, (datetime, pd.Timestamp)):
                    row_values.append(value.strftime('%Y-%m-%d %H:%M:%S'))
                elif isinstance(value, (int, float)):
                    if np.isfinite(value):
                        row_values.append(value)
                    else:
                        row_values.append('')
                else:
                    row_values.append(str(value))
            values.append(row_values)
        
        # Prepare the request
        range_name = f"{sheet_name}!{start_cell}"
        body = {
            'values': values,
            'majorDimension': 'ROWS'
        }
        
        # Write to Google Sheets
        result = self._make_request(
            self.service.spreadsheets().values().update,
            spreadsheetId=self.spreadsheet_id,
            range=range_name,
            valueInputOption='USER_ENTERED',
            body=body
        )
        
        if result:
            self.logger.info(f"Successfully wrote {len(values)} rows to {sheet_name}")
            return True
        else:
            return False
    
    def append_dataframe(self, df: pd.DataFrame, sheet_name: str, 
                        include_index: bool = False) -> bool:
        """Append a pandas DataFrame to existing data in Google Sheets."""
        if df.empty:
            self.logger.warning(f"DataFrame is empty, skipping append to {sheet_name}")
            return False
        
        self.logger.info(f"Appending {len(df)} rows to {sheet_name}")
        
        # Prepare data
        if include_index:
            df_to_write = df.reset_index()
        else:
            df_to_write = df.copy()
        
        # Convert to list of lists (no headers for append)
        values = []
        for _, row in df_to_write.iterrows():
            row_values = []
            for value in row:
                if pd.isna(value):
                    row_values.append('')
                elif isinstance(value, (datetime, pd.Timestamp)):
                    row_values.append(value.strftime('%Y-%m-%d %H:%M:%S'))
                elif isinstance(value, (int, float)):
                    if np.isfinite(value):
                        row_values.append(value)
                    else:
                        row_values.append('')
                else:
                    row_values.append(str(value))
            values.append(row_values)
        
        # Append to sheet
        range_name = f"{sheet_name}!A:A"
        body = {
            'values': values,
            'majorDimension': 'ROWS'
        }
        
        result = self._make_request(
            self.service.spreadsheets().values().append,
            spreadsheetId=self.spreadsheet_id,
            range=range_name,
            valueInputOption='USER_ENTERED',
            insertDataOption='INSERT_ROWS',
            body=body
        )
        
        if result:
            self.logger.info(f"Successfully appended {len(values)} rows to {sheet_name}")
            return True
        else:
            return False
    
    def format_worksheet(self, sheet_name: str, header_row: bool = True) -> bool:
        """Apply basic formatting to a worksheet."""
        self.logger.info(f"Formatting worksheet: {sheet_name}")
        
        # Get sheet ID
        spreadsheet = self._make_request(
            self.service.spreadsheets().get,
            spreadsheetId=self.spreadsheet_id
        )
        
        if not spreadsheet:
            return False
        
        sheet_id = None
        for sheet in spreadsheet.get('sheets', []):
            if sheet['properties']['title'] == sheet_name:
                sheet_id = sheet['properties']['sheetId']
                break
        
        if sheet_id is None:
            self.logger.error(f"Sheet {sheet_name} not found")
            return False
        
        requests = []
        
        if header_row:
            # Format header row
            requests.append({
                'repeatCell': {
                    'range': {
                        'sheetId': sheet_id,
                        'startRowIndex': 0,
                        'endRowIndex': 1
                    },
                    'cell': {
                        'userEnteredFormat': {
                            'backgroundColor': {
                                'red': 0.9,
                                'green': 0.9,
                                'blue': 0.9
                            },
                            'textFormat': {
                                'bold': True
                            }
                        }
                    },
                    'fields': 'userEnteredFormat(backgroundColor,textFormat)'
                }
            })
        
        # Auto-resize columns
        requests.append({
            'autoResizeDimensions': {
                'dimensions': {
                    'sheetId': sheet_id,
                    'dimension': 'COLUMNS',
                    'startIndex': 0,
                    'endIndex': 26  # A-Z columns
                }
            }
        })
        
        body = {'requests': requests}
        
        result = self._make_request(
            self.service.spreadsheets().batchUpdate,
            spreadsheetId=self.spreadsheet_id,
            body=body
        )
        
        return result is not None
    
    def create_crypto_intelligence_sheets(self) -> bool:
        """Create all required sheets for crypto intelligence tool."""
        required_sheets = [
            'MarketData',
            'NewsSentiment', 
            'TechnicalIndicators',
            'OpportunityScores',
            'Leaderboard',
            'GlobalMetrics'
        ]
        
        existing_sheets = self.get_worksheet_names()
        
        for sheet_name in required_sheets:
            if sheet_name not in existing_sheets:
                if self.create_worksheet(sheet_name):
                    self.format_worksheet(sheet_name)
                    self.logger.info(f"Created and formatted sheet: {sheet_name}")
                else:
                    self.logger.error(f"Failed to create sheet: {sheet_name}")
                    return False
            else:
                self.logger.info(f"Sheet already exists: {sheet_name}")
        
        return True
    
    def write_market_data(self, market_df: pd.DataFrame) -> bool:
        """Write market data to the MarketData sheet."""
        return self.write_dataframe(market_df, 'MarketData')
    
    def write_sentiment_data(self, sentiment_df: pd.DataFrame) -> bool:
        """Write sentiment analysis data to the NewsSentiment sheet."""
        return self.write_dataframe(sentiment_df, 'NewsSentiment')
    
    def write_technical_data(self, technical_df: pd.DataFrame) -> bool:
        """Write technical indicators to the TechnicalIndicators sheet."""
        return self.write_dataframe(technical_df, 'TechnicalIndicators')
    
    def write_opportunity_scores(self, opportunity_df: pd.DataFrame) -> bool:
        """Write opportunity scores to the OpportunityScores sheet."""
        return self.write_dataframe(opportunity_df, 'OpportunityScores')
    
    def write_leaderboard(self, leaderboard_df: pd.DataFrame) -> bool:
        """Write leaderboard to the Leaderboard sheet."""
        return self.write_dataframe(leaderboard_df, 'Leaderboard')
    
    def write_global_metrics(self, metrics_dict: Dict[str, Any]) -> bool:
        """Write global market metrics."""
        # Convert metrics dict to DataFrame
        metrics_df = pd.DataFrame([{
            'metric': key,
            'value': value,
            'timestamp': datetime.utcnow()
        } for key, value in metrics_dict.items()])
        
        return self.write_dataframe(metrics_df, 'GlobalMetrics')
    
    def get_api_usage_info(self) -> Dict[str, int]:
        """Get information about API usage."""
        return {
            'requests_made_last_100s': len(self.request_times),
            'remaining_requests': self.requests_per_100_seconds - len(self.request_times),
            'rate_limit': self.requests_per_100_seconds
        }


def main():
    """Example usage of the Google Sheets writer."""
    # This is a demo - you'll need actual credentials and spreadsheet ID
    
    # Example configuration (replace with your actual values)
    CREDENTIALS_FILE = 'path/to/your/service-account-credentials.json'
    SPREADSHEET_ID = 'your-google-sheets-id-here'
    
    try:
        # Initialize the sheets writer
        sheets_writer = FreeGoogleSheetsWriter(CREDENTIALS_FILE, SPREADSHEET_ID)
        
        # Create required sheets
        sheets_writer.create_crypto_intelligence_sheets()
        
        # Example: Write sample market data
        sample_market_data = pd.DataFrame({
            'coin_id': ['bitcoin', 'ethereum', 'cardano'],
            'symbol': ['BTC', 'ETH', 'ADA'],
            'current_price': [45000.0, 3000.0, 1.2],
            'market_cap': [850000000000, 360000000000, 38000000000],
            'volume_24h': [25000000000, 15000000000, 800000000],
            'price_change_24h': [2.5, -1.2, 5.8],
            'timestamp': datetime.utcnow()
        })
        
        success = sheets_writer.write_market_data(sample_market_data)
        
        if success:
            print("✅ Successfully wrote market data to Google Sheets")
        else:
            print("❌ Failed to write market data")
        
        # Show API usage
        usage = sheets_writer.get_api_usage_info()
        print(f"\nAPI Usage: {usage['requests_made_last_100s']}/{usage['rate_limit']} requests used")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo use this tool, you need to:")
        print("1. Create a Google Cloud Project")
        print("2. Enable the Google Sheets API")
        print("3. Create a service account and download credentials JSON")
        print("4. Create a Google Sheets document and share it with the service account email")
        print("5. Update the CREDENTIALS_FILE and SPREADSHEET_ID variables")


if __name__ == "__main__":
    main()