import pandas as pd
from pathlib import Path

class DataManager:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.initialize_csv()
    
    def initialize_csv(self):
        """Initialize CSV file if it doesn't exist"""
        if not Path(self.csv_file).exists():
            columns = ['timestamp', 'shoulder_width', 'hip_width', 
                      'right_leg_length', 'left_leg_length']
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.csv_file, index=False)
    
    def save_measurements(self, measurements):
        """Save measurements to CSV file"""
        df = pd.DataFrame([measurements])
        
        # Append to CSV without writing headers if file exists
        df.to_csv(self.csv_file, mode='a', header=False, index=False)
    
    def get_measurements_history(self):
        """Get all historical measurements"""
        return pd.read_csv(self.csv_file)
