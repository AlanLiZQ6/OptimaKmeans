import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
import json
warnings.filterwarnings('ignore')

FEATURES = ['Speed', 'Throttle', 'Brake', 'RPM', 'nGear', 'X', 'Y']

json_file_path = os.path.join(os.path.dirname(__file__), 'download_path.json')
with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file) 

def main():
    
    csv_data_path = os.path.join(json_data['EXTRACT_DIR'], 'telemetry_features.csv')
    data_frame = pd.read_csv(csv_data_path)
    data_raw = data_frame[FEATURES].values.astype(np.float64)
    nan_count = np.isnan(data_raw).sum()
    inf_count = np.isinf(data_raw).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"Detecting the invalid value !")
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_raw)
    
    os.makedirs(json_data['PROCESS_DIR'], exist_ok=True)
    
    extra_features = data_frame[['Driver', 'LapNumber']].copy()
    
    output_df = extra_features.copy()
    output_df[FEATURES] = data_scaled

    # Save as one CSV
    output_df.to_csv(os.path.join(json_data['PROCESS_DIR'], 'final_processed.csv'), index=False)
        
if __name__ == '__main__':

    main()
    