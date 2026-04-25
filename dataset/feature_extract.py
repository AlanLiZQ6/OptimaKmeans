import os
import glob
import numpy as np
import pandas as pd
import warnings
import json
warnings.filterwarnings('ignore')

FEATURES = ['Speed', 'Throttle', 'Brake', 'RPM', 'nGear', 'X', 'Y']
MIN_SAMPLES = 5

json_file_path = os.path.join(os.path.dirname(__file__), 'download_path.json')
with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file) 
    
def feature_extract(dataframe):
    
    required_cols = FEATURES + ['Driver', 'LapNumber']
    if 'IsAccurate' in dataframe.columns:
        cols_to_keep = required_cols + ['IsAccurate']
    
    result = dataframe[cols_to_keep].copy()
    return result

def data_filter(dataframe):
        
    # Only choose the accurate data
    if 'IsAccurate' in dataframe.columns:
        dataframe = dataframe[dataframe['IsAccurate'] == True].copy()
        dataframe = dataframe.drop(columns=['IsAccurate'])
        print("Only the Accurate data has been selected !")
    
    # Set the index to Driver name and group the data
    # Then, filter the data with enough laps.
    lap_counts = dataframe.groupby(['Driver', 'LapNumber']).size()
    valid_laps = lap_counts[lap_counts >= MIN_SAMPLES].index
    dataframe = dataframe.set_index(['Driver', 'LapNumber']).loc[valid_laps].reset_index()
    print("Each Driver will have at least 5 laps data.")
    
    # Drop the NaN value
    original = len(dataframe)
    dataframe = dataframe.dropna(subset=FEATURES)
    if len(dataframe) != original:
        print(f"We have dropped {original - len(dataframe)} rows with NaN values.")
    
    return dataframe
    
   
def main():
    
    csv_files = sorted(glob.glob(os.path.join(json_data['DATA_DIR'], '*.csv')))
    
    csv_files_list = []
    
    for file in csv_files:
        csv_files_list.append(file)
     
    print(f"Found {len(csv_files)} in raw data dir:")
    
    all_data = []
    for file in csv_files_list:
        df = pd.read_csv(file)
        filename = os.path.basename(file)
        print(f" {filename}: {len(df)} samples")
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal raw samples: {len(combined)}")
    
    features_table = feature_extract(combined)
    
    filted_dataframe = data_filter(features_table)
    
    os.makedirs(json_data['EXTRACT_DIR'], exist_ok=True)
    features_path = os.path.join(json_data['EXTRACT_DIR'], 'telemetry_features.csv')
    filted_dataframe.to_csv(features_path, index=False)


    #filtered = filter_data(selected)
    #save_features(filtered, OUTPUT_DIR)
    print(f"\nFeature extraction complete.")
    print(f"\nFiltered Feature data saved to {json_data['EXTRACT_DIR']}/")

if __name__ == '__main__':
    main()
