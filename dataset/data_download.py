import os
import json
import pandas as pd
import fastf1
import warnings
import argparse

json_file_path = os.path.join(os.path.dirname(__file__), 'download_path.json')
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file) 

def dir_setup():
    
    os.makedirs(data['CACHE_DIR'], exist_ok=True)
    os.makedirs(data['DATA_DIR'], exist_ok=True)
    fastf1.Cache.enable_cache(data['CACHE_DIR'])

def single_year_download(year, race, session_type):
    print(f"Start Download {year}-{race}-{session_type}")
    session = fastf1.get_session(year, race, session_type)
    session.load(telemetry=True, laps=True)
    data_to_csv(session, year=year, race=race, session_type=session_type)


def multiple_years_download(year_range, race, session_type):
    
    print(f"Start Download {year_range}-{race}-{session_type}")
    start_year, end_year = year_range.split('-')
    print(f"Start Year: {start_year}, End Year: {end_year}")
    for year in range(int(start_year), int(end_year) + 1):
        session = fastf1.get_session(year, race, session_type)
        session.load(telemetry=True, laps=True)
        data_to_csv(session, year=year, year_range=year_range, race=race, session_type=session_type)



def data_to_csv(session, year_range=None, year=None, race=None, session_type=None):
    
    driver_abbrs = session.results['Abbreviation'].tolist()
    
    for abbr in driver_abbrs:
        laps_info = session.laps.pick_drivers(abbr)
        all_telemetry = []
 
        for _, lap in laps_info.iterrows():
            try:
                info = lap.get_telemetry()
                if info is not None and len(info) > 0:
                    info_table = info.copy()
                    info_table['LapNumber'] = lap['LapNumber']
                    info_table['Driver'] = abbr
                    info_table['IsAccurate'] = lap.get('IsAccurate', True)
                    all_telemetry.append(info_table)
            except Exception:
                continue
 
        if all_telemetry:
            driver_df = pd.concat(all_telemetry, ignore_index=True)

            for col in driver_df.columns:
                
                if driver_df[col].dtype == 'timedelta64[ns]':
                    driver_df[col] = driver_df[col].astype(str)
                elif driver_df[col].dtype == 'datetime64[ns]':
                    driver_df[col] = driver_df[col].astype(str)

            filepath = os.path.join(data["DATA_DIR"], f"{year}_{race}_{session_type}_{abbr}.csv")
            driver_df.to_csv(filepath, index=False)
            print(f"{abbr}: download success ")
    
    
        

def main():
    
    parser = argparse.ArgumentParser(description="Download F1 telemetry data.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--year_range", type=str, default="None")
    parser.add_argument("--race", type=str, default="Monza")
    parser.add_argument("--session", type=str, default="R")

    args = parser.parse_args()
    
    dir_setup()
    
    # Start Download the dataset
    
    if args.year_range == "None":
        single_year_download(args.year, args.race, args.session)
    else:   
        multiple_years_download(args.year_range, args.race, args.session)
    
    print(f"\nDownload complete.")

if __name__ == '__main__':
    main()
