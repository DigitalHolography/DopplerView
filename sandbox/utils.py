import pandas as pd
import os

def dataframe_from_dataset(dataset):
    df = pd.DataFrame(columns=["person", "nb_measure", "R", "L", "unknown"])
    for measure in os.listdir(dataset):
        if not os.path.isdir(os.path.join(dataset, measure)):
            continue
        
        parts = measure.split('_')
        if '' in parts:
            parts.remove('')
    
        if len(parts) == 5:
            date, name, eye, HD, processing_number = parts[:5]            
        elif len(parts) == 6:
            date, name, eye, measure_number, HD, processing_number = parts[:6]
    
        if not name in df["person"].values:
            df.loc[len(df)] = {"person": name, "nb_measure": 0, "R":0, "L":0, "unknown":0}
        
        df.loc[df["person"] == name, "nb_measure"] += 1
    
        if "OD" in eye.upper() or "R" in eye.upper():
            df.loc[df["person"] == name, "R"] += 1
        elif "OS" in eye.upper() or "L" in eye.upper():
            df.loc[df["person"] == name, "L"] += 1
        else:
            df.loc[df["person"] == name, "unknown"] += 1
            
    return df.sort_values(by='nb_measure', ascending=False)