#!/usr/bin/env python
"""
Ocean Transport Data Processing Script

This script processes ocean transport data from multiple sources:
- AMOC (Atlantic Meridional Overturning Circulation) data
- ITF (Indonesian Throughflow) geostrophic transport data  
- Agulhas Current transport data

Author: Sandy Herho <sandy.herho@email.ucr.edu>
Date: May 26, 2025
"""

import numpy as np
import pandas as pd
import xarray as xr
import scipy.io as sio


def main():
    """Main function to process ocean transport datasets."""
    
    # =========================================================================
    # AMOC Data Processing
    # =========================================================================
    print("Processing AMOC data...")
    
    # Load AMOC dataset and convert units from Sv to Sv*1e-6
    ds_amoc = xr.open_dataset("../data/raw_data/GLOBAL_OMI_NATLANTIC_amoc_max26N_timeseries.nc") * 1e-6
    
    # Create time array for AMOC data (1993-2024)
    t_amoc = np.arange("1993-01-15", "2024-01-15", dtype='datetime64[M]')
    
    # Extract AMOC mean values
    amoc = ds_amoc["amoc_mean"].to_numpy()
    
    # Create DataFrame and save to CSV
    amoc_transport = pd.DataFrame({"time": t_amoc, "amoc": amoc})
    amoc_transport.to_csv("../data/raw_data/amoc_ts.csv", index=False)
    print(f"AMOC data saved: {len(amoc_transport)} records")
    
    # =========================================================================
    # ITF Data Processing  
    # =========================================================================
    print("Processing ITF data...")
    
    # Load ITF geostrophic transport data from MATLAB file
    itf_data = sio.loadmat("../data/raw_data/ITF_Geostrophic_transport19842017_monthly.mat")
    
    # Calculate median transport values across spatial dimensions
    itf_g = np.median(itf_data["ITF_G19842017_monthly"], axis=1)  # Geostrophic
    itf_t = np.median(itf_data["ITF_T19842017_monthly"], axis=1)  # Total
    itf_s = np.median(itf_data["ITF_S19842017_monthly"], axis=1)  # Shallow
    
    # Create time array for ITF data (1984-2018)
    t_itf = np.arange("1984-01-15", "2018-01-15", dtype='datetime64[M]')
    
    # Create DataFrame and save to CSV
    itf_transport = pd.DataFrame({
        "time": t_itf, 
        "itf_g": itf_g,
        "itf_t": itf_t, 
        "itf_s": itf_s
    })
    itf_transport.to_csv("../data/raw_data/itf_ts.csv", index=False)
    print(f"ITF data saved: {len(itf_transport)} records")
    
    # =========================================================================
    # Agulhas Current Data Processing
    # =========================================================================
    print("Processing Agulhas Current data...")
    
    # Load and resample Agulhas Current transport data to monthly means
    ds_aghulas = (xr.open_dataset("../data/raw_data/ACTtransport_proxy.nc")
                  .resample(time='ME').mean()) * 1e-6
    
    # Create time array for Agulhas data (1992-2015)
    t_aghulas = np.arange("1992-09-15", "2015-01-15", dtype='datetime64[M]')
    
    # Extract transport components
    aghulas_box = ds_aghulas['Tbox'].to_numpy()
    aghulas_jet = ds_aghulas['Tjet'].to_numpy()
    
    # Create DataFrame and save to CSV
    aghulas_transport = pd.DataFrame({
        "time": t_aghulas, 
        "aghulas_box": aghulas_box, 
        "aghulas_jet": aghulas_jet
    })
    aghulas_transport.to_csv("../data/raw_data/aghulas_ts.csv", index=False)
    print(f"Agulhas data saved: {len(aghulas_transport)} records")
    
    print("\nAll ocean transport datasets processed successfully!")


if __name__ == "__main__":
    main()
