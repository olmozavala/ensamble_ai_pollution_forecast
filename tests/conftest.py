import pytest
import os
from os.path import join
import tempfile
import pandas as pd
import numpy as np
import xarray as xr

@pytest.fixture
def test_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def sample_pollution_data(test_data_dir):
    """Create sample pollution CSV data for testing."""
    # Create a simple DataFrame with test data
    dates = pd.date_range('2010-01-01', '2010-01-02', freq='H')
    data = {
        'cont_co_PED': np.random.rand(len(dates)),
        'cont_nodos_PED': np.random.rand(len(dates)),
        'cont_otres_PED': np.random.rand(len(dates))
    }
    df = pd.DataFrame(data, index=dates)
    
    # Save to CSV in test directory
    csv_path = join(test_data_dir, 'pollution_2010.csv')
    df.to_csv(csv_path)
    return csv_path

@pytest.fixture
def sample_weather_data(test_data_dir):
    """Create sample weather NetCDF data for testing."""
    # Create a simple xarray Dataset with test data
    dates = pd.date_range('2010-01-01', '2010-01-02', freq='H')
    lats = np.linspace(19, 20, 25)
    lons = np.linspace(-99, -98, 25)
    
    data_vars = {
        'temperature': (('time', 'lat', 'lon'), 
                       np.random.rand(len(dates), len(lats), len(lons))),
        'humidity': (('time', 'lat', 'lon'), 
                    np.random.rand(len(dates), len(lats), len(lons)))
    }
    
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'time': dates,
            'lat': lats,
            'lon': lons
        }
    )
    
    # Save to NetCDF in test directory
    nc_path = join(test_data_dir, 'weather_2010.nc')
    ds.to_netcdf(nc_path)
    return nc_path 