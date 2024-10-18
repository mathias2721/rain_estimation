import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray
import os
import glob

MONTHES = [1, 2, 3]
ROOT = '/home/mathias/Desktop/data/rain'

def load_radar():
    return xr.concat([xr.open_dataset(f"{ROOT}/2010_{month}.nc") for month in MONTHES], dim="time")

#https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form
def load_era_pressure():
    nc_files = glob.glob(f'{ROOT}/era5_pressure/*.nc')
    data_array = xr.open_mfdataset(nc_files, combine='by_coords')
    name_var = {"d": "divergence",
                "t": "temperature",
                "v": "v_component_of_wind",
                "vo": "vorticity",
                "z": "geopotential", 
                "cc": "fraction_of_cloud_cover",
                "o3": "ozone_mass_mixing_ratio",
                "pv": "potential_vorticity",
                "r": "relative_humidity",
                "ciwc": "specific_cloud_ice_water_content",
                "clwc": "specific_cloud_liquid_water_content",
                "q": "specific_humidity", 
                "crwc": "specific_rain_water_content",
                "cswc": "specific_snow_water_content", 
                "u": "u_component_of_wind", 
                "w": "vertical_velocity"}
    data_array = data_array.rename(name_var)
    
    # Stack the variables
    var = list(data_array.data_vars)
    data_array =  xr.concat([data_array[name] for name in var], dim="variables").assign_coords(variables=var)
    return data_array

def load_era_land():
    tif_files = [os.path.join(f'{ROOT}/ERA5_Land_Data/', f) for f in os.listdir(f"{ROOT}/ERA5_Land_Data/")]
    #datasets = [rioxarray.open_rasterio(f) for f in tif_files]
    return rioxarray.open_rasterio(tif_files[0])

def load_coords():
    return gpd.read_file(f"{ROOT}/gauge_coords.geojson")

def load_gauges_values():
    r = pd.read_csv(f"{ROOT}/gauge.csv")
    r = r.set_index(pd.DatetimeIndex(r["timestamp"])).drop("timestamp", axis=1)
    return r

def read_radar_from_coords(radar, coords):
    return radar.sel(lat=coords.lat.to_xarray(), 
                     lon=coords.lon.to_xarray(), 
                     method="nearest")

def rasterize_rain(rain, coord, radar):
    raster = np.zeros((len(radar.lat), len(radar.lon), len(rain)))
    raster[:]=np.nan

    lat = np.argmin(np.abs(coords.lat.values[None] - radar.lat.values[:,None]), 0)
    lon = np.argmin(np.abs(coords.lon.values[None] - radar.lon.values[:,None]), 0)

    raster[lat.repeat(len(rain)), lon.repeat(len(rain)), np.concatenate([np.arange(len(rain))]*len(lat))]=rain.values.flatten()
    return raster

def read_radar_around_coords(radar, coords):
    return radar.sel(lat=coords.lat.to_xarray(), 
                     lon=coords.lon.to_xarray(), 
                     method="nearest")
