"""
CF-1 Standard Name Mappings

Maps short variable names to CF-1 convention standard names.
This is the single source of truth for variable name mappings.
"""

# Atmospheric variable mappings (short name -> CF-1 standard name)
CF_ATM = {
    'lat': 'latitude',
    'lon': 'longitude',
    'tair': 'air_temperature',
    'uatm': 'eastward_wind',
    'vatm': 'northward_wind',
    'tsfc': 'skin_temperature_at_surface',
    'qref': 'water_vapor_mixing_ratio_wrt_moist_air',
    'pressfc': 'air_pressure_at_surface'
}

# Ocean/ice variable mappings (short name -> CF-1 standard name)
CF_OCN = {
    'lat': 'latitude',
    'lon': 'longitude',
    'sst': 'sea_water_potential_temperature',
    'sss': 'sea_water_salinity',
    'aice': 'sea_ice_area_fraction',
    'hi': 'sea_ice_thickness',
    'hs': 'sea_ice_snow_thickness',
    'thick': 'sea_water_cell_thickness',
    'h': 'h',        # MOM6 layer thickness (alternative name)
    'sice': 'sea_ice_salinity',
    'uocn': 'eastward_sea_water_velocity',
    'vocn': 'northward_sea_water_velocity',
    'Temp': 'Temp',  # MOM6 temperature variable name
    'Salt': 'Salt',  # MOM6 salinity variable name
    'uo': 'uo',      # UFS ocean x-velocity
    'vo': 'vo',      # UFS ocean y-velocity
    'so': 'so',      # UFS salinity
    'temp': 'temp',  # UFS temperature
    'ho': 'ho',      # UFS layer thickness
}

# Default atmospheric level index for fallback
# This should be nlevs-1 where nlevs is the number of vertical levels in the atmospheric model
# Examples:
#   - 64-level model: DEFAULT_ATM_LEVEL = 63
#   - 128-level model: DEFAULT_ATM_LEVEL = 127
# This can be overridden in config files via domain.atm_level_index
DEFAULT_ATM_LEVEL = 126  # nlevs-1 for 127-level model
