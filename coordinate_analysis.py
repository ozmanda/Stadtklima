import re
import netCDF4 as nc
from utils import wgs84_to_lv, lv_to_wgs84, lv03_to_lv95, lv95_to_lv03


def deg_to_dms(deg, pretty_print=None, ndp=4):
    """Convert from decimal degrees to degrees, minutes, seconds."""

    m, s = divmod(abs(deg)*3600, 60)
    d, m = divmod(m, 60)
    if deg < 0:
        d = -d
    d, m = int(d), int(m)

    if pretty_print:
        if pretty_print=='latitude':
            hemi = 'N' if d>=0 else 'S'
        elif pretty_print=='longitude':
            hemi = 'E' if d>=0 else 'W'
        else:
            hemi = '?'
        return '{d:d}° {m:d}′ {s:.{ndp:d}f}″ {hemi:1s}'.format(
                    d=abs(d), m=m, s=s, hemi=hemi, ndp=ndp)
    return d, m, s


palmpath = 'S:/pools/t/T-IDP-Projekte-u-Vorlesungen/Meteoblue/Data/PALM Maps/mb_4_multi_stations_xy_N02.00m.nc'
file = nc.Dataset(palmpath)

# WGS84 origin latitude and longitude
lat = file.origin_lat
lon = file.origin_lon
print(f'{lat}, {lon}')

# transformation of WGS84 lat/lon to LV95 lat/lon
lv95_lat, lv95_lon = lv03_to_lv95(file.origin_y, file.origin_x)
lat_03_transformed, lon_03_transformed = lv_to_wgs84(file.origin_y, file.origin_x, type='lv03')
lat_95_transformed, lon_95_transformed = lv_to_wgs84(lv95_lat, lv95_lon, type='lv95')
print(f'Transformation from LV03:\nlat: {lat_03_transformed}\nlon: {lon_03_transformed}\n\n'
      f'Transformation from LV95:\nlat: {lat_95_transformed}\nlon: {lon_95_transformed} ')
print(f'  Lat_delta: {lat_03_transformed-lat_95_transformed}\n  Lon_delta: {lon_03_transformed-lon_95_transformed}')

# Swisstopo example - WGS84 -> LV95
lat_dms = '''46°02'38.87"N'''
lon_dms = '''8° 43' 49.79"E'''
deg, minutes, seconds, direction = re.split('[°\'"]', lat_dms)
lat_deg = (float(deg) + float(minutes)/60 + float(seconds)/(60*60)) * (-1 if direction in ['W', 'S'] else 1)
deg, minutes, seconds, direction = re.split('[°\'"]', lon_dms)
lon_deg = (float(deg) + float(minutes)/60 + float(seconds)/(60*60)) * (-1 if direction in ['W', 'S'] else 1)

lat_95_transformed, lon_95_transformed = wgs84_to_lv(lat_deg, lon_deg, type='lv95')
print(f'{lat_95_transformed, lon_95_transformed}')

# Station 1079
lat_1079 = 47.3824
lon_1079 = 8.51468
lv95_lat_1079, lv95_lon_1079 = wgs84_to_lv(lat_1079, lon_1079, type='lv95')
print(f'{lv95_lat_1079}, {lv95_lon_1079}')

lat_1079 = 1248509
lon_1079 = 2681255
wgs_lat_1079, wgs_lon_1079 = lv_to_wgs84(lat_1079, lon_1079, type='lv95')
print(f'{wgs_lat_1079}, {wgs_lon_1079}')

# Station 941
lat_941 = 47.41471
lon_941 = 8.5160335
wgs_lat_941, wgs_lon_941 = wgs84_to_lv(lat_941, lon_941, type='lv95')
print(f'{wgs_lat_941}, {wgs_lon_941}')

lat_941 = 1252127
lon_941 = 2683023
wgs_lat_941, wgs_lon_941 = lv_to_wgs84(lat_941, lon_941, type='lv95')
print(f'{wgs_lat_941}, {wgs_lon_941}')

# Siwsstopo example - LV95 -> WGS84
lv95_lat = 1100000
lv95_lon = 2700000
wgs_lat, wgs_lon = lv_to_wgs84(lv95_lat, lv95_lon, type='lv95')
deg_to_dms(wgs_lat, pretty_print='latitude')
deg_to_dms(wgs_lon, pretty_print='longitude')
print(f'{wgs_lat}, {wgs_lon}')


