Here is the breakdown of the variables in your dataset. These follow the standard naming conventions used by the European Centre for Medium-Range Weather Forecasts (ECMWF), heavily utilized in meteorological machine learning models (like GraphCast, Pangu-Weather, or FourCastNet) and reanalysis datasets like ERA5.

I have categorized them into **Surface & Single-Level Parameters**, **Pressure Level Parameters**, and **Computed Forcing Features** for clarity.

### 1. Surface and Single-Level Parameters

These variables represent conditions at the Earth's surface, at a specific height above ground, or integrated across the entire atmospheric column.

| Variable | Meaning | Standard ECMWF Unit |
| :--- | :--- | :--- |
| **10fg** | 10 metre wind gust (since previous post-processing) | m/s |
| **10si** | 10 metre wind speed | m/s |
| **10u** | 10 metre U wind component (Eastward) | m/s |
| **10v** | 10 metre V wind component (Northward) | m/s |
| **2d** | 2 metre dewpoint temperature | K |
| **2t** | 2 metre temperature | K |
| **cbh** | Cloud base height | m |
| **fog** | Fog fraction | Dimensionless (0 to 1) |
| **hcc** | High cloud cover | Dimensionless (0 to 1) |
| **lcc** | Low cloud cover | Dimensionless (0 to 1) |
| **lsm** | Land-sea mask | Dimensionless (0 to 1) |
| **mcc** | Medium cloud cover | Dimensionless (0 to 1) |
| **msl** | Mean sea level pressure | Pa |
| **skt** | Skin temperature (surface temperature) | K |
| **sp** | Surface pressure | Pa |
| **ssrd** | Surface solar radiation downwards | J/m² |
| **strd** | Surface thermal radiation downwards | J/m² |
| **tcc** | Total cloud cover | Dimensionless (0 to 1) |
| **tcw** | Total column water | kg/m² |
| **tp** | Total precipitation | m |
| **vis** | Visibility | m |
| **z** | Geopotential (at the surface, acts as orography) | m²/s² |

---

### 2. Pressure Level Parameters

These variables correspond to specific isobaric pressure levels in the atmosphere. In your dataset, they are appended with the pressure level in hectopascals (hPa): `50, 100, 150, 200, 250, 300, 400, 500, 700, 850, 925, 1000`.

| Prefix | Meaning | Standard ECMWF Unit |
| :--- | :--- | :--- |
| **q_X** | Specific humidity at `X` hPa | kg/kg |
| **t_X** | Temperature at `X` hPa | K |
| **u_X** | U wind component (Eastward) at `X` hPa | m/s |
| **v_X** | V wind component (Northward) at `X` hPa | m/s |
| **w_X** | Vertical velocity (Omega) at `X` hPa | Pa/s |
| **z_X** | Geopotential at `X` hPa | m²/s² |

*Note: Omega (`w`) is a pressure vertical velocity. Negative values indicate ascending air (updrafts), and positive values indicate descending air (downdrafts).*

---

### 3. Computed Forcings

These are not standard raw MARS output variables but are explicitly computed spatial-temporal features used to feed cyclical and astronomical context into machine learning models.

| Variable | Meaning | Unit |
| :--- | :--- | :--- |
| **cos_julian_day** / **sin_julian_day** | Trigonometric encoding of the day of the year. | Dimensionless (-1 to 1) |
| **cos_latitude** / **sin_latitude** | Trigonometric encoding of the spatial latitude. | Dimensionless (-1 to 1) |
| **cos_longitude** / **sin_longitude** | Trigonometric encoding of the spatial longitude. | Dimensionless (-1 to 1) |
| **cos_local_time** / **sin_local_time** | Trigonometric encoding of the local time of day. | Dimensionless (-1 to 1) |
| **insolation** | Incident solar radiation at the Top of the Atmosphere (TOA). | J/m² (or W/m²) |

---
