import pandas as pd

# This script will just convert the DAY column from day-of-month to day-of-year then save it 
# to a new file

# Parse weather data
weather_data = pd.read_csv('data/BearLake_inputs_outputs/inputs/BearLake-ERA5-daily-1994-2025.txt',  sep=r'\s+')

# Convert YEAR, MON, DAY (day-of-month) → DOY (day-of-year)
weather_data['DATE'] = pd.to_datetime(
    dict(year=weather_data['YEAR'], month=weather_data['MON'], day=weather_data['DAY'])
)
weather_data['DOY'] = weather_data['DATE'].dt.dayofyear

# (Optional) replace the old DAY column so DAY now means DOY
weather_data['DAY'] = weather_data['DOY']
weather_data = weather_data.drop(columns=['DATE'])  # keep tidy

print(weather_data.head())
weather_data.to_csv('data/parsed_data/weather_data.csv', index=False)
