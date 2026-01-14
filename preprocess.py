import pandas as pd 
import numpy as np
import os

def process_lake(lake):
    """Load lake weather/output data and align on shared YEAR/DOY keys."""
    base_dir = f'data/original_data/{lake}/'

    # Load the weather data
    weather_data = pd.read_csv(f'{base_dir}/inputs/{lake}-ERA5-daily-1994-2025.txt', sep='\t')
    weather_data["DOY"] = pd.to_datetime(
        dict(
            year=weather_data["YEAR"],
            month=weather_data["MON"],
            day=weather_data["DAY"],
        )
    ).dt.dayofyear

    # Load the parameter values 
    parameters = pd.read_csv(f'{base_dir}/parameter-values-tested.csv')
    parameters = parameters[['cdrn', 'eta', 'alb_snow', 'alb_slush']].to_numpy()

    # Load the bathymetry 
    bathymetry = pd.read_csv(f'{base_dir}/{lake}_bathy.csv')
    bathymetry = bathymetry[['area_ha']].to_numpy()

    outputs = []
    for i in range(1000):
        output = pd.read_csv(
            f'{base_dir}/outputs/profile-laketemp-{i}.txt',
            sep=r"\s+",
        )

        output = output.rename(columns={"DAY": "DOY"})
        overlap_keys = (
            output[["YEAR", "DOY"]]
            .merge(
                weather_data[["YEAR", "DOY"]].drop_duplicates(),
                on=["YEAR", "DOY"],
                how="inner",
            )
            .drop_duplicates()
        )
        output_overlap = output.merge(overlap_keys, on=["YEAR", "DOY"], how="inner")
        output_overlap = output_overlap.drop(columns=["MON"])

        # Get the weather data on the first iteration
        if i == 0:
            weather_overlap = weather_data.merge(overlap_keys, on=["YEAR", "DOY"], how="inner")
            weather_overlap = weather_overlap.drop(columns=["MON", "DAY"])
            weather = weather_overlap[['YEAR', 'DOY', 'T2M', 'D2M', 'WIND', 'SSRD', 'STRD', 'SP', 'TP']].to_numpy()

        print(output_overlap)
        output_i = output_overlap.to_numpy()
        outputs.append(output_i)

    # Concatenated outputs for all simulations
    outputs = np.array(outputs)

    # Create a directory for this lake
    directory = f'data/parsed_data/{lake}'
    os.makedirs(directory, exist_ok=True)
    # Save outputs for each simulation 
    np.save(f'{directory}/outputs.npy', outputs)
    # Save weather forcings
    np.save(f'{directory}/weather.npy', weather)
    # Save parameter values for each simulation
    np.save(f'{directory}/parameters.npy', parameters)
    # Save bathymetry 
    np.save(f'{directory}/bathymetry.npy', bathymetry)

    print(lake)
    print(parameters.shape)
    print(outputs.shape)
    print(weather.shape)
    print(bathymetry.shape)

lakes = ['BearLake', 'RedPond']
for lake in lakes:
    process_lake(lake)
