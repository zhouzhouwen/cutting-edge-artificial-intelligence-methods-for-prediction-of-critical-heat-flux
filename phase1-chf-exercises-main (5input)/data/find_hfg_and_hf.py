import pandas as pd
from iapws import IAPWS97

# Load the CSV file
file_path = '/home/user/ZHOU-Wen/phase1-chf-exercises-main/data/inputs/chf_public (copy).csv'
df = pd.read_csv(file_path)

# Function to calculate latent heat of vaporization
def calculate_latent_heat(pressure_mpa):
    pressure_mpa /= 1000
    h_liquid = IAPWS97(P=pressure_mpa, x=0).h  # Liquid water enthalpy
    h_vapor = IAPWS97(P=pressure_mpa, x=1).h  # Vapor enthalpy
    return h_vapor - h_liquid  # Latent heat

# Function to calculate saturated liquid enthalpy
def calculate_saturated_liquid_enthalpy(pressure_mpa):
    pressure_mpa /= 1000
    saturated_liquid = IAPWS97(P=pressure_mpa, x=0)
    return saturated_liquid.h

# Apply the functions to each row in the dataframe
df['latent_heat_of_vaporization [kJ/kg]'] = df['Pressure'].apply(calculate_latent_heat)
df['saturated_liquid_enthalpy [kJ/kg]'] = df['Pressure'].apply(calculate_saturated_liquid_enthalpy)

# Save the updated dataframe
df.to_csv('/home/user/ZHOU-Wen/phase1-chf-exercises-main/data/inputs/chf_public (copy)_with_inlet.csv', index=False)
