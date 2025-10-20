import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

class EnergyDataGenerator:
    """Generate realistic energy system data for testing and demonstration"""
    
    def __init__(self):
        self.base_solar_capacity = 100  # kW
        self.base_wind_capacity = 80    # kW
        self.base_consumption = 60      # kW average
        
    def generate_weather_data(self, start_date, end_date, freq='H'):
        """Generate realistic weather data"""
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        weather_data = []
        
        for date in dates:
            day_of_year = date.timetuple().tm_yday
            hour = date.hour
            
            # Temperature (seasonal variation + daily variation)
            base_temp = 15 + 10 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
            daily_temp_variation = 8 * np.sin((hour - 6) * np.pi / 12)
            temperature = base_temp + daily_temp_variation + random.uniform(-3, 3)
            
            # Solar irradiance (clear sky model with weather variations)
            solar_elevation = max(0, np.sin((hour - 6) * np.pi / 12)) if 6 <= hour <= 18 else 0
            cloud_factor = random.uniform(0.3, 1.0)  # 0.3 = very cloudy, 1.0 = clear
            solar_irradiance = solar_elevation * cloud_factor * 1000  # W/m²
            
            # Wind speed (realistic wind patterns)
            base_wind = 6 + 4 * np.sin(hour * np.pi / 12)  # Daily wind pattern
            wind_variability = random.uniform(0.5, 1.5)
            wind_speed = max(0, base_wind * wind_variability)
            
            # Humidity
            humidity = 40 + 30 * (1 - cloud_factor) + random.uniform(-10, 10)
            humidity = max(20, min(95, humidity))
            
            weather_data.append({
                'datetime': date,
                'temperature_c': round(temperature, 1),
                'solar_irradiance_w_m2': round(solar_irradiance, 1),
                'wind_speed_ms': round(wind_speed, 1),
                'humidity_percent': round(humidity, 1),
                'cloud_cover_percent': round((1 - cloud_factor) * 100, 1)
            })
        
        return pd.DataFrame(weather_data)
    
    def generate_solar_generation(self, weather_df, panel_capacity_kw=100):
        """Generate realistic solar power generation from weather data"""
        generation_data = []
        
        for _, row in weather_df.iterrows():
            # Solar panel efficiency factors
            irradiance = row['solar_irradiance_w_m2']
            temperature = row['temperature_c']
            
            # Standard Test Conditions: 1000 W/m², 25°C
            # Temperature coefficient: -0.4%/°C
            temp_factor = 1 - 0.004 * (temperature - 25)
            
            # Irradiance factor (linear relationship)
            irradiance_factor = irradiance / 1000
            
            # System efficiency factors
            inverter_efficiency = 0.95
            system_losses = 0.85  # Wiring, soiling, shading, etc.
            
            # Calculate generation
            generation_kw = (panel_capacity_kw * irradiance_factor * temp_factor * 
                           inverter_efficiency * system_losses)
            generation_kw = max(0, generation_kw)
            
            generation_data.append({
                'datetime': row['datetime'],
                'solar_generation_kw': round(generation_kw, 2),
                'capacity_factor': round((generation_kw / panel_capacity_kw) * 100, 1),
                'efficiency_factor': round(temp_factor * inverter_efficiency * system_losses, 3)
            })
        
        return pd.DataFrame(generation_data)
    
    def generate_wind_generation(self, weather_df, turbine_capacity_kw=80):
        """Generate realistic wind power generation from weather data"""
        generation_data = []
        
        # Wind turbine power curve parameters
        cut_in_speed = 3.0    # m/s
        rated_speed = 12.0    # m/s
        cut_out_speed = 25.0  # m/s
        
        for _, row in weather_df.iterrows():
            wind_speed = row['wind_speed_ms']
            
            if wind_speed < cut_in_speed or wind_speed > cut_out_speed:
                generation_kw = 0
            elif wind_speed >= rated_speed:
                generation_kw = turbine_capacity_kw
            else:
                # Power increases roughly with cube of wind speed
                power_ratio = ((wind_speed - cut_in_speed) / (rated_speed - cut_in_speed)) ** 3
                generation_kw = turbine_capacity_kw * power_ratio
            
            # Add some variability for turbulence and mechanical factors
            generation_kw *= random.uniform(0.9, 1.1)
            generation_kw = max(0, min(turbine_capacity_kw, generation_kw))
            
            generation_data.append({
                'datetime': row['datetime'],
                'wind_generation_kw': round(generation_kw, 2),
                'capacity_factor': round((generation_kw / turbine_capacity_kw) * 100, 1),
                'wind_speed_ms': wind_speed
            })
        
        return pd.DataFrame(generation_data)
    
    def generate_consumption_profile(self, start_date, end_date, base_load_kw=60):
        """Generate realistic energy consumption profile"""
        dates = pd.date_range(start=start_date, end=end_date, freq='H')
        consumption_data = []
        
        for date in dates:
            hour = date.hour
            day_of_week = date.weekday()  # 0 = Monday, 6 = Sunday
            month = date.month
            
            # Base load pattern (typical household/business)
            if 0 <= hour <= 6:      # Night - low consumption
                hourly_factor = 0.6
            elif 7 <= hour <= 8:    # Morning ramp-up
                hourly_factor = 0.8 + (hour - 7) * 0.1
            elif 9 <= hour <= 17:   # Day time
                hourly_factor = 0.9
            elif 18 <= hour <= 22:  # Evening peak
                hourly_factor = 1.2
            else:                   # Late evening
                hourly_factor = 0.8
            
            # Weekend factor (lower commercial, higher residential)
            weekend_factor = 0.85 if day_of_week >= 5 else 1.0
            
            # Seasonal factor (heating/cooling)
            if month in [12, 1, 2]:      # Winter
                seasonal_factor = 1.3
            elif month in [6, 7, 8]:     # Summer  
                seasonal_factor = 1.2
            else:                        # Spring/Fall
                seasonal_factor = 1.0
            
            # Calculate consumption with random variation
            consumption = (base_load_kw * hourly_factor * weekend_factor * 
                          seasonal_factor * random.uniform(0.8, 1.2))
            
            consumption_data.append({
                'datetime': date,
                'consumption_kw': round(max(10, consumption), 2),  # Minimum 10 kW
                'base_load_factor': hourly_factor,
                'weekend_factor': weekend_factor,
                'seasonal_factor': seasonal_factor
            })
        
        return pd.DataFrame(consumption_data)
    
    def generate_grid_data(self, start_date, end_date):
        """Generate grid-related data (pricing, carbon intensity, etc.)"""
        dates = pd.date_range(start=start_date, end=end_date, freq='H')
        grid_data = []
        
        base_price = 0.12  # $/kWh
        base_carbon_intensity = 0.5  # kg CO2/kWh
        
        for date in dates:
            hour = date.hour
            day_of_week = date.weekday()
            month = date.month
            
            # Time-of-use pricing
            if 17 <= hour <= 20:        # Peak hours
                price_factor = 1.5
            elif 22 <= hour <= 6:       # Off-peak hours
                price_factor = 0.7
            else:                       # Standard hours
                price_factor = 1.0
            
            # Weekend pricing (typically lower)
            if day_of_week >= 5:
                price_factor *= 0.9
            
            # Seasonal pricing variations
            if month in [6, 7, 8]:      # Summer (high AC demand)
                price_factor *= 1.1
            elif month in [12, 1, 2]:   # Winter (high heating demand)
                price_factor *= 1.05
            
            # Calculate final price with market volatility
            electricity_price = base_price * price_factor * random.uniform(0.9, 1.1)
            
            # Carbon intensity varies with grid mix
            # Higher during peak hours (more fossil fuels), lower at night (more renewables)
            if 17 <= hour <= 20:
                carbon_factor = 1.3
            elif 2 <= hour <= 5:
                carbon_factor = 0.7  # High renewable penetration at night
            else:
                carbon_factor = 1.0
            
            # Seasonal carbon intensity
            if month in [6, 7, 8]:      # Summer (more solar)
                carbon_factor *= 0.9
            elif month in [12, 1, 2]:   # Winter (more fossil heating)
                carbon_factor *= 1.1
            
            carbon_intensity = base_carbon_intensity * carbon_factor * random.uniform(0.9, 1.1)
            
            grid_data.append({
                'datetime': date,
                'electricity_price_kwh': round(electricity_price, 4),
                'carbon_intensity_kg_co2_kwh': round(carbon_intensity, 3),
                'price_category': self._get_price_category(hour),
                'demand_level': self._get_demand_level(hour, day_of_week)
            })
        
        return pd.DataFrame(grid_data)
    
    def generate_battery_data(self, consumption_df, generation_df, capacity_kwh=100):
        """Generate battery storage system data"""
        battery_data = []
        current_charge = capacity_kwh * 0.5  # Start at 50% charge
        
        # Combine consumption and generation data
        combined_df = pd.merge(consumption_df, generation_df, on='datetime', how='inner')
        
        for _, row in combined_df.iterrows():
            consumption = row['consumption_kw']
            
            # Get total generation (assuming we have both solar and wind)
            generation = 0
            if 'solar_generation_kw' in row:
                generation += row['solar_generation_kw']
            if 'wind_generation_kw' in row:
                generation += row['wind_generation_kw']
            
            net_power = generation - consumption
            
            # Battery charging/discharging logic
            if net_power > 0:  # Excess generation - charge battery
                max_charge_power = min(net_power, capacity_kwh * 0.5)  # Max charge rate = 0.5C
                charge_amount = min(max_charge_power, capacity_kwh - current_charge)
                current_charge += charge_amount * 0.95  # 95% charge efficiency
                battery_flow = charge_amount
                battery_mode = 'charging'
            elif net_power < 0:  # Deficit - discharge battery
                max_discharge_power = min(abs(net_power), capacity_kwh * 0.5)  # Max discharge rate = 0.5C
                discharge_amount = min(max_discharge_power, current_charge)
                current_charge -= discharge_amount
                battery_flow = -discharge_amount * 0.95  # 95% discharge efficiency
                battery_mode = 'discharging'
            else:
                battery_flow = 0
                battery_mode = 'idle'
            
            # Ensure battery charge stays within limits
            current_charge = max(0, min(capacity_kwh, current_charge))
            
            battery_data.append({
                'datetime': row['datetime'],
                'battery_charge_kwh': round(current_charge, 2),
                'battery_soc_percent': round((current_charge / capacity_kwh) * 100, 1),
                'battery_flow_kw': round(battery_flow, 2),
                'battery_mode': battery_mode,
                'net_power_kw': round(net_power, 2)
            })
        
        return pd.DataFrame(battery_data)
    
    def generate_complete_dataset(self, start_date, end_date, solar_capacity=100, 
                                 wind_capacity=80, base_consumption=60, battery_capacity=100):
        """Generate complete integrated energy system dataset"""
        print(f"Generating complete dataset from {start_date} to {end_date}")
        
        # Generate weather data
        print("Generating weather data...")
        weather_df = self.generate_weather_data(start_date, end_date)
        
        # Generate solar generation
        print("Generating solar generation data...")
        solar_df = self.generate_solar_generation(weather_df, solar_capacity)
        
        # Generate wind generation  
        print("Generating wind generation data...")
        wind_df = self.generate_wind_generation(weather_df, wind_capacity)
        
        # Generate consumption profile
        print("Generating consumption data...")
        consumption_df = self.generate_consumption_profile(start_date, end_date, base_consumption)
        
        # Generate grid data
        print("Generating grid data...")
        grid_df = self.generate_grid_data(start_date, end_date)
        
        # Combine all generation data
        generation_df = pd.merge(solar_df[['datetime', 'solar_generation_kw']], 
                               wind_df[['datetime', 'wind_generation_kw']], on='datetime')
        generation_df['total_generation_kw'] = (generation_df['solar_generation_kw'] + 
                                              generation_df['wind_generation_kw'])
        
        # Generate battery data
        print("Generating battery data...")
        battery_df = self.generate_battery_data(consumption_df, generation_df, battery_capacity)
        
        # Combine all datasets
        print("Combining datasets...")
        complete_df = weather_df
        complete_df = pd.merge(complete_df, solar_df, on='datetime')
        complete_df = pd.merge(complete_df, wind_df, on='datetime')  
        complete_df = pd.merge(complete_df, consumption_df, on='datetime')
        complete_df = pd.merge(complete_df, grid_df, on='datetime')
        complete_df = pd.merge(complete_df, battery_df, on='datetime')
        
        # Add calculated fields
        complete_df['net_energy_kw'] = (complete_df['total_generation_kw'] - 
                                       complete_df['consumption_kw'])
        complete_df['grid_import_kw'] = np.maximum(0, -complete_df['net_energy_kw'])
        complete_df['grid_export_kw'] = np.maximum(0, complete_df['net_energy_kw'])
        
        print(f"Dataset generation complete! Generated {len(complete_df)} records.")
        return complete_df
    
    def _get_price_category(self, hour):
        """Get price category for given hour"""
        if 17 <= hour <= 20:
            return "Peak"
        elif 22 <= hour <= 6:
            return "Off-Peak"
        else:
            return "Standard"
    
    def _get_demand_level(self, hour, day_of_week):
        """Get demand level description"""
        if day_of_week >= 5:  # Weekend
            return "Low"
        elif 17 <= hour <= 20:
            return "High"
        elif 9 <= hour <= 17:
            return "Medium"
        else:
            return "Low"
    
    def export_to_json(self, dataframe, filename):
        """Export dataframe to JSON format"""
        # Convert datetime to string for JSON serialization
        df_export = dataframe.copy()
        df_export['datetime'] = df_export['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Export to JSON
        with open(filename, 'w') as f:
            json.dump(df_export.to_dict('records'), f, indent=2)
        
        print(f"Data exported to {filename}")
    
    def export_to_csv(self, dataframe, filename):
        """Export dataframe to CSV format"""
        dataframe.to_csv(filename, index=False)
        print(f"Data exported to {filename}")

