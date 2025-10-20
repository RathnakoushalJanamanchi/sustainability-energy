"""
Weather API integration for accurate renewable energy forecasting
"""
import requests
import os
from datetime import datetime, timedelta
import random


class WeatherAPI:
    """Weather API client for fetching real weather data"""
    
    def __init__(self, api_key=None, location="San Francisco,US"):
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.location = location
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.use_mock = not self.api_key  # Use mock data if no API key
        
    def get_current_weather(self):
        """Get current weather conditions"""
        if self.use_mock:
            return self._get_mock_current_weather()
        
        try:
            url = f"{self.base_url}/weather"
            params = {
                'q': self.location,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            return {
                'timestamp': datetime.utcnow(),
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'cloud_cover': data['clouds']['all'],
                'wind_speed': data['wind']['speed'],
                'weather_description': data['weather'][0]['description'],
                'location': self.location,
                'is_forecast': False
            }
        except Exception as e:
            print(f"Weather API error: {e}. Using mock data.")
            return self._get_mock_current_weather()
    
    def get_forecast(self, hours=24):
        """Get weather forecast for next N hours"""
        if self.use_mock:
            return self._get_mock_forecast(hours)
        
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'q': self.location,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': min(40, hours // 3)  # API provides 3-hour intervals
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            forecasts = []
            for item in data['list']:
                forecasts.append({
                    'timestamp': datetime.fromtimestamp(item['dt']),
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'cloud_cover': item['clouds']['all'],
                    'wind_speed': item['wind']['speed'],
                    'weather_description': item['weather'][0]['description'],
                    'location': self.location,
                    'is_forecast': True,
                    'forecast_hours_ahead': (datetime.fromtimestamp(item['dt']) - datetime.utcnow()).total_seconds() / 3600
                })
            
            return forecasts
        except Exception as e:
            print(f"Weather forecast API error: {e}. Using mock data.")
            return self._get_mock_forecast(hours)
    
    def calculate_solar_irradiance(self, temperature, cloud_cover, hour):
        """
        Calculate estimated solar irradiance based on weather conditions
        Returns: Solar irradiance in W/m²
        """
        # Maximum solar irradiance at noon (clear sky)
        max_irradiance = 1000  # W/m²
        
        # Time of day factor (solar elevation)
        if 6 <= hour <= 18:
            # Simplified solar elevation model
            solar_elevation = max(0, (hour - 6) * (18 - hour) / 36)
            time_factor = solar_elevation
        else:
            time_factor = 0
        
        # Cloud cover impact (0-100%)
        cloud_factor = 1 - (cloud_cover / 100) * 0.7  # Clouds reduce irradiance by up to 70%
        
        # Temperature impact (minor effect)
        temp_factor = 1 - (temperature - 25) * 0.002  # Slight reduction at high temps
        temp_factor = max(0.8, min(1.2, temp_factor))
        
        # Calculate final irradiance
        irradiance = max_irradiance * time_factor * cloud_factor * temp_factor
        
        return max(0, irradiance)
    
    def get_energy_forecast(self, solar_capacity_kw=100, wind_capacity_kw=80, hours=24):
        """
        Generate energy generation forecast based on weather
        Returns: List of forecasted generation values
        """
        weather_forecasts = self.get_forecast(hours)
        energy_forecasts = []
        
        for weather in weather_forecasts:
            hour = weather['timestamp'].hour
            
            # Solar generation forecast
            irradiance = self.calculate_solar_irradiance(
                weather['temperature'],
                weather['cloud_cover'],
                hour
            )
            
            # Solar panel efficiency factors
            panel_efficiency = 0.18  # 18% efficient panels
            system_efficiency = 0.85  # Inverter and system losses
            
            solar_forecast_kw = (irradiance / 1000) * solar_capacity_kw * panel_efficiency * system_efficiency
            
            # Wind generation forecast
            wind_speed = weather['wind_speed']
            cut_in_speed = 3.0  # m/s
            rated_speed = 12.0  # m/s
            cut_out_speed = 25.0  # m/s
            
            if wind_speed < cut_in_speed or wind_speed > cut_out_speed:
                wind_forecast_kw = 0
            elif wind_speed >= rated_speed:
                wind_forecast_kw = wind_capacity_kw
            else:
                # Power curve approximation
                power_ratio = ((wind_speed - cut_in_speed) / (rated_speed - cut_in_speed)) ** 3
                wind_forecast_kw = wind_capacity_kw * power_ratio
            
            energy_forecasts.append({
                'timestamp': weather['timestamp'],
                'solar_forecast_kw': max(0, solar_forecast_kw),
                'wind_forecast_kw': max(0, wind_forecast_kw),
                'total_forecast_kw': max(0, solar_forecast_kw + wind_forecast_kw),
                'weather_description': weather['weather_description'],
                'temperature': weather['temperature'],
                'cloud_cover': weather['cloud_cover'],
                'wind_speed': weather['wind_speed'],
                'solar_irradiance_w_m2': irradiance,
                'confidence': self._calculate_forecast_confidence(weather['forecast_hours_ahead'])
            })
        
        return energy_forecasts
    
    def _calculate_forecast_confidence(self, hours_ahead):
        """Calculate forecast confidence based on time horizon"""
        if hours_ahead <= 3:
            return 0.95  # Very high confidence for short-term
        elif hours_ahead <= 12:
            return 0.85  # High confidence for mid-term
        elif hours_ahead <= 24:
            return 0.75  # Good confidence for daily
        else:
            return 0.65  # Moderate confidence for longer term
    
    def _get_mock_current_weather(self):
        """Generate mock current weather data"""
        hour = datetime.now().hour
        
        # Simulate realistic weather patterns
        base_temp = 20 + 10 * (hour - 12) / 12 if 6 <= hour <= 18 else 15
        
        return {
            'timestamp': datetime.utcnow(),
            'temperature': base_temp + random.uniform(-3, 3),
            'humidity': random.uniform(40, 80),
            'cloud_cover': random.uniform(10, 60),
            'wind_speed': 5 + random.uniform(-2, 5),
            'weather_description': random.choice(['clear sky', 'few clouds', 'scattered clouds', 'partly cloudy']),
            'location': self.location,
            'is_forecast': False
        }
    
    def _get_mock_forecast(self, hours=24):
        """Generate mock weather forecast"""
        forecasts = []
        current_time = datetime.utcnow()
        
        for i in range(0, hours, 3):  # 3-hour intervals
            forecast_time = current_time + timedelta(hours=i)
            hour = forecast_time.hour
            
            # Simulate realistic daily weather pattern
            base_temp = 20 + 8 * ((hour - 12) / 12) if 6 <= hour <= 18 else 12
            
            forecasts.append({
                'timestamp': forecast_time,
                'temperature': base_temp + random.uniform(-2, 2),
                'humidity': random.uniform(40, 80),
                'cloud_cover': random.uniform(15, 70),
                'wind_speed': 5 + random.uniform(-2, 4),
                'weather_description': random.choice(['clear sky', 'few clouds', 'scattered clouds', 'partly cloudy']),
                'location': self.location,
                'is_forecast': True,
                'forecast_hours_ahead': i
            })
        
        return forecasts


# Global weather API instance
_weather_api = None

def get_weather_api(api_key=None, location="San Francisco,US"):
    """Get global weather API instance"""
    global _weather_api
    if _weather_api is None:
        _weather_api = WeatherAPI(api_key=api_key, location=location)
    return _weather_api
