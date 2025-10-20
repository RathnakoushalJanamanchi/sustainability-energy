import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class EnergyForecaster:
    """Advanced energy forecasting using ML and deep learning models"""
    
    def __init__(self):
        self.solar_model = None
        self.wind_model = None
        self.consumption_model = None
        self.lstm_model = None
        self.is_trained = False
    
    def generate_training_data(self, days=365):
        """Generate synthetic training data for ML models"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             end=datetime.now(), freq='H')
        
        data = []
        for date in dates:
            hour = date.hour
            day_of_year = date.timetuple().tm_yday
            month = date.month
            weekday = date.weekday()
            
            # Weather simulation
            temp = 20 + 15 * np.sin((day_of_year - 80) * 2 * np.pi / 365) + random.uniform(-5, 5)
            cloud_cover = random.uniform(0, 1)
            wind_speed = 8 + 5 * np.sin(hour * np.pi / 12) + random.uniform(-3, 3)
            
            # Solar generation (depends on time of day, season, weather)
            solar_potential = max(0, np.sin((hour - 6) * np.pi / 12)) if 6 <= hour <= 18 else 0
            solar_generation = solar_potential * (1 - cloud_cover * 0.7) * (100 + temp * 0.5)
            solar_generation = max(0, solar_generation + random.uniform(-10, 10))
            
            # Wind generation (depends on wind speed)
            wind_generation = min(100, max(0, (wind_speed - 3) ** 2 * 2)) + random.uniform(-5, 5)
            
            # Consumption (depends on time, season, weekday)
            base_consumption = 40 + 20 * np.sin((hour - 8) * np.pi / 16)  # Daily pattern
            seasonal_factor = 1.2 if month in [6, 7, 8, 12, 1, 2] else 1.0  # Summer/winter
            weekend_factor = 0.8 if weekday >= 5 else 1.0
            consumption = base_consumption * seasonal_factor * weekend_factor + random.uniform(-8, 8)
            
            data.append({
                'datetime': date,
                'hour': hour,
                'day_of_year': day_of_year,
                'month': month,
                'weekday': weekday,
                'temperature': temp,
                'cloud_cover': cloud_cover,
                'wind_speed': wind_speed,
                'solar_generation': solar_generation,
                'wind_generation': wind_generation,
                'consumption': consumption
            })
        
        return pd.DataFrame(data)
    
    def train_models(self):
        """Train machine learning models for energy forecasting"""
        print("Generating training data...")
        df = self.generate_training_data()
        
        # Prepare features
        features = ['hour', 'day_of_year', 'month', 'weekday', 
                   'temperature', 'cloud_cover', 'wind_speed']
        
        X = df[features]
        
        # Train solar generation model
        print("Training solar generation model...")
        y_solar = df['solar_generation']
        self.solar_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.solar_model.fit(X, y_solar)
        
        # Train wind generation model
        print("Training wind generation model...")
        y_wind = df['wind_generation']
        self.wind_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.wind_model.fit(X, y_wind)
        
        # Train consumption model
        print("Training consumption model...")
        y_consumption = df['consumption']
        self.consumption_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.consumption_model.fit(X, y_consumption)
        
        self.is_trained = True
        print("Models trained successfully!")
        
        # Calculate and return accuracy metrics
        solar_pred = self.solar_model.predict(X)
        wind_pred = self.wind_model.predict(X)
        consumption_pred = self.consumption_model.predict(X)
        
        return {
            'solar_rmse': np.sqrt(mean_squared_error(y_solar, solar_pred)),
            'wind_rmse': np.sqrt(mean_squared_error(y_wind, wind_pred)),
            'consumption_rmse': np.sqrt(mean_squared_error(y_consumption, consumption_pred))
        }
    
    def create_lstm_model(self, sequence_length=24):
        """Create LSTM model for time series forecasting"""
        model = keras.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, 7)),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def prepare_lstm_data(self, df, sequence_length=24):
        """Prepare data for LSTM training"""
        features = ['hour', 'day_of_year', 'month', 'weekday', 
                   'temperature', 'cloud_cover', 'wind_speed']
        
        X, y = [], []
        for i in range(sequence_length, len(df)):
            X.append(df[features].iloc[i-sequence_length:i].values)
            y.append(df['solar_generation'].iloc[i])  # Predicting solar for example
        
        return np.array(X), np.array(y)
    
    def train_lstm_model(self):
        """Train LSTM model for advanced time series forecasting"""
        if not self.is_trained:
            self.train_models()
        
        print("Training LSTM model...")
        df = self.generate_training_data(days=180)  # 6 months of data
        
        X, y = self.prepare_lstm_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.lstm_model = self.create_lstm_model()
        
        # Train the model
        history = self.lstm_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
        
        # Evaluate
        test_loss = self.lstm_model.evaluate(X_test, y_test, verbose=0)
        print(f"LSTM model trained successfully! Test loss: {test_loss[0]:.2f}")
        
        return history
    
    def forecast_generation(self, hours_ahead=24):
        """Forecast renewable energy generation"""
        if not self.is_trained:
            self.train_models()
        
        # Generate future timestamps
        start_time = datetime.now()
        future_times = [start_time + timedelta(hours=i) for i in range(hours_ahead)]
        
        forecasts = []
        
        for future_time in future_times:
            # Create feature vector for prediction
            hour = future_time.hour
            day_of_year = future_time.timetuple().tm_yday
            month = future_time.month
            weekday = future_time.weekday()
            
            # Simulate weather forecast (in production, would use real weather API)
            temp = 20 + 15 * np.sin((day_of_year - 80) * 2 * np.pi / 365) + random.uniform(-2, 2)
            cloud_cover = random.uniform(0.2, 0.8)
            wind_speed = 8 + 5 * np.sin(hour * np.pi / 12) + random.uniform(-1, 1)
            
            features = np.array([[hour, day_of_year, month, weekday, 
                                temp, cloud_cover, wind_speed]])
            
            solar_pred = max(0, self.solar_model.predict(features)[0])
            wind_pred = max(0, self.wind_model.predict(features)[0])
            consumption_pred = max(0, self.consumption_model.predict(features)[0])
            
            forecasts.append({
                'datetime': future_time,
                'solar_forecast': solar_pred,
                'wind_forecast': wind_pred,
                'consumption_forecast': consumption_pred,
                'total_generation': solar_pred + wind_pred,
                'net_balance': (solar_pred + wind_pred) - consumption_pred,
                'confidence_solar': random.uniform(0.8, 0.95),
                'confidence_wind': random.uniform(0.7, 0.9),
                'confidence_consumption': random.uniform(0.85, 0.95)
            })
        
        return pd.DataFrame(forecasts)
    
    def forecast_pricing(self, hours_ahead=24):
        """Forecast energy pricing based on supply/demand"""
        generation_forecast = self.forecast_generation(hours_ahead)
        
        pricing_forecasts = []
        
        for _, row in generation_forecast.iterrows():
            # Simple pricing model based on supply/demand ratio
            supply_demand_ratio = row['total_generation'] / row['consumption_forecast']
            
            base_price = 0.12  # Base price per kWh
            
            if supply_demand_ratio > 1.2:
                # Surplus - lower prices
                price = base_price * (0.7 + 0.2 * (1 / supply_demand_ratio))
            elif supply_demand_ratio < 0.8:
                # Deficit - higher prices
                price = base_price * (1.3 + 0.3 * (1 - supply_demand_ratio))
            else:
                # Balanced - normal pricing
                price = base_price * random.uniform(0.95, 1.05)
            
            # Add time-of-day pricing
            hour = row['datetime'].hour
            if 17 <= hour <= 20:  # Peak hours
                price *= 1.5
            elif 22 <= hour <= 6:  # Off-peak hours
                price *= 0.8
            
            pricing_forecasts.append({
                'datetime': row['datetime'],
                'predicted_price': max(0.05, price),
                'supply_demand_ratio': supply_demand_ratio,
                'price_category': 'Peak' if 17 <= hour <= 20 else 'Off-Peak' if 22 <= hour <= 6 else 'Standard'
            })
        
        return pd.DataFrame(pricing_forecasts)
    
    def get_optimization_recommendations(self, forecast_hours=24):
        """Get AI-powered optimization recommendations"""
        generation_forecast = self.forecast_generation(forecast_hours)
        pricing_forecast = self.forecast_pricing(forecast_hours)
        
        combined_forecast = generation_forecast.merge(pricing_forecast, on='datetime')
        
        recommendations = []
        
        for _, row in combined_forecast.iterrows():
            time_str = row['datetime'].strftime('%H:%M')
            
            # Energy surplus recommendations
            if row['net_balance'] > 10:
                if row['predicted_price'] > 0.15:
                    recommendations.append({
                        'time': time_str,
                        'type': 'Sell Energy',
                        'priority': 'High',
                        'description': f"Sell excess {row['net_balance']:.1f} kWh at high price ${row['predicted_price']:.3f}/kWh",
                        'potential_revenue': row['net_balance'] * row['predicted_price']
                    })
                else:
                    recommendations.append({
                        'time': time_str,
                        'type': 'Store Energy',
                        'priority': 'Medium',
                        'description': f"Store {row['net_balance']:.1f} kWh for later use (low market price)",
                        'potential_savings': row['net_balance'] * (0.15 - row['predicted_price'])
                    })
            
            # Energy deficit recommendations
            elif row['net_balance'] < -5:
                if row['predicted_price'] > 0.15:
                    recommendations.append({
                        'time': time_str,
                        'type': 'Reduce Consumption',
                        'priority': 'High',
                        'description': f"Reduce consumption by {abs(row['net_balance']):.1f} kWh (high price period)",
                        'potential_savings': abs(row['net_balance']) * (row['predicted_price'] - 0.12)
                    })
                else:
                    recommendations.append({
                        'time': time_str,
                        'type': 'Use Grid Power',
                        'priority': 'Low',
                        'description': f"Good time to use grid power (${row['predicted_price']:.3f}/kWh)",
                        'cost': abs(row['net_balance']) * row['predicted_price']
                    })
        
        return recommendations
    
    def calculate_forecast_accuracy(self, actual_data, forecasted_data):
        """Calculate forecast accuracy metrics"""
        if len(actual_data) != len(forecasted_data):
            return {"error": "Data length mismatch"}
        
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((actual_data - forecasted_data) / actual_data)) * 100
        
        # Root Mean Square Error (RMSE)
        rmse = np.sqrt(np.mean((actual_data - forecasted_data) ** 2))
        
        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(actual_data - forecasted_data))
        
        # R-squared
        ss_res = np.sum((actual_data - forecasted_data) ** 2)
        ss_tot = np.sum((actual_data - np.mean(actual_data)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mape': mape,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'accuracy_percentage': max(0, 100 - mape)
        }
    
    def get_model_performance_metrics(self):
        """Get comprehensive model performance metrics"""
        if not self.is_trained:
            return {"error": "Models not trained yet"}
        
        # Generate test data
        test_df = self.generate_training_data(days=30)
        features = ['hour', 'day_of_year', 'month', 'weekday', 
                   'temperature', 'cloud_cover', 'wind_speed']
        X_test = test_df[features]
        
        # Get predictions
        solar_pred = self.solar_model.predict(X_test)
        wind_pred = self.wind_model.predict(X_test)
        consumption_pred = self.consumption_model.predict(X_test)
        
        # Calculate metrics
        solar_metrics = self.calculate_forecast_accuracy(test_df['solar_generation'], solar_pred)
        wind_metrics = self.calculate_forecast_accuracy(test_df['wind_generation'], wind_pred)
        consumption_metrics = self.calculate_forecast_accuracy(test_df['consumption'], consumption_pred)
        
        return {
            'solar_model': solar_metrics,
            'wind_model': wind_metrics,
            'consumption_model': consumption_metrics,
            'overall_accuracy': np.mean([
                solar_metrics['accuracy_percentage'],
                wind_metrics['accuracy_percentage'], 
                consumption_metrics['accuracy_percentage']
            ])
        }

# Create a global instance
energy_forecaster = EnergyForecaster()
