"""
Database operations and utilities for energy platform
"""
from database.models import (
    get_session, init_db, get_or_create_default_user,
    User, EnergyReading, Transaction, TradingHistory, 
    WeatherData, EnergyForecast, SystemAlert
)
from datetime import datetime, timedelta
from sqlalchemy import func, desc
import uuid


class EnergyDatabase:
    """Main database operations class"""
    
    def __init__(self):
        self.session = None
        self.user = None
    
    def connect(self):
        """Initialize database connection"""
        try:
            init_db()
            self.session = get_session()
            self.user = get_or_create_default_user(self.session)
            if self.user:
                return True
            else:
                print("Failed to create/get default user")
                return False
        except Exception as e:
            print(f"Database connection error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def close(self):
        """Close database connection"""
        if self.session:
            self.session.close()
    
    # Energy Reading Operations
    def log_energy_reading(self, reading_data):
        """Log real-time energy reading to database"""
        try:
            # Calculate derived fields
            net_energy = reading_data.get('total_generation_kw', 0) - reading_data.get('consumption_kw', 0)
            cost = reading_data.get('grid_import_kw', 0) * reading_data.get('price_per_kwh', 0) / 60  # Per minute
            revenue = reading_data.get('grid_export_kw', 0) * reading_data.get('price_per_kwh', 0) * 0.85 / 60
            carbon_saved = reading_data.get('total_generation_kw', 0) * reading_data.get('carbon_intensity_kg_per_kwh', 0.45) / 60
            
            reading = EnergyReading(
                user_id=self.user.id,
                timestamp=reading_data.get('timestamp', datetime.utcnow()),
                solar_generation_kw=reading_data.get('solar_generation', 0),
                wind_generation_kw=reading_data.get('wind_generation', 0),
                total_generation_kw=reading_data.get('total_generation_kw', 0),
                consumption_kw=reading_data.get('consumption', 0),
                grid_import_kw=reading_data.get('grid_import', 0),
                grid_export_kw=reading_data.get('grid_export', 0),
                battery_level_pct=reading_data.get('battery_level', 50),
                price_per_kwh=reading_data.get('price_per_kwh', 0.12),
                carbon_intensity_kg_per_kwh=reading_data.get('carbon_intensity', 0.45),
                net_energy_kw=net_energy,
                cost_usd=cost,
                revenue_usd=revenue,
                carbon_saved_kg=carbon_saved
            )
            
            self.session.add(reading)
            self.session.commit()
            return True
        except Exception as e:
            print(f"Error logging energy reading: {e}")
            self.session.rollback()
            return False
    
    def get_recent_readings(self, hours=24, limit=1000):
        """Get recent energy readings"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            readings = self.session.query(EnergyReading).filter(
                EnergyReading.user_id == self.user.id,
                EnergyReading.timestamp >= cutoff_time
            ).order_by(desc(EnergyReading.timestamp)).limit(limit).all()
            
            return [{
                'timestamp': r.timestamp,
                'solar_generation': r.solar_generation_kw,
                'wind_generation': r.wind_generation_kw,
                'total_generation': r.total_generation_kw,
                'consumption': r.consumption_kw,
                'grid_import': r.grid_import_kw,
                'grid_export': r.grid_export_kw,
                'battery_level': r.battery_level_pct,
                'price_per_kwh': r.price_per_kwh,
                'cost_usd': r.cost_usd,
                'revenue_usd': r.revenue_usd,
                'carbon_saved_kg': r.carbon_saved_kg
            } for r in readings]
        except Exception as e:
            print(f"Error fetching readings: {e}")
            return []
    
    # Transaction Operations
    def create_transaction(self, transaction_type, amount_usd, description=None, energy_kwh=0, price_per_kwh=0):
        """Create a financial transaction"""
        try:
            transaction_id = f"TXN-{uuid.uuid4().hex[:12].upper()}"
            
            transaction = Transaction(
                user_id=self.user.id,
                transaction_type=transaction_type,
                description=description,
                energy_kwh=energy_kwh,
                price_per_kwh=price_per_kwh,
                amount_usd=amount_usd,
                status='completed',
                transaction_id=transaction_id
            )
            
            self.session.add(transaction)
            self.session.commit()
            return transaction_id
        except Exception as e:
            print(f"Error creating transaction: {e}")
            self.session.rollback()
            return None
    
    def get_transaction_history(self, days=30, limit=100):
        """Get transaction history"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            transactions = self.session.query(Transaction).filter(
                Transaction.user_id == self.user.id,
                Transaction.timestamp >= cutoff_time
            ).order_by(desc(Transaction.timestamp)).limit(limit).all()
            
            return [{
                'timestamp': t.timestamp,
                'type': t.transaction_type,
                'description': t.description,
                'amount': t.amount_usd,
                'energy_kwh': t.energy_kwh,
                'transaction_id': t.transaction_id,
                'status': t.status
            } for t in transactions]
        except Exception as e:
            print(f"Error fetching transactions: {e}")
            return []
    
    def get_financial_summary(self, days=30):
        """Get financial summary for period"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            
            # Total revenue (energy sales)
            total_revenue = self.session.query(func.sum(Transaction.amount_usd)).filter(
                Transaction.user_id == self.user.id,
                Transaction.transaction_type == 'sale',
                Transaction.timestamp >= cutoff_time
            ).scalar() or 0
            
            # Total costs (energy purchases + fees)
            total_costs = self.session.query(func.sum(Transaction.amount_usd)).filter(
                Transaction.user_id == self.user.id,
                Transaction.transaction_type.in_(['purchase', 'fee', 'maintenance']),
                Transaction.timestamp >= cutoff_time
            ).scalar() or 0
            
            # Net savings
            net_savings = total_revenue - total_costs
            
            return {
                'total_revenue': total_revenue,
                'total_costs': total_costs,
                'net_savings': net_savings,
                'period_days': days
            }
        except Exception as e:
            print(f"Error fetching financial summary: {e}")
            return {'total_revenue': 0, 'total_costs': 0, 'net_savings': 0, 'period_days': days}
    
    # Trading Operations
    def execute_trade(self, trade_type, energy_kwh, price_per_kwh):
        """Execute an energy trade"""
        try:
            total_amount = energy_kwh * price_per_kwh
            trade_id = f"TRADE-{uuid.uuid4().hex[:12].upper()}"
            
            trade = TradingHistory(
                user_id=self.user.id,
                trade_type=trade_type,
                energy_kwh=energy_kwh,
                price_per_kwh=price_per_kwh,
                total_amount_usd=total_amount,
                market_price_at_trade=price_per_kwh,
                profit_loss_usd=0 if trade_type == 'buy' else total_amount * 0.05,  # 5% profit on sales
                status='completed',
                trade_id=trade_id
            )
            
            self.session.add(trade)
            
            # Also create corresponding transaction
            transaction_type = 'sale' if trade_type == 'sell' else 'purchase'
            self.create_transaction(
                transaction_type=transaction_type,
                amount_usd=total_amount if trade_type == 'sell' else -total_amount,
                description=f"Energy {trade_type}: {energy_kwh:.2f} kWh @ ${price_per_kwh:.3f}/kWh",
                energy_kwh=energy_kwh,
                price_per_kwh=price_per_kwh
            )
            
            self.session.commit()
            return trade_id
        except Exception as e:
            print(f"Error executing trade: {e}")
            self.session.rollback()
            return None
    
    def get_trading_history(self, days=30, limit=100):
        """Get trading history"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            trades = self.session.query(TradingHistory).filter(
                TradingHistory.user_id == self.user.id,
                TradingHistory.timestamp >= cutoff_time
            ).order_by(desc(TradingHistory.timestamp)).limit(limit).all()
            
            return [{
                'timestamp': t.timestamp,
                'trade_type': t.trade_type,
                'energy_kwh': t.energy_kwh,
                'price_per_kwh': t.price_per_kwh,
                'total_amount': t.total_amount_usd,
                'profit_loss': t.profit_loss_usd,
                'trade_id': t.trade_id,
                'status': t.status
            } for t in trades]
        except Exception as e:
            print(f"Error fetching trading history: {e}")
            return []
    
    # Weather Data Operations
    def log_weather_data(self, weather_data):
        """Log weather data for forecasting"""
        try:
            weather = WeatherData(
                timestamp=weather_data.get('timestamp', datetime.utcnow()),
                location=weather_data.get('location', 'default'),
                temperature_c=weather_data.get('temperature', 0),
                humidity_pct=weather_data.get('humidity', 0),
                cloud_cover_pct=weather_data.get('cloud_cover', 0),
                wind_speed_ms=weather_data.get('wind_speed', 0),
                solar_irradiance_w_m2=weather_data.get('solar_irradiance', 0),
                is_forecast=weather_data.get('is_forecast', False),
                forecast_hours_ahead=weather_data.get('forecast_hours', 0)
            )
            
            self.session.add(weather)
            self.session.commit()
            return True
        except Exception as e:
            print(f"Error logging weather data: {e}")
            self.session.rollback()
            return False
    
    # User Preferences Operations
    def update_user_preferences(self, preferences):
        """Update user preferences"""
        try:
            for key, value in preferences.items():
                if hasattr(self.user, key):
                    setattr(self.user, key, value)
            
            self.session.commit()
            return True
        except Exception as e:
            print(f"Error updating preferences: {e}")
            self.session.rollback()
            return False
    
    def get_user_preferences(self):
        """Get user preferences"""
        return {
            'monthly_savings_goal': self.user.monthly_savings_goal,
            'carbon_reduction_goal_kg': self.user.carbon_reduction_goal_kg,
            'energy_independence_goal_pct': self.user.energy_independence_goal_pct,
            'solar_capacity_kw': self.user.solar_capacity_kw,
            'wind_capacity_kw': self.user.wind_capacity_kw,
            'battery_capacity_kwh': self.user.battery_capacity_kwh,
            'auto_trading_enabled': self.user.auto_trading_enabled,
            'peak_alert_enabled': self.user.peak_alert_enabled,
            'email_notifications': self.user.email_notifications
        }
    
    # Analytics Operations
    def get_energy_statistics(self, days=30):
        """Get energy usage statistics"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            
            stats = self.session.query(
                func.sum(EnergyReading.total_generation_kw).label('total_generation'),
                func.sum(EnergyReading.consumption_kw).label('total_consumption'),
                func.sum(EnergyReading.solar_generation_kw).label('total_solar'),
                func.sum(EnergyReading.wind_generation_kw).label('total_wind'),
                func.sum(EnergyReading.grid_import_kw).label('total_grid_import'),
                func.sum(EnergyReading.grid_export_kw).label('total_grid_export'),
                func.sum(EnergyReading.carbon_saved_kg).label('total_carbon_saved'),
                func.avg(EnergyReading.battery_level_pct).label('avg_battery_level')
            ).filter(
                EnergyReading.user_id == self.user.id,
                EnergyReading.timestamp >= cutoff_time
            ).first()
            
            if stats:
                return {
                    'total_generation_kwh': (stats.total_generation or 0) / 60,  # Convert from kW-minutes to kWh
                    'total_consumption_kwh': (stats.total_consumption or 0) / 60,
                    'total_solar_kwh': (stats.total_solar or 0) / 60,
                    'total_wind_kwh': (stats.total_wind or 0) / 60,
                    'total_grid_import_kwh': (stats.total_grid_import or 0) / 60,
                    'total_grid_export_kwh': (stats.total_grid_export or 0) / 60,
                    'total_carbon_saved_kg': stats.total_carbon_saved or 0,
                    'avg_battery_level_pct': stats.avg_battery_level or 50,
                    'energy_independence_pct': ((stats.total_generation or 1) / (stats.total_consumption or 1)) * 100 if stats.total_consumption else 0,
                    'period_days': days
                }
            return {}
        except Exception as e:
            print(f"Error fetching energy statistics: {e}")
            return {}


# Global database instance
_db_instance = None

def get_db():
    """Get global database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = EnergyDatabase()
        _db_instance.connect()
    return _db_instance
