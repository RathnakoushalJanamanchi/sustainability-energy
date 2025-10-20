"""
Database models for sustainable energy platform
"""
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

Base = declarative_base()

class User(Base):
    """User profile and preferences"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Energy system configuration
    solar_capacity_kw = Column(Float, default=100.0)
    wind_capacity_kw = Column(Float, default=80.0)
    battery_capacity_kwh = Column(Float, default=100.0)
    
    # Goals and preferences
    monthly_savings_goal = Column(Float, default=200.0)
    carbon_reduction_goal_kg = Column(Float, default=500.0)
    energy_independence_goal_pct = Column(Float, default=80.0)
    
    # Preferences
    auto_trading_enabled = Column(Boolean, default=True)
    peak_alert_enabled = Column(Boolean, default=True)
    email_notifications = Column(Boolean, default=True)
    
    # Relationships
    energy_readings = relationship("EnergyReading", back_populates="user")
    transactions = relationship("Transaction", back_populates="user")
    trading_history = relationship("TradingHistory", back_populates="user")


class EnergyReading(Base):
    """Real-time energy generation and consumption data"""
    __tablename__ = 'energy_readings'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Generation data
    solar_generation_kw = Column(Float, nullable=False)
    wind_generation_kw = Column(Float, nullable=False)
    total_generation_kw = Column(Float, nullable=False)
    
    # Consumption data
    consumption_kw = Column(Float, nullable=False)
    
    # Grid interaction
    grid_import_kw = Column(Float, default=0.0)
    grid_export_kw = Column(Float, default=0.0)
    
    # Battery data
    battery_level_pct = Column(Float, default=50.0)
    battery_charge_kw = Column(Float, default=0.0)
    battery_discharge_kw = Column(Float, default=0.0)
    
    # Pricing and carbon
    price_per_kwh = Column(Float, nullable=False)
    carbon_intensity_kg_per_kwh = Column(Float, default=0.45)
    
    # Calculated fields
    net_energy_kw = Column(Float)
    cost_usd = Column(Float)
    revenue_usd = Column(Float)
    carbon_saved_kg = Column(Float)
    
    # Relationships
    user = relationship("User", back_populates="energy_readings")


class Transaction(Base):
    """Financial transactions for energy purchases and sales"""
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Transaction details
    transaction_type = Column(String(20), nullable=False)  # 'purchase', 'sale', 'fee', 'maintenance'
    description = Column(Text)
    
    # Financial data
    energy_kwh = Column(Float, default=0.0)
    price_per_kwh = Column(Float)
    amount_usd = Column(Float, nullable=False)
    
    # Status
    status = Column(String(20), default='completed')  # 'pending', 'completed', 'failed'
    payment_method = Column(String(50))
    transaction_id = Column(String(100), unique=True)
    
    # Relationships
    user = relationship("User", back_populates="transactions")


class TradingHistory(Base):
    """Energy trading history and opportunities"""
    __tablename__ = 'trading_history'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Trading details
    trade_type = Column(String(10), nullable=False)  # 'buy' or 'sell'
    energy_kwh = Column(Float, nullable=False)
    price_per_kwh = Column(Float, nullable=False)
    total_amount_usd = Column(Float, nullable=False)
    
    # Market data
    market_price_at_trade = Column(Float)
    profit_loss_usd = Column(Float)
    
    # Trading partner (for P2P trading)
    counterparty_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    
    # Status
    status = Column(String(20), default='completed')
    trade_id = Column(String(100), unique=True)
    
    # Relationships
    user = relationship("User", back_populates="trading_history", foreign_keys="[TradingHistory.user_id]")


class WeatherData(Base):
    """Historical weather data for forecasting"""
    __tablename__ = 'weather_data'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    location = Column(String(100), default='default')
    
    # Weather parameters
    temperature_c = Column(Float)
    humidity_pct = Column(Float)
    cloud_cover_pct = Column(Float)
    wind_speed_ms = Column(Float)
    solar_irradiance_w_m2 = Column(Float)
    
    # Forecast or actual
    is_forecast = Column(Boolean, default=False)
    forecast_hours_ahead = Column(Integer, default=0)


class EnergyForecast(Base):
    """AI-generated energy forecasts"""
    __tablename__ = 'energy_forecasts'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    forecast_timestamp = Column(DateTime, nullable=False, index=True)
    
    # Forecasted values
    solar_forecast_kw = Column(Float)
    wind_forecast_kw = Column(Float)
    consumption_forecast_kw = Column(Float)
    price_forecast_usd_per_kwh = Column(Float)
    
    # Model confidence
    confidence_pct = Column(Float)
    model_version = Column(String(50))


class SystemAlert(Base):
    """System alerts and notifications"""
    __tablename__ = 'system_alerts'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Alert details
    alert_type = Column(String(50), nullable=False)  # 'peak_pricing', 'low_battery', 'trading_opportunity', etc.
    severity = Column(String(20), default='info')  # 'info', 'warning', 'critical'
    title = Column(String(200))
    message = Column(Text)
    
    # Status
    is_read = Column(Boolean, default=False)
    is_dismissed = Column(Boolean, default=False)
    action_taken = Column(Text)


# Database engine and session factory
def get_engine():
    """Get database engine"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    return create_engine(database_url, echo=False)


def get_session():
    """Get database session"""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_db():
    """Initialize database - create all tables"""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("Database initialized successfully!")


def get_or_create_default_user(session):
    """Get or create default user for single-user deployment"""
    user = session.query(User).filter_by(username='default_user').first()
    
    if not user:
        user = User(
            username='default_user',
            email='user@energy-platform.com',
            solar_capacity_kw=100.0,
            wind_capacity_kw=80.0,
            battery_capacity_kwh=100.0,
            monthly_savings_goal=200.0,
            carbon_reduction_goal_kg=500.0,
            energy_independence_goal_pct=80.0
        )
        session.add(user)
        session.commit()
        print(f"Created default user with ID: {user.id}")
    
    return user
