import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class FinancialCalculator:
    """Comprehensive financial calculations for energy systems"""
    
    def __init__(self):
        self.base_rate = 0.12  # Base electricity rate $/kWh
        self.feed_in_tariff = 0.08  # Rate for selling back to grid $/kWh
        self.peak_rate_multiplier = 1.5  # Peak hour rate multiplier
        self.off_peak_rate_multiplier = 0.8  # Off-peak hour rate multiplier
        
    def calculate_energy_costs(self, consumption_kwh, grid_import_kwh, hour_of_day):
        """Calculate energy costs based on consumption and time of day pricing"""
        # Determine rate based on time of day
        if 17 <= hour_of_day <= 20:  # Peak hours
            rate = self.base_rate * self.peak_rate_multiplier
        elif 22 <= hour_of_day <= 6:  # Off-peak hours
            rate = self.base_rate * self.off_peak_rate_multiplier
        else:  # Standard hours
            rate = self.base_rate
        
        # Calculate cost for grid imports only
        cost = grid_import_kwh * rate
        return {
            'total_cost': cost,
            'rate_per_kwh': rate,
            'grid_import_kwh': grid_import_kwh,
            'rate_category': self._get_rate_category(hour_of_day)
        }
    
    def calculate_energy_revenue(self, excess_generation_kwh, hour_of_day):
        """Calculate revenue from selling excess energy back to grid"""
        # Feed-in tariff is typically lower than purchase rate
        revenue = excess_generation_kwh * self.feed_in_tariff
        return {
            'total_revenue': revenue,
            'feed_in_rate': self.feed_in_tariff,
            'excess_kwh': excess_generation_kwh
        }
    
    def calculate_net_savings(self, generation_kwh, consumption_kwh, hour_of_day):
        """Calculate net savings from renewable energy generation"""
        if generation_kwh >= consumption_kwh:
            # Surplus generation
            self_consumption = consumption_kwh
            excess = generation_kwh - consumption_kwh
            
            # Value of self-consumed energy (avoided grid purchase)
            avoided_cost = self_consumption * self._get_rate_for_hour(hour_of_day)
            
            # Revenue from excess energy
            excess_revenue = self.calculate_energy_revenue(excess, hour_of_day)['total_revenue']
            
            total_savings = avoided_cost + excess_revenue
            grid_cost = 0
            
        else:
            # Need to import from grid
            self_consumption = generation_kwh
            grid_import = consumption_kwh - generation_kwh
            
            # Value of self-consumed energy
            avoided_cost = self_consumption * self._get_rate_for_hour(hour_of_day)
            
            # Cost of grid import
            grid_cost_data = self.calculate_energy_costs(consumption_kwh, grid_import, hour_of_day)
            grid_cost = grid_cost_data['total_cost']
            
            total_savings = avoided_cost - grid_cost
            excess_revenue = 0
        
        return {
            'net_savings': total_savings,
            'avoided_costs': avoided_cost,
            'excess_revenue': excess_revenue if 'excess_revenue' in locals() else 0,
            'grid_costs': grid_cost,
            'self_consumption_kwh': self_consumption,
            'excess_generation_kwh': excess if 'excess' in locals() else 0,
            'grid_import_kwh': grid_import if 'grid_import' in locals() else 0
        }
    
    def calculate_roi(self, initial_investment, annual_savings):
        """Calculate Return on Investment metrics"""
        if annual_savings <= 0:
            return {
                'roi_percentage': 0,
                'payback_period_years': float('inf'),
                'npv_10_years': -initial_investment,
                'irr': 0
            }
        
        roi_percentage = (annual_savings / initial_investment) * 100
        payback_period = initial_investment / annual_savings
        
        # Net Present Value calculation (10 years, 5% discount rate)
        discount_rate = 0.05
        npv = -initial_investment
        for year in range(1, 11):
            npv += annual_savings / ((1 + discount_rate) ** year)
        
        # Simple IRR approximation
        irr = annual_savings / initial_investment
        
        return {
            'roi_percentage': roi_percentage,
            'payback_period_years': payback_period,
            'npv_10_years': npv,
            'irr': irr,
            'monthly_savings': annual_savings / 12,
            'break_even_year': int(payback_period) + 1 if payback_period != float('inf') else None
        }
    
    def calculate_carbon_financial_impact(self, carbon_saved_kg, carbon_price_per_ton=50):
        """Calculate financial impact of carbon savings"""
        carbon_tons = carbon_saved_kg / 1000
        carbon_value = carbon_tons * carbon_price_per_ton
        
        return {
            'carbon_tons_saved': carbon_tons,
            'carbon_financial_value': carbon_value,
            'carbon_price_per_ton': carbon_price_per_ton
        }
    
    def analyze_energy_trading_opportunities(self, generation_forecast, consumption_forecast, price_forecast):
        """Analyze energy trading opportunities based on forecasts"""
        trading_opportunities = []
        
        for i, (gen, cons, price) in enumerate(zip(generation_forecast, consumption_forecast, price_forecast)):
            hour = i % 24
            net_energy = gen - cons
            
            if net_energy > 5:  # Surplus threshold
                opportunity = {
                    'hour': hour,
                    'type': 'sell',
                    'energy_kwh': net_energy,
                    'price_per_kwh': price,
                    'potential_revenue': net_energy * price,
                    'recommendation': 'Sell excess energy' if price > self.base_rate * 1.2 else 'Store for later'
                }
            elif net_energy < -5:  # Deficit threshold
                opportunity = {
                    'hour': hour,
                    'type': 'buy',
                    'energy_kwh': abs(net_energy),
                    'price_per_kwh': price,
                    'cost': abs(net_energy) * price,
                    'recommendation': 'Buy from grid' if price < self.base_rate * 1.1 else 'Reduce consumption'
                }
            else:
                opportunity = {
                    'hour': hour,
                    'type': 'balanced',
                    'energy_kwh': abs(net_energy),
                    'recommendation': 'Maintain current usage'
                }
            
            trading_opportunities.append(opportunity)
        
        return trading_opportunities
    
    def calculate_monthly_bill_projection(self, daily_consumption, daily_generation, days_in_month=30):
        """Calculate projected monthly electricity bill"""
        total_consumption = daily_consumption * days_in_month
        total_generation = daily_generation * days_in_month
        
        # Assume mixed time-of-day usage
        peak_consumption = total_consumption * 0.3  # 30% during peak
        standard_consumption = total_consumption * 0.5  # 50% during standard
        off_peak_consumption = total_consumption * 0.2  # 20% during off-peak
        
        # Calculate costs without solar
        baseline_cost = (
            peak_consumption * self.base_rate * self.peak_rate_multiplier +
            standard_consumption * self.base_rate +
            off_peak_consumption * self.base_rate * self.off_peak_rate_multiplier
        )
        
        # Calculate with renewable generation
        net_consumption = max(0, total_consumption - total_generation)
        excess_generation = max(0, total_generation - total_consumption)
        
        # Proportional reduction in each time period
        reduction_factor = net_consumption / total_consumption if total_consumption > 0 else 0
        
        actual_cost = (
            peak_consumption * reduction_factor * self.base_rate * self.peak_rate_multiplier +
            standard_consumption * reduction_factor * self.base_rate +
            off_peak_consumption * reduction_factor * self.base_rate * self.off_peak_rate_multiplier
        )
        
        # Revenue from excess
        excess_revenue = excess_generation * self.feed_in_tariff
        
        net_bill = actual_cost - excess_revenue
        monthly_savings = baseline_cost - net_bill
        
        return {
            'baseline_monthly_bill': baseline_cost,
            'actual_monthly_bill': max(0, net_bill),  # Can't be negative
            'monthly_savings': monthly_savings,
            'excess_revenue': excess_revenue,
            'savings_percentage': (monthly_savings / baseline_cost * 100) if baseline_cost > 0 else 0,
            'net_consumption_kwh': net_consumption,
            'excess_generation_kwh': excess_generation
        }
    
    def calculate_system_performance_metrics(self, generation_data, consumption_data, investment_amount):
        """Calculate comprehensive system performance metrics"""
        total_generation = sum(generation_data)
        total_consumption = sum(consumption_data)
        
        # Energy metrics
        capacity_factor = (total_generation / (max(generation_data) * len(generation_data))) * 100 if generation_data else 0
        self_sufficiency = min(100, (total_generation / total_consumption) * 100) if total_consumption > 0 else 0
        
        # Financial metrics
        annual_generation = total_generation * (365 / len(generation_data)) if generation_data else 0
        annual_consumption = total_consumption * (365 / len(consumption_data)) if consumption_data else 0
        
        # Estimate annual savings
        annual_avoided_costs = annual_generation * self.base_rate  # Simplified calculation
        roi_data = self.calculate_roi(investment_amount, annual_avoided_costs)
        
        return {
            'capacity_factor': capacity_factor,
            'self_sufficiency_percentage': self_sufficiency,
            'annual_generation_kwh': annual_generation,
            'annual_consumption_kwh': annual_consumption,
            'annual_savings_estimate': annual_avoided_costs,
            'roi_data': roi_data,
            'system_efficiency': min(100, (total_generation / (total_generation + total_consumption)) * 200) if (total_generation + total_consumption) > 0 else 0
        }
    
    def _get_rate_for_hour(self, hour):
        """Get electricity rate for specific hour"""
        if 17 <= hour <= 20:
            return self.base_rate * self.peak_rate_multiplier
        elif 22 <= hour <= 6:
            return self.base_rate * self.off_peak_rate_multiplier
        else:
            return self.base_rate
    
    def _get_rate_category(self, hour):
        """Get rate category name for specific hour"""
        if 17 <= hour <= 20:
            return "Peak"
        elif 22 <= hour <= 6:
            return "Off-Peak"
        else:
            return "Standard"
    
    def generate_payment_history(self, days=30):
        """Generate realistic payment history for demonstration"""
        payments = []
        current_date = datetime.now()
        
        for i in range(days):
            date = current_date - timedelta(days=i)
            
            # Simulate different types of transactions
            transaction_type = random.choices(
                ['grid_purchase', 'grid_sale', 'maintenance', 'system_fee'],
                weights=[40, 30, 20, 10]
            )[0]
            
            if transaction_type == 'grid_purchase':
                amount = random.uniform(15, 45)
                description = "Grid electricity purchase"
                transaction_id = f"GP{random.randint(10000, 99999)}"
            elif transaction_type == 'grid_sale':
                amount = -random.uniform(8, 25)  # Negative = credit
                description = "Excess energy sale to grid"
                transaction_id = f"GS{random.randint(10000, 99999)}"
            elif transaction_type == 'maintenance':
                amount = random.uniform(25, 100)
                description = "System maintenance fee"
                transaction_id = f"MN{random.randint(10000, 99999)}"
            else:  # system_fee
                amount = random.uniform(5, 15)
                description = "Grid connection fee"
                transaction_id = f"SF{random.randint(10000, 99999)}"
            
            payments.append({
                'date': date.strftime('%Y-%m-%d'),
                'transaction_id': transaction_id,
                'type': transaction_type.replace('_', ' ').title(),
                'description': description,
                'amount': round(amount, 2),
                'status': 'Completed',
                'balance_impact': 'Credit' if amount < 0 else 'Debit'
            })
        
        return sorted(payments, key=lambda x: x['date'], reverse=True)
    
    def calculate_optimal_battery_sizing(self, daily_generation, daily_consumption, battery_cost_per_kwh=500):
        """Calculate optimal battery storage capacity"""
        # Find typical daily surplus/deficit patterns
        daily_surplus = max(0, daily_generation - daily_consumption)
        daily_deficit = max(0, daily_consumption - daily_generation)
        
        # Optimal battery size is typically 1-2 days of average deficit or surplus
        recommended_capacity = max(daily_surplus, daily_deficit) * 1.5
        
        # Financial analysis
        battery_cost = recommended_capacity * battery_cost_per_kwh
        
        # Estimate annual benefit (store cheap energy, use during expensive periods)
        daily_arbitrage_benefit = recommended_capacity * (self.base_rate * self.peak_rate_multiplier - self.base_rate * self.off_peak_rate_multiplier)
        annual_benefit = daily_arbitrage_benefit * 365
        
        payback_period = battery_cost / annual_benefit if annual_benefit > 0 else float('inf')
        
        return {
            'recommended_capacity_kwh': recommended_capacity,
            'battery_cost': battery_cost,
            'annual_benefit': annual_benefit,
            'payback_period_years': payback_period,
            'daily_arbitrage_benefit': daily_arbitrage_benefit,
            'cost_per_kwh': battery_cost_per_kwh
        }

