import math  # Import math module for mathematical operations
import csv  # Import csv module to read data from CSV files
from typing import List, Tuple, Callable  # Type hints for better code clarity and safety
from datetime import datetime, timedelta  # To handle date and time calculations

class RombergIntegrator:
    """
    Implements the Romberg Method for numerical integration
    using Richardson extrapolation to improve trapezoidal rule accuracy
    """
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 20):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def integrate(self, func: Callable[[float], float], a: float, b: float) -> Tuple[float, int]:
        """
        Perform Romberg integration of function from a to b
        Returns: (integral_value, iterations_used)
        """
        # Initialize Romberg table
        R = [[0.0 for _ in range(self.max_iterations)] for _ in range(self.max_iterations)]
        
        # First column: Trapezoidal rule with increasing subdivisions
        h = b - a
        R[0][0] = 0.5 * h * (func(a) + func(b))
        
        for i in range(1, self.max_iterations):
            h = h / 2.0
            sum_term = 0.0
            
            # Calculate sum of function values at new points
            for k in range(1, 2**i, 2):
                x = a + k * h
                sum_term += func(x)
            
            R[i][0] = 0.5 * R[i-1][0] + h * sum_term
            
            # Richardson extrapolation for higher order approximations
            for j in range(1, i + 1):
                R[i][j] = R[i][j-1] + (R[i][j-1] - R[i-1][j-1]) / (4**j - 1)
            
            # Check convergence
            if i > 0 and abs(R[i][i] - R[i-1][i-1]) < self.tolerance:
                return R[i][i], i + 1
        
        return R[self.max_iterations-1][self.max_iterations-1], self.max_iterations

class PollutionDataAnalyzer:
    """
    Analyzes pollution data using Romberg integration for various metrics
    """
    
    def __init__(self):
        self.romberg = RombergIntegrator()
        self.data_points = []
        self.time_points = []
    
    def generate_sample_data(self, days: int = 30) -> None:
        """Generate realistic pollution concentration data"""
        import random
        
        # Clear existing data
        self.data_points = []
        self.time_points = []
        
        # Generate data points (pollution concentration in µg/m³)
        for i in range(days * 24):  # Hourly data for specified days
            # Simulate daily pollution pattern with random variations
            hour_of_day = i % 24
            
            # Base pollution pattern (higher during day, lower at night)
            base_pollution = 30 + 15 * math.sin(math.pi * hour_of_day / 12)
            
            # Add weekly pattern (higher on weekdays)
            day_of_week = (i // 24) % 7
            weekly_factor = 1.2 if day_of_week < 5 else 0.8
            
            # Add random noise and occasional spikes
            noise = random.gauss(0, 5)
            spike = 20 if random.random() < 0.05 else 0  # 5% chance of pollution spike
            
            concentration = max(0, base_pollution * weekly_factor + noise + spike)
            
            self.data_points.append(concentration)
            self.time_points.append(i)
    
    def load_data_from_csv(self, filename: str) -> None:
        """Load pollution data from CSV file"""
        try:
            with open(filename, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                
                self.data_points = []
                self.time_points = []
                
                for i, row in enumerate(reader):
                    self.time_points.append(i)
                    self.data_points.append(float(row[1]))  # Assuming concentration in second column
                    
        except FileNotFoundError:
            print(f"File {filename} not found. Using generated sample data instead.")
            self.generate_sample_data()
    
    def create_interpolation_function(self) -> Callable[[float], float]:
        """Create piecewise linear interpolation function for the data"""
        def interpolate(t: float) -> float:
            if not self.data_points:
                return 0.0
            
            # Clamp to data range
            if t <= self.time_points[0]:
                return self.data_points[0]
            if t >= self.time_points[-1]:
                return self.data_points[-1]
            
            # Find interpolation points
            for i in range(len(self.time_points) - 1):
                if self.time_points[i] <= t <= self.time_points[i + 1]:
                    # Linear interpolation
                    t1, t2 = self.time_points[i], self.time_points[i + 1]
                    y1, y2 = self.data_points[i], self.data_points[i + 1]
                    
                    if t2 - t1 == 0:
                        return y1
                    
                    return y1 + (y2 - y1) * (t - t1) / (t2 - t1)
            
            return 0.0
        
        return interpolate
    
    def calculate_total_exposure(self, start_time: float = None, end_time: float = None) -> dict:
        """Calculate total pollution exposure over time period using Romberg integration"""
        if not self.data_points:
            return {"error": "No data available"}
        
        start_time = start_time or self.time_points[0]
        end_time = end_time or self.time_points[-1]
        
        interpolation_func = self.create_interpolation_function()
        
        total_exposure, iterations = self.romberg.integrate(
            interpolation_func, start_time, end_time
        )
        
        time_period = end_time - start_time
        average_concentration = total_exposure / time_period if time_period > 0 else 0
        
        return {
            "total_exposure": total_exposure,
            "average_concentration": average_concentration,
            "time_period": time_period,
            "integration_iterations": iterations,
            "start_time": start_time,
            "end_time": end_time
        }
    
    def analyze_peak_exposure(self, threshold: float = 50.0) -> dict:
        """Analyze exposure above a certain threshold"""
        if not self.data_points:
            return {"error": "No data available"}
        
        def excess_pollution(t: float) -> float:
            """Function representing pollution above threshold"""
            base_value = self.create_interpolation_function()(t)
            return max(0, base_value - threshold)
        
        total_excess, iterations = self.romberg.integrate(
            excess_pollution, 
            self.time_points[0], 
            self.time_points[-1]
        )
        
        # Count hours above threshold
        hours_above_threshold = sum(1 for conc in self.data_points if conc > threshold)
        
        return {
            "threshold": threshold,
            "total_excess_exposure": total_excess,
            "hours_above_threshold": hours_above_threshold,
            "percentage_above_threshold": (hours_above_threshold / len(self.data_points)) * 100,
            "integration_iterations": iterations
        }
    
    def calculate_daily_averages(self) -> List[dict]:
        """Calculate daily average concentrations using Romberg integration"""
        if not self.data_points:
            return []
        
        daily_averages = []
        interpolation_func = self.create_interpolation_function()
        
        # Group by days (assuming hourly data)
        total_hours = len(self.data_points)
        total_days = total_hours // 24
        
        for day in range(total_days):
            start_hour = day * 24
            end_hour = (day + 1) * 24
            
            if end_hour <= len(self.time_points):
                daily_integral, iterations = self.romberg.integrate(
                    interpolation_func, start_hour, end_hour
                )
                
                daily_average = daily_integral / 24  # Average over 24 hours
                
                daily_averages.append({
                    "day": day + 1,
                    "average_concentration": daily_average,
                    "integration_iterations": iterations
                })
        
        return daily_averages
    
    def generate_report(self) -> str:
        """Generate comprehensive pollution analysis report"""
        if not self.data_points:
            return "No data available for analysis."
        
        report = []
        report.append("=" * 60)
        report.append("POLLUTION DATA ANALYSIS REPORT")
        report.append("Using Romberg Method for Numerical Integration")
        report.append("=" * 60)
        
        # Basic statistics
        max_conc = max(self.data_points)
        min_conc = min(self.data_points)
        report.append(f"\nDATA SUMMARY:")
        report.append(f"Total data points: {len(self.data_points)}")
        report.append(f"Time period: {len(self.data_points)} hours ({len(self.data_points)/24:.1f} days)")
        report.append(f"Maximum concentration: {max_conc:.2f} µg/m³")
        report.append(f"Minimum concentration: {min_conc:.2f} µg/m³")
        
        # Total exposure analysis
        exposure_analysis = self.calculate_total_exposure()
        report.append(f"\nTOTAL EXPOSURE ANALYSIS:")
        report.append(f"Total exposure: {exposure_analysis['total_exposure']:.2f} µg·h/m³")
        report.append(f"Average concentration: {exposure_analysis['average_concentration']:.2f} µg/m³")
        report.append(f"Integration iterations: {exposure_analysis['integration_iterations']}")
        
        # Peak exposure analysis (WHO guideline: 25 µg/m³ for PM2.5)
        peak_analysis = self.analyze_peak_exposure(25.0)
        report.append(f"\nPEAK EXPOSURE ANALYSIS (Threshold: 25.0 µg/m³):")
        report.append(f"Total excess exposure: {peak_analysis['total_excess_exposure']:.2f} µg·h/m³")
        report.append(f"Hours above threshold: {peak_analysis['hours_above_threshold']}")
        report.append(f"Percentage above threshold: {peak_analysis['percentage_above_threshold']:.1f}%")
        
        # Daily averages
        daily_data = self.calculate_daily_averages()
        if daily_data:
            report.append(f"\nDAILY AVERAGE CONCENTRATIONS:")
            for day_info in daily_data[:7]:  # Show first 7 days
                report.append(f"Day {day_info['day']}: {day_info['average_concentration']:.2f} µg/m³")
            
            if len(daily_data) > 7:
                report.append(f"... and {len(daily_data) - 7} more days")
        
        # Health risk assessment
        avg_conc = exposure_analysis['average_concentration']
        report.append(f"\nHEALTH RISK ASSESSMENT:")
        if avg_conc > 35:
            risk_level = "HIGH"
        elif avg_conc > 25:
            risk_level = "MODERATE"
        elif avg_conc > 15:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        report.append(f"Risk Level: {risk_level}")
        report.append(f"WHO PM2.5 guideline (24h): 15 µg/m³")
        report.append(f"Your average: {avg_conc:.2f} µg/m³")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)

def create_sample_csv():
    """Create a sample CSV file with pollution data"""
    analyzer = PollutionDataAnalyzer()
    analyzer.generate_sample_data(30)  # 30 days of data
    
    with open('pollution_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Hour', 'Concentration_ugm3', 'Location'])
        
        for i, concentration in enumerate(analyzer.data_points):
            writer.writerow([i, f"{concentration:.2f}", "City_Center"])
    
    print("Sample CSV file 'pollution_data.csv' created successfully!")

def main():
    """Main execution function"""
    print("Pollution Data Analysis using Romberg Method")
    print("=" * 45)
    
    # Initialize analyzer
    analyzer = PollutionDataAnalyzer()
    
    # Try to load data from CSV, generate sample data if not found
    analyzer.load_data_from_csv('pollution_data.csv')
    
    if not analyzer.data_points:
        print("Generating sample pollution data...")
        analyzer.generate_sample_data(15)  # 15 days of sample data
    
    # Generate and display comprehensive report
    report = analyzer.generate_report()
    print(report)
    
    # Interactive analysis
    print("\nInteractive Analysis:")
    print("1. Analyze specific time period")
    print("2. Check different threshold levels")
    print("3. Create sample CSV file")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                start = float(input("Enter start time (hours): "))
                end = float(input("Enter end time (hours): "))
                
                result = analyzer.calculate_total_exposure(start, end)
                print(f"\nExposure Analysis for hours {start}-{end}:")
                print(f"Total exposure: {result['total_exposure']:.2f} µg·h/m³")
                print(f"Average concentration: {result['average_concentration']:.2f} µg/m³")
                print(f"Integration iterations: {result['integration_iterations']}")
                
            elif choice == '2':
                threshold = float(input("Enter threshold concentration (µg/m³): "))
                
                result = analyzer.analyze_peak_exposure(threshold)
                print(f"\nPeak Analysis for threshold {threshold} µg/m³:")
                print(f"Total excess exposure: {result['total_excess_exposure']:.2f} µg·h/m³")
                print(f"Hours above threshold: {result['hours_above_threshold']}")
                print(f"Percentage above threshold: {result['percentage_above_threshold']:.1f}%")
                
            elif choice == '3':
                create_sample_csv()
                
            elif choice == '4':
                print("Thank you for using the Pollution Data Analyzer!")
                break
                
            else:
                print("Invalid choice. Please enter 1-4.")
                
        except ValueError:
            print("Invalid input. Please enter numeric values.")
        except KeyboardInterrupt:
            print("\nProgram interrupted by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()