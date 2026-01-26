import MetaTrader5 as mt5
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
from dotenv import load_dotenv
import sys

load_dotenv()


class SimpleZonePlotter:
    """Simple plotter for supply/demand zones from CSV data."""
    
    def __init__(self, symbol='EURUSDm', timeframe=mt5.TIMEFRAME_M15, bars=3000):
        self.symbol = symbol
        self.timeframe = timeframe
        self.bars = bars
        self.df = None
        self.zones = []
    
    def init_mt5(self):
        """Initialize MT5."""
        path = os.getenv("MT5_PATH")
        login = os.getenv("MT5_LOGIN")
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")
        params = {}
        if path: params["path"] = path
        
        if not mt5.initialize(**params):
            print(f"❌ MT5 Init failed")
            return False
        if login and password and server:
            mt5.login(login=int(login), password=password, server=server)
        print(f"✅ Connected to MT5")
        return True
    
    def fetch_data(self):
        """Fetch historical price data."""
        print(f"📊 Fetching {self.bars} bars for {self.symbol}...")
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, self.bars)
        
        if rates is None:
            print(f"❌ Failed to fetch data")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        print(f"✅ Loaded {len(df)} bars")
        self.df = df
        return df
    
    def load_zones_from_csv(self, filename='supply_demand_zones_bayesian.csv'):
        """Load zone data from CSV file."""
        if not os.path.exists(filename):
            print(f"❌ Zone file not found: {filename}")
            print(f"   Please run detect_supply_demand_zones.py first to generate zone data.")
            return False
        
        print(f"📂 Loading zones from {filename}...")
        zones_df = pd.DataFrame(pd.read_csv(filename))
        
        # Convert to list of dictionaries
        self.zones = zones_df.to_dict('records')
        
        # Convert time strings back to datetime
        for zone in self.zones:
            if 'time' in zone and isinstance(zone['time'], str):
                zone['time'] = pd.to_datetime(zone['time'])
        
        print(f"✅ Loaded {len(self.zones)} zones")
        print(f"   Supply zones: {sum(1 for z in self.zones if z['type'] == 'supply')}")
        print(f"   Demand zones: {sum(1 for z in self.zones if z['type'] == 'demand')}")
        
        return True
    
    def plot_zones(self):
        """Plot zones on price chart."""
        if self.df is None or len(self.df) == 0:
            print("❌ No price data available")
            return
        
        if not self.zones:
            print("❌ No zones to plot")
            return
        
        print(f"\n📈 Generating interactive chart...")
        
        # Create figure
        fig = go.Figure()
        
        # 1. Add Candlestick Chart
        fig.add_trace(go.Candlestick(
            x=self.df.index,
            open=self.df['open'], 
            high=self.df['high'],
            low=self.df['low'], 
            close=self.df['close'],
            name='Price'
        ))
        
        # 2. Add Zones
        sorted_zones = sorted(self.zones, key=lambda x: x.get('composite_score', 0), reverse=True)
        
        # Lists for hover scatter points
        hover_x = []
        hover_y = []
        hover_text = []
        hover_colors = []
        
        last_time = self.df.index[-1]
        
        for zone in sorted_zones:
            zone_time = zone['time']
            zone_price = zone['price']
            score = zone.get('composite_score', zone.get('strength', 0))
            z_type = zone['type']
            
            # Color logic based on composite score
            if score >= 0.8:
                color = 'rgba(0, 255, 0, 0.4)' if z_type == 'demand' else 'rgba(255, 0, 0, 0.4)'
                line_color = '#00ff00' if z_type == 'demand' else '#ff0000'
            elif score >= 0.6:
                color = 'rgba(144, 238, 144, 0.4)' if z_type == 'demand' else 'rgba(255, 107, 107, 0.4)'
                line_color = '#90ee90' if z_type == 'demand' else '#ff6b6b'
            elif score >= 0.4:
                color = 'rgba(255, 255, 0, 0.3)'
                line_color = '#ffff00'
            else:
                color = 'rgba(128, 128, 128, 0.3)'
                line_color = '#808080'
            
            # Define Zone Height (0.2% thickness)
            height = zone_price * 0.002
            y0 = zone_price - height/2
            y1 = zone_price + height/2
            
            # Width of the initial block (48 hours equivalent)
            t_delta = pd.Timedelta(hours=48)
            x0 = zone_time
            x1 = zone_time + t_delta
            
            # A. Draw the Zone Box (Origin)
            fig.add_shape(type="rect",
                x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color=line_color, width=1),
                fillcolor=color
            )
            
            # B. Draw Extension Line (from origin to end of data)
            fig.add_shape(type="line",
                x0=x0, y0=zone_price, x1=last_time, y1=zone_price,
                line=dict(color=line_color, width=1, dash="dash"),
                opacity=0.5
            )
            
            # C. Collect Data for Hover Labels
            hover_x.append(zone_time)
            hover_y.append(zone_price)
            hover_colors.append(line_color)
            
            mtf_str = f"MTF: {zone.get('num_timeframes', 1)} TFs"
            hover_text.append(
                f"<b>{z_type.upper()}</b><br>"
                f"Price: {zone_price:.5f}<br>"
                f"Score: {score:.2f}<br>"
                f"Touches: {zone.get('touches', 'N/A')}<br>"
                f"{mtf_str}"
            )
        
        # 3. Add Scatter for Hover Info
        fig.add_trace(go.Scatter(
            x=hover_x, y=hover_y,
            mode='markers',
            marker=dict(size=10, color=hover_colors, symbol='diamond'),
            text=hover_text,
            hoverinfo='text',
            name='Zone Info'
        ))
        
        # 4. Layout Styling
        fig.update_layout(
            template='plotly_dark',
            height=800,
            title_text=f"{self.symbol} - Supply/Demand Zones ({len(self.df)} bars)",
            xaxis_rangeslider_visible=False,
            hovermode='closest',
            xaxis_title="Time",
            yaxis_title="Price"
        )
        
        fig.show()
        print("✅ Interactive chart opened in browser")


def main():
    print("="*60)
    print("  SIMPLE ZONE VISUALIZATION")
    print("  Plots zones from CSV on price chart")
    print("="*60)
    
    # Configuration
    symbol = 'EURUSDm'
    timeframe = mt5.TIMEFRAME_M15
    bars = 3000
    zone_file = 'supply_demand_zones_bayesian.csv'
    
    # Create plotter
    plotter = SimpleZonePlotter(symbol, timeframe, bars)
    
    # Initialize MT5
    if not plotter.init_mt5():
        return
    
    # Fetch price data
    if plotter.fetch_data() is None:
        return
    
    # Load zones from CSV
    if not plotter.load_zones_from_csv(zone_file):
        return
    
    # Plot zones
    plotter.plot_zones()
    
    # Cleanup
    mt5.shutdown()
    print(f"\n👋 Complete")


if __name__ == "__main__":
    main()
