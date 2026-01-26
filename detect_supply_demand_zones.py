import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import talib
from datetime import datetime
import os
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# Import regime labeler
from regime_labeler import RegimeLabelGenerator

# Import Bayesian zone detector
from bayesian_supply_demand_zones import BayesianZoneDetector, GaussianKDEZoneDetector

class SupplyDemandZoneDetector:
    """Detect and label supply/demand zones during sideways regimes."""
    
    def __init__(self, symbol='BTCUSDm', timeframe=mt5.TIMEFRAME_M15, bars=3000,
                 mtf_timeframes=None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.bars = bars
        self.df = None
        self.zones = []
        self.bayesian_detector = None
        self.kde_detector = None
        self.zone_probabilities = {}
        
        # Multi-timeframe confluence settings
        if mtf_timeframes is None:
            self.mtf_timeframes = [
                mt5.TIMEFRAME_M15,  # Base timeframe
                mt5.TIMEFRAME_H1,   # 4x base
                mt5.TIMEFRAME_H4    # 16x base
            ]
        else:
            self.mtf_timeframes = mtf_timeframes
        
        self.mtf_zones = {}  # Store zones for each timeframe
        self.mtf_tolerance = 0.002  # 0.2% price tolerance for alignment
        
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
        """Fetch historical data."""
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
    
    def identify_sideways_periods(self):
        """Identify sideways regimes using consensus method."""
        print(f"🔍 Identifying sideways periods...")
        
        labeler = RegimeLabelGenerator(self.symbol, self.timeframe, self.bars)
        labeler.df = self.df
        
        result = labeler.method_consensus()
        sideways_mask = result['labels'] == 'sideways'
        
        sideways_pct = sideways_mask.sum() / len(sideways_mask) * 100
        print(f"   Sideways: {sideways_mask.sum()} bars ({sideways_pct:.1f}%)")
        
        return sideways_mask
    
    def find_swing_points(self, window=5):
        """Find swing highs and lows."""
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(self.df) - window):
            current_high = self.df['high'].iloc[i]
            current_low = self.df['low'].iloc[i]
            
            # Swing high: higher than all surrounding candles
            left_highs = self.df['high'].iloc[i-window:i]
            right_highs = self.df['high'].iloc[i+1:i+window+1]
            
            if (current_high >= left_highs.max()) and (current_high >= right_highs.max()):
                swing_highs.append(i)
            
            # Swing low: lower than all surrounding candles
            left_lows = self.df['low'].iloc[i-window:i]
            right_lows = self.df['low'].iloc[i+1:i+window+1]
            
            if (current_low <= left_lows.min()) and (current_low <= right_lows.min()):
                swing_lows.append(i)
        
        return swing_highs, swing_lows
    
    def validate_reversal(self, idx, zone_type, reversal_threshold=0.002, lookback=20):
        """
        Validate if a zone caused a reversal.
        
        Args:
            idx: Index of potential zone
            zone_type: 'supply' or 'demand'
            reversal_threshold: Minimum price move (%) to confirm reversal
            lookback: How many candles to look ahead for reversal
            
        Returns:
            (is_valid, reversal_strength, validation_idx)
            validation_idx: The index where the zone was confirmed (idx + lookback)
        """
        if idx + lookback >= len(self.df):
            return False, 0, idx
        
        zone_price = self.df['high'].iloc[idx] if zone_type == 'supply' else self.df['low'].iloc[idx]
        
        # Look ahead for reversal
        future_prices = self.df['close'].iloc[idx+1:idx+lookback+1]
        
        # Validation happens at idx + lookback (when we can confirm the reversal)
        validation_idx = idx + lookback
        
        if zone_type == 'supply':
            # For supply, price should fall
            max_drop = (zone_price - future_prices.min()) / zone_price
            if max_drop >= reversal_threshold:
                return True, max_drop, validation_idx
        else:  # demand
            # For demand, price should rise
            max_rise = (future_prices.max() - zone_price) / zone_price
            if max_rise >= reversal_threshold:
                return True, max_rise, validation_idx
        
        return False, 0, idx
    
    def detect_zones(self, sideways_mask, reversal_threshold=0.002, min_touches=2):
        """
        Detect supply/demand zones during sideways periods.
        
        IMPORTANT: Zones are displayed at their VALIDATION time (T+20), not formation time (T).
        This eliminates lookahead bias - you would only know a zone is valid after the reversal
        is confirmed, which takes 20 candles.
        
        Returns list of zones with:
        - type: 'supply' or 'demand'
        - price: zone price level
        - time: when zone was VALIDATED (confirmed), not when it formed
        - formation_time: original formation time for reference
        - strength: reversal magnitude
        - touches: number of times price tested the zone
        - validated: if zone caused reversal
        """
        print(f"\n🎯 Detecting supply/demand zones...")
        
        # Find swing points
        swing_highs, swing_lows = self.find_swing_points(window=5)
        
        zones = []
        
        # Process swing highs (potential supply zones)
        for idx in swing_highs:
            # Only consider if in sideways period
            if not sideways_mask.iloc[idx]:
                continue
            
            zone_price = self.df['high'].iloc[idx]
            zone_time = self.df.index[idx]
            
            # Validate reversal
            is_valid, reversal_strength, validation_idx = self.validate_reversal(idx, 'supply', reversal_threshold)
            
            if is_valid:
                # Count touches (how many times price came close to this level)
                price_tolerance = zone_price * 0.001  # 0.1% tolerance
                touches = ((self.df['high'].iloc[idx:idx+50] >= zone_price - price_tolerance) & 
                          (self.df['high'].iloc[idx:idx+50] <= zone_price + price_tolerance)).sum()
                
                if touches >= min_touches:
                    # Use validation time instead of formation time (eliminates lookahead bias)
                    validation_time = self.df.index[validation_idx]
                    
                    zones.append({
                        'type': 'supply',
                        'price': zone_price,
                        'time': validation_time,  # Time when zone was confirmed, not formed
                        'formation_time': zone_time,  # Original formation time for reference
                        'idx': idx,
                        'validation_idx': validation_idx,
                        'strength': reversal_strength,
                        'touches': touches,
                        'validated': True
                    })
        
        # Process swing lows (potential demand zones)
        for idx in swing_lows:
            # Only consider if in sideways period
            if not sideways_mask.iloc[idx]:
                continue
            
            zone_price = self.df['low'].iloc[idx]
            zone_time = self.df.index[idx]
            
            # Validate reversal
            is_valid, reversal_strength, validation_idx = self.validate_reversal(idx, 'demand', reversal_threshold)
            
            if is_valid:
                # Count touches
                price_tolerance = zone_price * 0.001
                touches = ((self.df['low'].iloc[idx:idx+50] >= zone_price - price_tolerance) & 
                          (self.df['low'].iloc[idx:idx+50] <= zone_price + price_tolerance)).sum()
                
                if touches >= min_touches:
                    # Use validation time instead of formation time (eliminates lookahead bias)
                    validation_time = self.df.index[validation_idx]
                    
                    zones.append({
                        'type': 'demand',
                        'price': zone_price,
                        'time': validation_time,  # Time when zone was confirmed, not formed
                        'formation_time': zone_time,  # Original formation time for reference
                        'idx': idx,
                        'validation_idx': validation_idx,
                        'strength': reversal_strength,
                        'touches': touches,
                        'validated': True
                    })
        
        # Sort by time
        zones.sort(key=lambda x: x['time'])
        
        print(f"✅ Found {len(zones)} validated zones")
        print(f"   Supply zones: {sum(1 for z in zones if z['type'] == 'supply')}")
        print(f"   Demand zones: {sum(1 for z in zones if z['type'] == 'demand')}")
        
        self.zones = zones
        
        # Initialize and update Bayesian probabilities
        if self.bayesian_detector is None:
            self.initialize_bayesian_detector()
        
        self.update_bayesian_probabilities()
        self.score_zones_with_bayesian()
        
        return zones
    
    def cluster_zones(self, zones, n_clusters=10):
        """
        Cluster nearby zones to find key levels.
        Useful for identifying the most important support/resistance.
        """
        if len(zones) < n_clusters:
            n_clusters = max(2, len(zones) // 2)
        
        prices = np.array([z['price'] for z in zones]).reshape(-1, 1)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(prices)
        
        # Add cluster info to zones
        for i, zone in enumerate(zones):
            zone['cluster'] = clusters[i]
            zone['cluster_center'] = kmeans.cluster_centers_[clusters[i]][0]
        
        print(f"\n📊 Clustered zones into {n_clusters} key levels")
        
        return zones
    
    def initialize_bayesian_detector(self):
        """Initialize Bayesian zone detector with price range from data."""
        print(f"\n🧮 Initializing Bayesian probability detector...")
        
        price_min = self.df['low'].min() - (self.df['low'].min() * 0.01)
        price_max = self.df['high'].max() + (self.df['high'].max() * 0.01)
        
        self.bayesian_detector = BayesianZoneDetector(
            price_min=price_min,
            price_max=price_max,
            n_bins=200,
            rejection_likelihood=0.85,
            breakthrough_likelihood=0.15,
            decay_rate=0.98
        )
        
        # Set volume profile prior
        self.bayesian_detector.set_volume_profile_prior(
            self.df['close'].values,
            self.df['tick_volume'].values
        )
        
        # Initialize KDE detector
        self.kde_detector = GaussianKDEZoneDetector(bandwidth=2.0)
        
        print(f"   ✅ Bayesian detector initialized")
        print(f"   Price range: {price_min:.5f} - {price_max:.5f}")
    
    def update_bayesian_probabilities(self):
        """Update Bayesian probabilities based on price action."""
        print(f"\n📊 Updating Bayesian probabilities from price action...")
        
        from bayesian_supply_demand_zones import calculate_rejection_strength
        
        # Process each candle to update probabilities
        update_count = 0
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            
            # Detect rejection (large wick)
            rejection_strength = calculate_rejection_strength(row)
            
            if rejection_strength > 0.5:
                # Strong rejection at high or low
                upper_wick = row['high'] - max(row['open'], row['close'])
                lower_wick = min(row['open'], row['close']) - row['low']
                
                if upper_wick > lower_wick:
                    # Upper wick rejection (supply zone)
                    self.bayesian_detector.update_rejection(row['high'], rejection_strength)
                    self.kde_detector.add_reversal(row['high'], rejection_strength)
                else:
                    # Lower wick rejection (demand zone)
                    self.bayesian_detector.update_rejection(row['low'], rejection_strength)
                    self.kde_detector.add_reversal(row['low'], rejection_strength)
                
                update_count += 1
            
            # Detect breakthrough (strong momentum candle)
            body_size = abs(row['close'] - row['open'])
            total_range = row['high'] - row['low']
            
            if total_range > 0 and body_size / total_range > 0.7:
                # Strong momentum candle
                momentum = body_size / total_range
                mid_price = (row['high'] + row['low']) / 2
                self.bayesian_detector.update_breakthrough(mid_price, momentum)
            
            # Apply time decay every 10 candles
            if i % 10 == 0:
                self.bayesian_detector.apply_time_decay()
        
        print(f"   ✅ Processed {len(self.df)} candles, {update_count} probability updates")
    
    def score_zones_with_bayesian(self):
        """Score traditional zones with Bayesian probabilities."""
        print(f"\n🎯 Scoring zones with Bayesian probabilities...")
        
        if not self.zones:
            return
        
        # Get KDE density across price range
        price_range = np.linspace(
            self.bayesian_detector.price_min,
            self.bayesian_detector.price_max,
            500
        )
        kde_density = self.kde_detector.get_probability_density(price_range)
        max_kde = kde_density.max() if kde_density.max() > 0 else 1.0
        
        for zone in self.zones:
            zone_price = zone['price']
            
            # Get Bayesian probability
            bin_idx = self.bayesian_detector._get_bin_index(zone_price)
            if bin_idx is not None:
                bayesian_prob = self.bayesian_detector.p_zone[bin_idx]
            else:
                bayesian_prob = 0.1
            
            # Get KDE density (normalized)
            price_idx = np.argmin(np.abs(price_range - zone_price))
            kde_value = kde_density[price_idx] / max_kde if max_kde > 0 else 0
            
            # Calculate composite score (before MTF confluence)
            # Emphasize Bayesian probability: bayesian 60%, traditional 30%, kde 10%
            composite_score = (
                zone['strength'] * 0.3 +
                bayesian_prob * 0.6 +
                kde_value * 0.1
            )
            
            # Add to zone
            zone['bayesian_probability'] = bayesian_prob
            zone['kde_density'] = kde_value
            zone['composite_score'] = composite_score
        
        # Sort by composite score
        self.zones.sort(key=lambda x: x['composite_score'], reverse=True)
        
        print(f"   ✅ Scored {len(self.zones)} zones")
        avg_composite = np.mean([z['composite_score'] for z in self.zones])
        print(f"   Average composite score: {avg_composite:.3f}")
        print(f"   High probability zones (>0.7): {sum(1 for z in self.zones if z['composite_score'] > 0.7)}")
    
    def get_high_probability_zones(self, threshold=0.7):
        """Get only high-probability zones based on composite score."""
        return [z for z in self.zones if z.get('composite_score', 0) >= threshold]
    
    def detect_mtf_zones(self):
        """Detect zones across multiple timeframes for confluence analysis."""
        print(f"\n🔄 Detecting zones across multiple timeframes...")
        
        mtf_zones = {}
        original_timeframe = self.timeframe
        original_df = self.df
        
        for tf in self.mtf_timeframes:
            print(f"   Processing {self._get_tf_name(tf)}...")
            
            self.timeframe = tf
            self.fetch_data()
            
            if self.df is None:
                continue
            
            sideways_mask = self.identify_sideways_periods()
            zones = self.detect_zones(sideways_mask, reversal_threshold=0.002, min_touches=2)
            
            mtf_zones[tf] = zones
            print(f"      Found {len(zones)} zones on {self._get_tf_name(tf)}")
        
        self.timeframe = original_timeframe
        self.df = original_df
        self.mtf_zones = mtf_zones
        
        print(f"   ✅ MTF detection complete")
        return mtf_zones
    
    def _get_tf_name(self, timeframe):
        """Get human-readable timeframe name."""
        tf_names = {
            mt5.TIMEFRAME_M15: 'M15', mt5.TIMEFRAME_H1: 'H1', mt5.TIMEFRAME_H4: 'H4'
        }
        return tf_names.get(timeframe, f'TF{timeframe}')
    
    def calculate_confluence_score(self, zone_price, zone_type):
        """Calculate confluence score based on alignment across timeframes."""
        if not self.mtf_zones:
            return 0.0, []
        
        aligned_timeframes = []
        
        for tf, zones in self.mtf_zones.items():
            for zone in zones:
                if zone['type'] != zone_type:
                    continue
                
                price_diff = abs(zone_price - zone['price']) / zone_price
                if price_diff <= self.mtf_tolerance:
                    aligned_timeframes.append(tf)
                    break
        
        num_aligned = len(aligned_timeframes)
        
        if num_aligned == 1:
            confluence_score = 0.0
        elif num_aligned == 2:
            confluence_score = 0.5
        elif num_aligned == 3:
            confluence_score = 0.75
        else:
            confluence_score = 1.0
        
        return confluence_score, aligned_timeframes
    
    def add_mtf_confluence_to_zones(self):
        """Add MTF confluence scores to existing zones."""
        print(f"\n🎯 Adding MTF confluence scores to zones...")
        
        if not self.mtf_zones:
            print("   ⚠️  No MTF zones detected, skipping confluence scoring")
            return
        
        confluence_count = 0
        
        for zone in self.zones:
            confluence_score, aligned_tfs = self.calculate_confluence_score(
                zone['price'], zone['type']
            )
            
            zone['mtf_confluence'] = confluence_score
            zone['aligned_timeframes'] = aligned_tfs
            zone['num_timeframes'] = len(aligned_tfs)
            
            if confluence_score > 0:
                confluence_count += 1
            
            # Recalculate composite score with MTF confluence
            # Option 2 - Aggressive (Bayesian + Confluence Focus)
            traditional_strength = zone.get('strength', 0)
            bayesian_prob = zone.get('bayesian_probability', 0)
            kde_density = zone.get('kde_density', 0)
            
            zone['composite_score'] = (
                traditional_strength * 0.15 +
                bayesian_prob * 0.4 +
                kde_density * 0.05 +
                confluence_score * 0.4
            )
        
        self.zones.sort(key=lambda x: x['composite_score'], reverse=True)
        
        print(f"   ✅ Added MTF confluence to {len(self.zones)} zones")
        print(f"   Zones with MTF confluence: {confluence_count}")
        
        high_confluence = [z for z in self.zones if z['mtf_confluence'] >= 0.5]
        if high_confluence:
            print(f"\n   🌟 Top MTF Confluence Zones:")
            for i, zone in enumerate(high_confluence[:5], 1):
                tf_names = [self._get_tf_name(tf) for tf in zone['aligned_timeframes']]
                print(f"      {i}. {zone['type'].upper()} @ {zone['price']:.5f}")
                print(f"         Composite: {zone['composite_score']:.3f}, MTF: {zone['mtf_confluence']:.2f}")
                print(f"         Timeframes: {', '.join(tf_names)}")

    
    def visualize_zones(self):
        """
        Visualize zones with clean rectangular boxes only.
        Opens in default web browser.
        """
        import plotly.graph_objects as go
        
        print(f"\n📈 Generating interactive Plotly chart for {len(self.df)} bars...")

        # Create single chart (no subplots)
        fig = go.Figure()

        # 1. Add Candlestick Chart with explicit colors
        fig.add_trace(go.Candlestick(
            x=self.df.index,
            open=self.df['open'], high=self.df['high'],
            low=self.df['low'], close=self.df['close'],
            name='Price',
            increasing_line_color='#26a69a',  # Green for bullish
            decreasing_line_color='#ef5350',  # Red for bearish
            showlegend=True
        ))

        # 2. Add Zone Rectangles Only (no extension lines or markers)
        # Filter zones to only show those within the current data timeframe
        first_time = self.df.index[0]
        last_time = self.df.index[-1]
        
        sorted_zones = sorted(self.zones, key=lambda x: x.get('composite_score', 0), reverse=True)
        visible_zones = [z for z in sorted_zones if first_time <= z['time'] <= last_time]
        
        print(f"   Displaying {len(visible_zones)} zones within current timeframe (out of {len(sorted_zones)} total)")

        for zone in visible_zones:
            zone_time = zone['time']
            zone_price = zone['price']
            score = zone.get('composite_score', zone['strength'])
            z_type = zone['type']
            
            # Color logic
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

            # Draw only the Zone Rectangle (no extension lines)
            fig.add_shape(type="rect",
                x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color=line_color, width=1),
                fillcolor=color
            )

        # 3. Layout Styling
        fig.update_layout(
            template='plotly_dark',
            height=800,
            title_text=f"{self.symbol} - Supply/Demand Zones ({len(self.df)} bars)",
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            xaxis_title="Time",
            yaxis_title="Price",
            dragmode='pan'  # Prevent accidental legend clicks
        )

        fig.show()
        print("✅ Interactive chart opened in browser")
    
    def export_zones(self, filename='supply_demand_zones_bayesian.csv'):
        """Export zones to CSV for ML training with Bayesian probabilities."""
        if not self.zones:
            print("⚠️  No zones to export")
            return
        
        zones_df = pd.DataFrame(self.zones)
        zones_df.to_csv(filename, index=False)
        
        print(f"\n💾 Zones exported to {filename}")
        print(f"   Total zones: {len(zones_df)}")
        print(f"   Avg traditional strength: {zones_df['strength'].mean()*100:.2f}%")
        if 'composite_score' in zones_df.columns:
            print(f"   Avg composite score: {zones_df['composite_score'].mean():.3f}")
            print(f"   Avg Bayesian probability: {zones_df['bayesian_probability'].mean():.3f}")
            print(f"   High probability zones (>0.7): {(zones_df['composite_score'] > 0.7).sum()}")
        print(f"   Avg touches: {zones_df['touches'].mean():.1f}")
    
    def create_zone_labels(self):
        """
        Create ground truth labels for ML training with Bayesian probabilities.
        
        For each candle, label:
        - is_near_supply: 1 if within X% of supply zone, 0 otherwise
        - is_near_demand: 1 if within X% of demand zone, 0 otherwise
        - distance_to_nearest_supply: distance in %
        - distance_to_nearest_demand: distance in %
        - bayesian_probability: Bayesian probability at current price
        - composite_zone_score: Highest composite score of nearby zones
        - high_prob_zone: 1 if near high-probability zone (composite > 0.7)
        """
        print(f"\n🏷️  Creating ground truth labels with Bayesian probabilities...")
        
        labels = pd.DataFrame(index=self.df.index)
        labels['is_near_supply'] = 0
        labels['is_near_demand'] = 0
        labels['distance_to_nearest_supply'] = np.inf
        labels['distance_to_nearest_demand'] = np.inf
        labels['bayesian_probability'] = 0.0
        labels['composite_zone_score'] = 0.0
        labels['high_prob_zone'] = 0
        
        proximity_threshold = 0.002  # 0.2% proximity
        
        for i, row in self.df.iterrows():
            current_price = row['close']
            
            # Get Bayesian probability at current price
            if self.bayesian_detector is not None:
                bin_idx = self.bayesian_detector._get_bin_index(current_price)
                if bin_idx is not None:
                    labels.loc[i, 'bayesian_probability'] = self.bayesian_detector.p_zone[bin_idx]
            
            # Find nearest supply zone
            supply_zones = [z for z in self.zones if z['type'] == 'supply']
            if supply_zones:
                distances = [abs(current_price - z['price']) / current_price for z in supply_zones]
                min_dist = min(distances)
                labels.loc[i, 'distance_to_nearest_supply'] = min_dist
                
                if min_dist <= proximity_threshold:
                    labels.loc[i, 'is_near_supply'] = 1
                    # Get composite score of nearest zone
                    nearest_zone = supply_zones[distances.index(min_dist)]
                    labels.loc[i, 'composite_zone_score'] = nearest_zone.get('composite_score', nearest_zone['strength'])
            
            # Find nearest demand zone
            demand_zones = [z for z in self.zones if z['type'] == 'demand']
            if demand_zones:
                distances = [abs(current_price - z['price']) / current_price for z in demand_zones]
                min_dist = min(distances)
                labels.loc[i, 'distance_to_nearest_demand'] = min_dist
                
                if min_dist <= proximity_threshold:
                    labels.loc[i, 'is_near_demand'] = 1
                    # Get composite score of nearest zone
                    nearest_zone = demand_zones[distances.index(min_dist)]
                    current_score = labels.loc[i, 'composite_zone_score']
                    zone_score = nearest_zone.get('composite_score', nearest_zone['strength'])
                    labels.loc[i, 'composite_zone_score'] = max(current_score, zone_score)
            
            # Mark high probability zones
            if labels.loc[i, 'composite_zone_score'] > 0.7:
                labels.loc[i, 'high_prob_zone'] = 1
        
        # Export labels
        combined = pd.concat([self.df, labels], axis=1)
        combined.to_csv('zone_labels_bayesian_for_ml.csv')
        
        print(f"✅ Labels created with Bayesian probabilities")
        print(f"   Near supply: {labels['is_near_supply'].sum()} bars")
        print(f"   Near demand: {labels['is_near_demand'].sum()} bars")
        print(f"   High prob zones: {labels['high_prob_zone'].sum()} bars")
        print(f"   Avg Bayesian probability: {labels['bayesian_probability'].mean():.3f}")
        print(f"   Saved to zone_labels_bayesian_for_ml.csv")
        
        return labels

def main():
    print("="*60)
    print("  SUPPLY/DEMAND ZONE DETECTION")
    print("  Symbol: EURUSDm | Timeframe: M15")
    print("="*60)
    
    detector = SupplyDemandZoneDetector('EURUSDm', mt5.TIMEFRAME_M15, bars=3000)
    
    if not detector.init_mt5():
        return
    
    # Fetch data
    detector.fetch_data()
    
    # Identify sideways periods
    sideways_mask = detector.identify_sideways_periods()
    
    # Detect zones
    zones = detector.detect_zones(
        sideways_mask, 
        reversal_threshold=0.002,  # 0.2% minimum reversal
        min_touches=2
    )
    
    if zones:
        # Detect zones across multiple timeframes for confluence
        detector.detect_mtf_zones()
        
        # Add MTF confluence scores to zones
        detector.add_mtf_confluence_to_zones()
        
        # Cluster zones to find key levels
        detector.cluster_zones(zones, n_clusters=10)
        
        # Visualize
        detector.visualize_zones()
        
        # Export zones
        detector.export_zones()
        
        # Create ML labels
        detector.create_zone_labels()
        
        # Summary
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        print(f"\n✅ Zone Detection Complete")
        print(f"   Total zones found: {len(zones)}")
        print(f"   Supply zones: {sum(1 for z in zones if z['type'] == 'supply')}")
        print(f"   Demand zones: {sum(1 for z in zones if z['type'] == 'demand')}")
        print(f"\n📊 Next Steps:")
        print(f"   1. Review 'supply_demand_zones.png' to validate zones")
        print(f"   2. Use 'zone_labels_for_ml.csv' to train prediction model")
        print(f"   3. Train NN to predict: is_near_supply, is_near_demand")
    else:
        print(f"\n⚠️  No zones detected. Try adjusting parameters:")
        print(f"   - Lower reversal_threshold")
        print(f"   - Reduce min_touches")
        print(f"   - Increase bars")
    
    mt5.shutdown()
    print(f"\n👋 Complete")

if __name__ == "__main__":
    main()
