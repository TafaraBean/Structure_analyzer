"""
Bayesian Supply/Demand Zone Detection
======================================
This script implements a probabilistic approach to identifying supply and demand zones
using Bayesian statistics. Instead of binary zone detection, it generates probability
heatmaps that update as new market data arrives.

Two methods are implemented:
1. Discrete Binning Method: Grid-based Bayesian updates
2. Gaussian KDE Method: Continuous probability density estimation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class BayesianZoneDetector:
    """
    Implements Bayesian probability-based supply/demand zone detection.
    
    Uses Bayes' Theorem to update zone probabilities as new price action occurs:
    P(Zone|Evidence) = P(Evidence|Zone) * P(Zone) / P(Evidence)
    """
    
    def __init__(self, 
                 price_min: float,
                 price_max: float,
                 n_bins: int = 200,
                 rejection_likelihood: float = 0.85,
                 breakthrough_likelihood: float = 0.15,
                 decay_rate: float = 0.98):
        """
        Initialize the Bayesian Zone Detector.
        
        Parameters:
        -----------
        price_min : float
            Minimum price for the grid
        price_max : float
            Maximum price for the grid
        n_bins : int
            Number of discrete price bins
        rejection_likelihood : float
            P(Rejection|Zone exists) - probability of seeing rejection if zone is real
        breakthrough_likelihood : float
            P(Breakthrough|Zone exists) - probability of breakthrough if zone is real
        decay_rate : float
            Time decay factor for old zones (0.95-0.99 typical)
        """
        self.price_min = price_min
        self.price_max = price_max
        self.n_bins = n_bins
        self.rejection_likelihood = rejection_likelihood
        self.breakthrough_likelihood = breakthrough_likelihood
        self.decay_rate = decay_rate
        
        # Create price bins
        self.bins = np.linspace(price_min, price_max, n_bins)
        self.bin_width = (price_max - price_min) / n_bins
        
        # Initialize with flat prior (uniform distribution)
        self.p_zone = np.ones(n_bins) * 0.1  # 10% prior probability
        
        # Track history for visualization
        self.history = []
        
    def set_volume_profile_prior(self, prices: np.ndarray, volumes: np.ndarray):
        """
        Set prior probabilities based on Volume Profile.
        Low volume nodes (LVN) get higher prior probabilities as potential zones.
        
        Parameters:
        -----------
        prices : np.ndarray
            Historical prices
        volumes : np.ndarray
            Corresponding volumes
        """
        # Bin the volumes by price
        volume_profile = np.zeros(self.n_bins)
        
        for price, volume in zip(prices, volumes):
            bin_idx = self._get_bin_index(price)
            if bin_idx is not None:
                volume_profile[bin_idx] += volume
        
        # Normalize volume profile
        if volume_profile.sum() > 0:
            volume_profile = volume_profile / volume_profile.max()
        
        # Invert: Low volume = High zone probability
        # Apply sigmoid transformation for smooth transition
        self.p_zone = 1 / (1 + np.exp(5 * (volume_profile - 0.3)))
        
        # Ensure probabilities are in reasonable range
        self.p_zone = np.clip(self.p_zone, 0.05, 0.95)
        
    def _get_bin_index(self, price: float) -> Optional[int]:
        """Get the bin index for a given price."""
        if price < self.price_min or price > self.price_max:
            return None
        idx = int((price - self.price_min) / self.bin_width)
        return min(idx, self.n_bins - 1)
    
    def update_rejection(self, price: float, strength: float = 1.0):
        """
        Update probabilities when a rejection occurs at a price level.
        
        Parameters:
        -----------
        price : float
            Price where rejection occurred
        strength : float
            Strength of rejection (0-1), based on wick size, volume, etc.
        """
        bin_idx = self._get_bin_index(price)
        if bin_idx is None:
            return
        
        # Bayes update: P(Zone|Rejection) ∝ P(Rejection|Zone) * P(Zone)
        likelihood = self.rejection_likelihood * strength
        
        # Update the bin and neighboring bins (zone has width)
        spread = max(1, int(3 / self.bin_width))  # ~3 price units spread
        
        for i in range(max(0, bin_idx - spread), min(self.n_bins, bin_idx + spread + 1)):
            distance_weight = np.exp(-((i - bin_idx) ** 2) / (2 * (spread / 2) ** 2))
            
            # Bayesian update
            prior = self.p_zone[i]
            posterior = (likelihood * distance_weight * prior) / \
                       (likelihood * distance_weight * prior + (1 - likelihood) * (1 - prior))
            
            self.p_zone[i] = posterior
        
        # Clip to valid probability range
        self.p_zone = np.clip(self.p_zone, 0.01, 0.99)
        
    def update_breakthrough(self, price: float, momentum: float = 1.0):
        """
        Update probabilities when price breaks through a level.
        
        Parameters:
        -----------
        price : float
            Price where breakthrough occurred
        momentum : float
            Strength of breakthrough (0-1), based on candle size, volume
        """
        bin_idx = self._get_bin_index(price)
        if bin_idx is None:
            return
        
        # Bayes update: P(Zone|Breakthrough) - reduces probability
        likelihood = self.breakthrough_likelihood * momentum
        
        spread = max(1, int(3 / self.bin_width))
        
        for i in range(max(0, bin_idx - spread), min(self.n_bins, bin_idx + spread + 1)):
            distance_weight = np.exp(-((i - bin_idx) ** 2) / (2 * (spread / 2) ** 2))
            
            prior = self.p_zone[i]
            # Inverse update - breakthrough reduces zone probability
            posterior = ((1 - likelihood * distance_weight) * prior) / \
                       ((1 - likelihood * distance_weight) * prior + likelihood * distance_weight * (1 - prior))
            
            self.p_zone[i] = posterior
        
        self.p_zone = np.clip(self.p_zone, 0.01, 0.99)
        
    def apply_time_decay(self):
        """Apply time decay to all probabilities (zones become less relevant over time)."""
        # Decay towards the mean (0.1)
        self.p_zone = self.p_zone * self.decay_rate + 0.1 * (1 - self.decay_rate)
        
    def get_zones(self, threshold: float = 0.7) -> List[Tuple[float, float, float]]:
        """
        Extract discrete zones from the probability distribution.
        
        Parameters:
        -----------
        threshold : float
            Minimum probability to consider as a zone
            
        Returns:
        --------
        List of (price_level, probability, width) tuples
        """
        zones = []
        in_zone = False
        zone_start = None
        zone_probs = []
        
        for i, prob in enumerate(self.p_zone):
            price = self.bins[i]
            
            if prob >= threshold:
                if not in_zone:
                    in_zone = True
                    zone_start = price
                zone_probs.append(prob)
            else:
                if in_zone:
                    # Zone ended
                    zone_center = zone_start + len(zone_probs) * self.bin_width / 2
                    zone_prob = np.mean(zone_probs)
                    zone_width = len(zone_probs) * self.bin_width
                    zones.append((zone_center, zone_prob, zone_width))
                    
                    in_zone = False
                    zone_probs = []
        
        return zones
    
    def plot_probability_heatmap(self, current_price: Optional[float] = None, 
                                 title: str = "Bayesian Supply/Demand Probability Heatmap"):
        """
        Visualize the probability distribution as a heatmap.
        
        Parameters:
        -----------
        current_price : float, optional
            Current market price to mark on the chart
        title : str
            Chart title
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Create heatmap
        heatmap_data = self.p_zone.reshape(-1, 1)
        im = ax1.imshow(heatmap_data.T, aspect='auto', cmap='RdYlGn', 
                       extent=[self.price_min, self.price_max, 0, 1],
                       origin='lower', vmin=0, vmax=1)
        
        ax1.set_xlabel('Price', fontsize=12)
        ax1.set_ylabel('Probability Density', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        
        # Mark current price
        if current_price is not None:
            ax1.axvline(current_price, color='blue', linestyle='--', 
                       linewidth=2, label=f'Current Price: {current_price:.2f}')
            ax1.legend()
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Zone Probability', fontsize=11)
        
        # Plot probability curve
        ax2.fill_between(self.bins, 0, self.p_zone, alpha=0.6, color='green')
        ax2.plot(self.bins, self.p_zone, color='darkgreen', linewidth=2)
        ax2.axhline(0.7, color='red', linestyle='--', alpha=0.5, label='High Probability Threshold (0.7)')
        ax2.set_xlabel('Price', fontsize=12)
        ax2.set_ylabel('P(Zone)', fontsize=12)
        ax2.set_title('Probability Distribution', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        if current_price is not None:
            ax2.axvline(current_price, color='blue', linestyle='--', linewidth=2)
        
        plt.tight_layout()
        return fig


class GaussianKDEZoneDetector:
    """
    Implements continuous probability density estimation using Gaussian KDE.
    Each historical reversal point is treated as a sample from a zone distribution.
    """
    
    def __init__(self, bandwidth: float = 1.0):
        """
        Initialize the KDE Zone Detector.
        
        Parameters:
        -----------
        bandwidth : float
            Bandwidth for Gaussian kernel (controls smoothness)
        """
        self.bandwidth = bandwidth
        self.reversal_points = []
        self.reversal_weights = []
        
    def add_reversal(self, price: float, strength: float = 1.0):
        """
        Add a reversal point to the distribution.
        
        Parameters:
        -----------
        price : float
            Price where reversal occurred
        strength : float
            Strength of reversal (affects kernel height)
        """
        self.reversal_points.append(price)
        self.reversal_weights.append(strength)
        
    def get_probability_density(self, price_range: np.ndarray) -> np.ndarray:
        """
        Calculate probability density across a price range.
        
        Parameters:
        -----------
        price_range : np.ndarray
            Array of prices to evaluate
            
        Returns:
        --------
        np.ndarray : Probability density at each price
        """
        if len(self.reversal_points) < 2:
            return np.zeros_like(price_range)
        
        # Create weighted KDE
        points = np.array(self.reversal_points)
        weights = np.array(self.reversal_weights)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Manual weighted KDE using Gaussian kernels
        density = np.zeros_like(price_range, dtype=float)
        
        for point, weight in zip(points, weights):
            # Gaussian kernel: exp(-0.5 * ((x - point) / bandwidth)^2)
            kernel = np.exp(-0.5 * ((price_range - point) / self.bandwidth) ** 2)
            kernel = kernel / (self.bandwidth * np.sqrt(2 * np.pi))  # Normalize
            density += weight * kernel
        
        return density
    
    def get_zones(self, price_range: np.ndarray, threshold_percentile: float = 70) -> List[Tuple[float, float]]:
        """
        Extract zones from KDE peaks.
        
        Parameters:
        -----------
        price_range : np.ndarray
            Price range to analyze
        threshold_percentile : float
            Percentile threshold for zone detection
            
        Returns:
        --------
        List of (price_center, density_value) tuples
        """
        density = self.get_probability_density(price_range)
        threshold = np.percentile(density, threshold_percentile)
        
        # Find peaks above threshold
        zones = []
        in_zone = False
        zone_start_idx = None
        
        for i, d in enumerate(density):
            if d >= threshold:
                if not in_zone:
                    in_zone = True
                    zone_start_idx = i
            else:
                if in_zone:
                    # Find peak in this zone
                    zone_end_idx = i
                    zone_density = density[zone_start_idx:zone_end_idx]
                    peak_idx = zone_start_idx + np.argmax(zone_density)
                    zones.append((price_range[peak_idx], density[peak_idx]))
                    
                    in_zone = False
        
        return zones
    
    def plot_kde_heatmap(self, price_range: np.ndarray, current_price: Optional[float] = None,
                        title: str = "Gaussian KDE Supply/Demand Zones"):
        """
        Visualize the KDE probability density.
        
        Parameters:
        -----------
        price_range : np.ndarray
            Price range to plot
        current_price : float, optional
            Current market price
        title : str
            Chart title
        """
        density = self.get_probability_density(price_range)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Heatmap
        heatmap_data = density.reshape(-1, 1)
        im = ax1.imshow(heatmap_data.T, aspect='auto', cmap='RdYlGn',
                       extent=[price_range.min(), price_range.max(), 0, 1],
                       origin='lower')
        
        ax1.set_xlabel('Price', fontsize=12)
        ax1.set_ylabel('Probability Density', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        
        if current_price is not None:
            ax1.axvline(current_price, color='blue', linestyle='--', 
                       linewidth=2, label=f'Current Price: {current_price:.2f}')
            ax1.legend()
        
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Probability Density', fontsize=11)
        
        # Density curve
        ax2.fill_between(price_range, 0, density, alpha=0.6, color='green')
        ax2.plot(price_range, density, color='darkgreen', linewidth=2)
        
        # Mark reversal points
        ax2.scatter(self.reversal_points, 
                   [0] * len(self.reversal_points),
                   c='red', s=50, alpha=0.6, marker='^', 
                   label='Reversal Points', zorder=5)
        
        ax2.set_xlabel('Price', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Probability Density Function', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        if current_price is not None:
            ax2.axvline(current_price, color='blue', linestyle='--', linewidth=2)
        
        plt.tight_layout()
        return fig


def detect_swing_points(df: pd.DataFrame, window: int = 5) -> Tuple[List[float], List[float]]:
    """
    Detect swing highs and lows in price data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLC data
    window : int
        Lookback window for swing detection
        
    Returns:
    --------
    Tuple of (swing_highs, swing_lows) as lists of prices
    """
    swing_highs = []
    swing_lows = []
    
    for i in range(window, len(df) - window):
        # Swing high: highest high in window
        if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
            swing_highs.append(df['high'].iloc[i])
            
        # Swing low: lowest low in window
        if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
            swing_lows.append(df['low'].iloc[i])
    
    return swing_highs, swing_lows


def calculate_rejection_strength(row: pd.Series) -> float:
    """
    Calculate rejection strength based on wick size relative to body.
    
    Parameters:
    -----------
    row : pd.Series
        OHLC candle data
        
    Returns:
    --------
    float : Rejection strength (0-1)
    """
    body_size = abs(row['close'] - row['open'])
    upper_wick = row['high'] - max(row['open'], row['close'])
    lower_wick = min(row['open'], row['close']) - row['low']
    
    total_range = row['high'] - row['low']
    
    if total_range == 0:
        return 0
    
    # Rejection is strong if wick is large relative to body
    max_wick = max(upper_wick, lower_wick)
    rejection_ratio = max_wick / total_range
    
    return min(rejection_ratio, 1.0)


def simulate_live_trading(df: pd.DataFrame, detector: BayesianZoneDetector, 
                         lookback: int = 100) -> BayesianZoneDetector:
    """
    Simulate live trading by processing candles sequentially and updating probabilities.
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLC data
    detector : BayesianZoneDetector
        The detector instance to update
    lookback : int
        Number of candles to process
        
    Returns:
    --------
    Updated BayesianZoneDetector
    """
    print(f"Simulating live trading on {lookback} candles...")
    
    for i in range(len(df) - lookback, len(df)):
        row = df.iloc[i]
        
        # Detect rejection (large wick)
        rejection_strength = calculate_rejection_strength(row)
        
        if rejection_strength > 0.5:
            # Strong rejection at high or low
            if row['high'] - max(row['open'], row['close']) > \
               min(row['open'], row['close']) - row['low']:
                # Upper wick rejection (supply zone)
                detector.update_rejection(row['high'], rejection_strength)
            else:
                # Lower wick rejection (demand zone)
                detector.update_rejection(row['low'], rejection_strength)
        
        # Detect breakthrough (strong momentum candle)
        body_size = abs(row['close'] - row['open'])
        total_range = row['high'] - row['low']
        
        if total_range > 0 and body_size / total_range > 0.7:
            # Strong momentum candle
            momentum = body_size / total_range
            mid_price = (row['high'] + row['low']) / 2
            detector.update_breakthrough(mid_price, momentum)
        
        # Apply time decay every 10 candles
        if i % 10 == 0:
            detector.apply_time_decay()
    
    print("Simulation complete!")
    return detector


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Bayesian Supply/Demand Zone Detection")
    print("=" * 70)
    
    # Generate sample data (replace with real data)
    print("\n[1/5] Generating sample OHLC data...")
    np.random.seed(42)
    n_candles = 500
    
    # Simulate price movement with trend and noise
    base_price = 2000
    trend = np.linspace(0, 50, n_candles)
    noise = np.cumsum(np.random.randn(n_candles) * 2)
    close_prices = base_price + trend + noise
    
    # Generate OHLC
    df = pd.DataFrame({
        'open': close_prices + np.random.randn(n_candles) * 0.5,
        'high': close_prices + np.abs(np.random.randn(n_candles) * 2),
        'low': close_prices - np.abs(np.random.randn(n_candles) * 2),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n_candles)
    })
    
    print(f"   Generated {len(df)} candles")
    print(f"   Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    
    # ========================================================================
    # METHOD 1: Discrete Binning with Bayesian Updates
    # ========================================================================
    print("\n[2/5] Initializing Bayesian Zone Detector (Discrete Binning)...")
    
    price_min = df['low'].min() - 5
    price_max = df['high'].max() + 5
    
    bayesian_detector = BayesianZoneDetector(
        price_min=price_min,
        price_max=price_max,
        n_bins=200,
        rejection_likelihood=0.85,
        breakthrough_likelihood=0.15,
        decay_rate=0.98
    )
    
    # Set volume profile prior
    print("   Setting Volume Profile prior...")
    bayesian_detector.set_volume_profile_prior(
        df['close'].values, 
        df['volume'].values
    )
    
    # Simulate live trading
    print("\n[3/5] Simulating live trading updates...")
    bayesian_detector = simulate_live_trading(df, bayesian_detector, lookback=200)
    
    # Extract high-probability zones
    zones = bayesian_detector.get_zones(threshold=0.7)
    print(f"\n   Detected {len(zones)} high-probability zones (P > 0.7):")
    for i, (price, prob, width) in enumerate(zones[:10], 1):
        print(f"   Zone {i}: Price={price:.2f}, P(Zone)={prob:.3f}, Width={width:.2f}")
    
    # ========================================================================
    # METHOD 2: Gaussian KDE
    # ========================================================================
    print("\n[4/5] Initializing Gaussian KDE Zone Detector...")
    
    kde_detector = GaussianKDEZoneDetector(bandwidth=2.0)
    
    # Detect swing points
    swing_highs, swing_lows = detect_swing_points(df, window=5)
    
    print(f"   Found {len(swing_highs)} swing highs and {len(swing_lows)} swing lows")
    
    # Add reversals to KDE
    for high in swing_highs:
        kde_detector.add_reversal(high, strength=1.0)
    
    for low in swing_lows:
        kde_detector.add_reversal(low, strength=1.0)
    
    # Get zones from KDE
    price_range = np.linspace(price_min, price_max, 500)
    kde_zones = kde_detector.get_zones(price_range, threshold_percentile=75)
    
    print(f"   Detected {len(kde_zones)} KDE zones (top 25% density):")
    for i, (price, density) in enumerate(kde_zones[:10], 1):
        print(f"   Zone {i}: Price={price:.2f}, Density={density:.6f}")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    print("\n[5/5] Generating visualizations...")
    
    current_price = df['close'].iloc[-1]
    
    # Plot Bayesian method
    fig1 = bayesian_detector.plot_probability_heatmap(
        current_price=current_price,
        title="Bayesian S/D Zones - Discrete Binning Method"
    )
    plt.savefig('bayesian_zones_discrete.png', dpi=150, bbox_inches='tight')
    print("   Saved: bayesian_zones_discrete.png")
    
    # Plot KDE method
    fig2 = kde_detector.plot_kde_heatmap(
        price_range=price_range,
        current_price=current_price,
        title="Bayesian S/D Zones - Gaussian KDE Method"
    )
    plt.savefig('bayesian_zones_kde.png', dpi=150, bbox_inches='tight')
    print("   Saved: bayesian_zones_kde.png")
    
    # Combined comparison plot
    fig3, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Bayesian discrete
    axes[0].fill_between(bayesian_detector.bins, 0, bayesian_detector.p_zone, 
                         alpha=0.6, color='blue', label='Bayesian Discrete')
    axes[0].axhline(0.7, color='red', linestyle='--', alpha=0.5, label='Threshold (0.7)')
    axes[0].axvline(current_price, color='green', linestyle='--', linewidth=2, label='Current Price')
    axes[0].set_ylabel('P(Zone)', fontsize=12)
    axes[0].set_title('Bayesian Discrete Binning Method', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # KDE
    kde_density = kde_detector.get_probability_density(price_range)
    axes[1].fill_between(price_range, 0, kde_density, 
                         alpha=0.6, color='green', label='Gaussian KDE')
    axes[1].scatter(kde_detector.reversal_points, [0] * len(kde_detector.reversal_points),
                   c='red', s=30, alpha=0.6, marker='^', label='Reversal Points')
    axes[1].axvline(current_price, color='blue', linestyle='--', linewidth=2, label='Current Price')
    axes[1].set_xlabel('Price', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Gaussian KDE Method', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bayesian_zones_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved: bayesian_zones_comparison.png")
    
    print("\n" + "=" * 70)
    print("COMPLETE! Check the generated PNG files for visualizations.")
    print("=" * 70)
    
    # Show plots
    plt.show()
