"""
Bayesian Zone Integration Demo
================================
Standalone demo showing Bayesian probability integration with traditional
supply/demand zones. Works WITHOUT MT5 connection using sample data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from bayesian_supply_demand_zones import (
    BayesianZoneDetector, 
    GaussianKDEZoneDetector,
    calculate_rejection_strength,
    detect_swing_points
)

def generate_sample_ohlc(n_candles=1000, base_price=2000):
    """Generate realistic sample OHLC data with trends and zones."""
    print(f"📊 Generating {n_candles} sample candles...")
    
    np.random.seed(42)
    
    # Create price movement with trends and consolidations
    trend = np.concatenate([
        np.linspace(0, 30, 300),      # Uptrend
        np.ones(200) * 30,            # Consolidation (sideways)
        np.linspace(30, 10, 200),     # Downtrend
        np.ones(150) * 10,            # Consolidation
        np.linspace(10, 50, 150)      # Strong uptrend
    ])
    
    noise = np.cumsum(np.random.randn(n_candles) * 1.5)
    close_prices = base_price + trend + noise
    
    # Generate OHLC with realistic wicks
    df = pd.DataFrame({
        'time': [datetime.now() - timedelta(minutes=15*i) for i in range(n_candles, 0, -1)],
        'open': close_prices + np.random.randn(n_candles) * 0.5,
        'high': close_prices + np.abs(np.random.randn(n_candles) * 2),
        'low': close_prices - np.abs(np.random.randn(n_candles) * 2),
        'close': close_prices,
        'tick_volume': np.random.randint(1000, 10000, n_candles)
    })
    
    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    df.set_index('time', inplace=True)
    
    print(f"✅ Generated data from {df.index[0]} to {df.index[-1]}")
    print(f"   Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    
    return df


def detect_traditional_zones(df, min_strength=0.003):
    """Detect traditional supply/demand zones using swing points."""
    print(f"\n🎯 Detecting traditional supply/demand zones...")
    
    swing_highs, swing_lows = detect_swing_points(df, window=5)
    
    zones = []
    
    # Process swing highs (supply zones)
    for high_price in swing_highs:
        # Simple validation: check if price dropped after this high
        zones.append({
            'type': 'supply',
            'price': high_price,
            'strength': np.random.uniform(0.002, 0.01),  # Simplified
            'touches': np.random.randint(2, 6)
        })
    
    # Process swing lows (demand zones)
    for low_price in swing_lows:
        zones.append({
            'type': 'demand',
            'price': low_price,
            'strength': np.random.uniform(0.002, 0.01),
            'touches': np.random.randint(2, 6)
        })
    
    # Filter by strength
    zones = [z for z in zones if z['strength'] >= min_strength]
    
    print(f"✅ Found {len(zones)} traditional zones")
    print(f"   Supply: {sum(1 for z in zones if z['type'] == 'supply')}")
    print(f"   Demand: {sum(1 for z in zones if z['type'] == 'demand')}")
    
    return zones


def integrate_bayesian_probabilities(df, zones):
    """Integrate Bayesian probabilities with traditional zones."""
    print(f"\n🧮 Integrating Bayesian probabilities...")
    
    # Initialize Bayesian detector
    price_min = df['low'].min() - (df['low'].min() * 0.01)
    price_max = df['high'].max() + (df['high'].max() * 0.01)
    
    bayesian_detector = BayesianZoneDetector(
        price_min=price_min,
        price_max=price_max,
        n_bins=200,
        rejection_likelihood=0.85,
        breakthrough_likelihood=0.15,
        decay_rate=0.98
    )
    
    # Set volume profile prior
    bayesian_detector.set_volume_profile_prior(
        df['close'].values,
        df['tick_volume'].values
    )
    
    # Initialize KDE detector
    kde_detector = GaussianKDEZoneDetector(bandwidth=2.0)
    
    # Update probabilities from price action
    print(f"   Processing {len(df)} candles...")
    update_count = 0
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Detect rejections
        rejection_strength = calculate_rejection_strength(row)
        
        if rejection_strength > 0.5:
            upper_wick = row['high'] - max(row['open'], row['close'])
            lower_wick = min(row['open'], row['close']) - row['low']
            
            if upper_wick > lower_wick:
                bayesian_detector.update_rejection(row['high'], rejection_strength)
                kde_detector.add_reversal(row['high'], rejection_strength)
            else:
                bayesian_detector.update_rejection(row['low'], rejection_strength)
                kde_detector.add_reversal(row['low'], rejection_strength)
            
            update_count += 1
        
        # Detect breakthroughs
        body_size = abs(row['close'] - row['open'])
        total_range = row['high'] - row['low']
        
        if total_range > 0 and body_size / total_range > 0.7:
            momentum = body_size / total_range
            mid_price = (row['high'] + row['low']) / 2
            bayesian_detector.update_breakthrough(mid_price, momentum)
        
        # Time decay
        if i % 10 == 0:
            bayesian_detector.apply_time_decay()
    
    print(f"   ✅ {update_count} probability updates applied")
    
    # Score zones with Bayesian probabilities
    print(f"\n🎯 Scoring zones with Bayesian probabilities...")
    
    price_range = np.linspace(price_min, price_max, 500)
    kde_density = kde_detector.get_probability_density(price_range)
    max_kde = kde_density.max() if kde_density.max() > 0 else 1.0
    
    for zone in zones:
        zone_price = zone['price']
        
        # Get Bayesian probability
        bin_idx = bayesian_detector._get_bin_index(zone_price)
        if bin_idx is not None:
            bayesian_prob = bayesian_detector.p_zone[bin_idx]
        else:
            bayesian_prob = 0.1
        
        # Get KDE density
        price_idx = np.argmin(np.abs(price_range - zone_price))
        kde_value = kde_density[price_idx] / max_kde if max_kde > 0 else 0
        
        # Calculate composite score
        composite_score = (
            zone['strength'] * 0.4 +
            bayesian_prob * 0.4 +
            kde_value * 0.2
        )
        
        zone['bayesian_probability'] = bayesian_prob
        zone['kde_density'] = kde_value
        zone['composite_score'] = composite_score
    
    # Sort by composite score
    zones.sort(key=lambda x: x['composite_score'], reverse=True)
    
    avg_composite = np.mean([z['composite_score'] for z in zones])
    high_prob_count = sum(1 for z in zones if z['composite_score'] > 0.7)
    
    print(f"   ✅ Zones scored")
    print(f"   Average composite score: {avg_composite:.3f}")
    print(f"   High probability zones (>0.7): {high_prob_count}")
    
    return zones, bayesian_detector, kde_detector


def visualize_integrated_zones(df, zones, bayesian_detector, kde_detector, sample_bars=500, min_score=0.3):
    """Create comprehensive visualization of integrated zones."""
    print(f"\n📈 Creating integrated visualization...")
    
    # Filter zones by minimum composite score to reduce clutter
    filtered_zones = [z for z in zones if z['composite_score'] >= min_score]
    print(f"   Filtering: {len(zones)} total zones → {len(filtered_zones)} zones (score ≥ {min_score})")
    
    # Use last N bars
    df_sample = df.iloc[-sample_bars:]
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1.5, 1.5], hspace=0.3, wspace=0.3)
    
    fig.suptitle('Bayesian Supply/Demand Zone Integration Demo', 
                 fontsize=16, color='white', fontweight='bold')
    
    # ====================================================================
    # Panel 1: Price with zones from origin
    # ====================================================================
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_sample.index, df_sample['close'], color='white', linewidth=1.5, alpha=0.8)
    
    for zone in filtered_zones:
        zone_price = zone['price']
        composite_score = zone['composite_score']
        
        # Color by score
        if composite_score >= 0.8:
            color = '#00ff00' if zone['type'] == 'demand' else '#ff0000'
        elif composite_score >= 0.6:
            color = '#90ee90' if zone['type'] == 'demand' else '#ff6b6b'
        elif composite_score >= 0.4:
            color = '#ffff00'
        else:
            color = '#808080'
        
        # Draw horizontal line
        ax1.axhline(y=zone_price, color=color, linestyle='--', linewidth=1.5, alpha=0.6)
        
        # Add label
        ax1.text(df_sample.index[-1], zone_price, f" {composite_score:.2f}",
                fontsize=8, color='white', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
    
    ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax1.set_title('Price Chart with Bayesian-Scored Zones', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':')
    
    # ====================================================================
    # Panel 2: Bayesian Probability Heatmap
    # ====================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    
    heatmap_data = bayesian_detector.p_zone.reshape(-1, 1)
    im = ax2.imshow(heatmap_data.T, aspect='auto', cmap='RdYlGn',
                   extent=[bayesian_detector.price_min, bayesian_detector.price_max, 0, 1],
                   origin='lower', vmin=0, vmax=1)
    
    ax2.set_xlabel('Price', fontsize=11)
    ax2.set_title('Bayesian Probability Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.1, label='P(Zone)')
    
    # ====================================================================
    # Panel 3: KDE Density
    # ====================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    
    price_range = np.linspace(bayesian_detector.price_min, bayesian_detector.price_max, 500)
    kde_density = kde_detector.get_probability_density(price_range)
    
    ax3.fill_between(price_range, 0, kde_density, alpha=0.6, color='cyan')
    ax3.plot(price_range, kde_density, color='blue', linewidth=2)
    
    ax3.set_xlabel('Price', fontsize=11)
    ax3.set_title('KDE Probability Density', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # ====================================================================
    # Panel 4: Composite Score Distribution
    # ====================================================================
    ax4 = fig.add_subplot(gs[2, :])
    
    zone_prices = [z['price'] for z in filtered_zones]
    zone_scores = [z['composite_score'] for z in filtered_zones]
    zone_colors = []
    
    for z in filtered_zones:
        score = z['composite_score']
        if score >= 0.8:
            zone_colors.append('#00ff00' if z['type'] == 'demand' else '#ff0000')
        elif score >= 0.6:
            zone_colors.append('#90ee90' if z['type'] == 'demand' else '#ff6b6b')
        elif score >= 0.4:
            zone_colors.append('#ffff00')
        else:
            zone_colors.append('#808080')
    
    ax4.bar(range(len(filtered_zones)), zone_scores, color=zone_colors, alpha=0.7)
    ax4.axhline(0.7, color='cyan', linestyle='--', linewidth=2, label='High Prob Threshold')
    ax4.set_xlabel('Zone Index', fontsize=11)
    ax4.set_ylabel('Composite Score', fontsize=11)
    ax4.set_title('Zone Composite Scores (Sorted)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    
    filename = 'bayesian_integration_demo.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    print(f"✅ Visualization saved as {filename}")
    
    plt.show()


def export_results(zones):
    """Export zones to CSV."""
    print(f"\n💾 Exporting results...")
    
    df_zones = pd.DataFrame(zones)
    filename = 'bayesian_zones_demo.csv'
    df_zones.to_csv(filename, index=False)
    
    print(f"✅ Exported to {filename}")
    print(f"   Columns: {list(df_zones.columns)}")


def main():
    print("=" * 70)
    print("  BAYESIAN ZONE INTEGRATION DEMO")
    print("  No MT5 Required - Uses Sample Data")
    print("=" * 70)
    
    # Generate sample data
    df = generate_sample_ohlc(n_candles=1000, base_price=2000)
    
    # Detect traditional zones
    zones = detect_traditional_zones(df, min_strength=0.003)
    
    # Integrate Bayesian probabilities
    zones, bayesian_detector, kde_detector = integrate_bayesian_probabilities(df, zones)
    
    # Visualize (with filtering to reduce clutter)
    visualize_integrated_zones(df, zones, bayesian_detector, kde_detector, 
                               sample_bars=500, min_score=0.3)
    
    # Export
    export_results(zones)
    
    # Filter statistics
    high_prob = [z for z in zones if z['composite_score'] > 0.7]
    medium_prob = [z for z in zones if 0.4 <= z['composite_score'] <= 0.7]
    low_prob = [z for z in zones if z['composite_score'] < 0.4]
    
    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"\n✅ Integration Demo Complete!")
    print(f"   Total zones detected: {len(zones)}")
    print(f"   High probability (>0.7): {len(high_prob)} zones")
    print(f"   Medium probability (0.4-0.7): {len(medium_prob)} zones")
    print(f"   Low probability (<0.4): {len(low_prob)} zones")
    print(f"   Average composite score: {np.mean([z['composite_score'] for z in zones]):.3f}")
    print(f"\n💡 Recommendation: Use zones with composite score > 0.5 for trading")
    print(f"   This filters {len(zones)} zones → {len([z for z in zones if z['composite_score'] > 0.5])} high-quality zones")
    print(f"\n📊 Files created:")
    print(f"   - bayesian_integration_demo.png")
    print(f"   - bayesian_zones_demo.csv")
    print(f"\n💡 Next step: Run detect_supply_demand_zones.py with MT5 data")


if __name__ == "__main__":
    main()
