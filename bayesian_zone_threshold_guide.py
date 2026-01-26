"""
Bayesian Zone Filtering - Quick Reference Guide
================================================

This guide explains how to use composite score thresholds to filter
supply/demand zones by quality.
"""

# ============================================================================
# COMPOSITE SCORE SCALE
# ============================================================================

"""
Composite Score Range: 0.0 to 1.0

Components:
- Traditional Strength (40%): Reversal magnitude + touch count
- Bayesian Probability (40%): Dynamic probability from price action
- KDE Density (20%): Clustering of reversal points

Score Interpretation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Score Range  │ Quality Level      │ Recommendation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
0.8 - 1.0    │ VERY HIGH         │ ✅ Primary trading zones
0.6 - 0.8    │ HIGH              │ ✅ Good secondary zones
0.5 - 0.6    │ MEDIUM-HIGH       │ ⚠️  Use with confirmation
0.3 - 0.5    │ MEDIUM            │ ⚠️  Watch only, don't trade
0.0 - 0.3    │ LOW               │ ❌ Ignore (likely noise)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ============================================================================
# RECOMMENDED THRESHOLDS BY USE CASE
# ============================================================================

THRESHOLDS = {
    # Conservative trading (fewer, higher quality zones)
    'conservative': 0.7,
    
    # Balanced trading (good quality zones)
    'balanced': 0.6,
    
    # Moderate trading (includes medium-high zones)
    'moderate': 0.5,
    
    # Aggressive trading (more zones, requires confirmation)
    'aggressive': 0.4,
    
    # Visualization only (show medium+ zones)
    'visualization': 0.3,
    
    # Analysis/research (show all zones)
    'research': 0.0
}

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

# Example 1: Filter zones for conservative trading
def get_conservative_zones(detector):
    """Get only the highest quality zones."""
    return detector.get_high_probability_zones(threshold=0.7)

# Example 2: Filter zones for balanced trading
def get_balanced_zones(detector):
    """Get high-quality zones for trading."""
    return detector.get_high_probability_zones(threshold=0.6)

# Example 3: Custom filtering with additional criteria
def get_custom_filtered_zones(detector, min_score=0.6, min_touches=3):
    """
    Custom filtering combining composite score and touch count.
    
    Args:
        min_score: Minimum composite score (0-1)
        min_touches: Minimum number of times price tested the zone
    """
    return [
        z for z in detector.zones 
        if z['composite_score'] >= min_score and z['touches'] >= min_touches
    ]

# Example 4: Separate supply and demand zones
def get_filtered_zones_by_type(detector, min_score=0.6):
    """Get filtered zones separated by type."""
    high_prob_zones = detector.get_high_probability_zones(threshold=min_score)
    
    supply_zones = [z for z in high_prob_zones if z['type'] == 'supply']
    demand_zones = [z for z in high_prob_zones if z['type'] == 'demand']
    
    return supply_zones, demand_zones

# Example 5: Position sizing based on composite score
def calculate_position_size(base_size, zone_score):
    """
    Scale position size by zone quality.
    
    Args:
        base_size: Base position size (e.g., 0.01 lots)
        zone_score: Composite score of the zone (0-1)
    
    Returns:
        Scaled position size
    """
    if zone_score >= 0.8:
        return base_size * 1.5  # 50% larger for very high probability
    elif zone_score >= 0.6:
        return base_size * 1.0  # Full size for high probability
    elif zone_score >= 0.5:
        return base_size * 0.75  # 75% size for medium-high
    else:
        return 0  # Don't trade zones below 0.5

# ============================================================================
# DEMO SCRIPT USAGE
# ============================================================================

"""
To change the visualization threshold in bayesian_zone_integration_demo.py:

# Show only high-quality zones (recommended for trading)
visualize_integrated_zones(df, zones, bayesian_detector, kde_detector, 
                           sample_bars=500, min_score=0.6)

# Show medium+ zones (current default)
visualize_integrated_zones(df, zones, bayesian_detector, kde_detector, 
                           sample_bars=500, min_score=0.3)

# Show all zones (for analysis)
visualize_integrated_zones(df, zones, bayesian_detector, kde_detector, 
                           sample_bars=500, min_score=0.0)
"""

# ============================================================================
# REAL-WORLD EXAMPLE
# ============================================================================

def trading_example():
    """
    Example of using Bayesian zones in a trading system.
    """
    from detect_supply_demand_zones import SupplyDemandZoneDetector
    import MetaTrader5 as mt5
    
    # Initialize detector
    detector = SupplyDemandZoneDetector('EURUSDm', mt5.TIMEFRAME_M15)
    detector.init_mt5()
    detector.fetch_data()
    
    # Detect zones with Bayesian scoring
    sideways_mask = detector.identify_sideways_periods()
    zones = detector.detect_zones(sideways_mask)
    
    # Get high-quality zones for trading
    trading_zones = detector.get_high_probability_zones(threshold=0.6)
    
    print(f"Total zones detected: {len(zones)}")
    print(f"High-quality trading zones: {len(trading_zones)}")
    
    # Separate by type
    supply_zones = [z for z in trading_zones if z['type'] == 'supply']
    demand_zones = [z for z in trading_zones if z['type'] == 'demand']
    
    print(f"\nTrading Setup:")
    print(f"  Supply zones (resistance): {len(supply_zones)}")
    print(f"  Demand zones (support): {len(demand_zones)}")
    
    # Show top 5 zones by score
    top_zones = sorted(trading_zones, key=lambda x: x['composite_score'], reverse=True)[:5]
    
    print(f"\nTop 5 Zones:")
    for i, zone in enumerate(top_zones, 1):
        print(f"  {i}. {zone['type'].upper()} @ {zone['price']:.5f}")
        print(f"     Score: {zone['composite_score']:.3f}")
        print(f"     Bayesian P: {zone['bayesian_probability']:.3f}")
        print(f"     Touches: {zone['touches']}")

# ============================================================================
# THRESHOLD COMPARISON
# ============================================================================

def compare_thresholds(detector):
    """
    Compare how many zones pass different thresholds.
    Useful for finding the right balance.
    """
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    print("Threshold Comparison:")
    print("=" * 50)
    
    for threshold in thresholds:
        zones = detector.get_high_probability_zones(threshold=threshold)
        supply = sum(1 for z in zones if z['type'] == 'supply')
        demand = sum(1 for z in zones if z['type'] == 'demand')
        
        print(f"Threshold {threshold:.1f}: {len(zones):3d} zones "
              f"(Supply: {supply:2d}, Demand: {demand:2d})")
    
    print("=" * 50)
    print("\n💡 Recommendation:")
    print("   - For trading: Use 0.6 or higher")
    print("   - For visualization: Use 0.3-0.5")
    print("   - For analysis: Use 0.0 (all zones)")

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 70)
    print("THRESHOLD RECOMMENDATIONS")
    print("=" * 70)
    
    for use_case, threshold in THRESHOLDS.items():
        print(f"{use_case.upper():15s}: {threshold:.1f}")
    
    print("\n" + "=" * 70)
    print("ANSWER TO YOUR QUESTION:")
    print("=" * 70)
    print("""
Is 0.3 a strong threshold?

NO - 0.3 is a LENIENT threshold suitable for visualization.

For actual trading, use:
  • 0.6 (balanced) - Good for most strategies
  • 0.7 (conservative) - Best for high-probability setups
  • 0.5 (moderate) - If you want more zones with confirmation

The demo uses 0.3 to show you what zones exist, but you should
filter to 0.6+ for real trading decisions.

Example:
  115 total zones detected
  → 3 zones with score ≥ 0.3 (visualization)
  → Likely 0-2 zones with score ≥ 0.6 (trading)
    """)
