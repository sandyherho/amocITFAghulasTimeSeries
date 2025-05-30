#!/usr/bin/env python3
"""
Robust Causality Analysis for Ocean Transport Time Series
Analyzes ITF → Agulhas → AMOC pathways using multiple causality metrics
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime

# Create output directories if they don't exist
os.makedirs('../data/processed_data', exist_ok=True)
os.makedirs('../stats', exist_ok=True)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_align_data():
    """Load all time series and align to common time period"""
    
    # Load data
    itf_df = pd.read_csv('../data/raw_data/itf_ts.csv')
    agulhas_df = pd.read_csv('../data/raw_data/aghulas_ts.csv')
    amoc_df = pd.read_csv('../data/raw_data/amoc_ts.csv')
    
    # Convert time to datetime
    itf_df['time'] = pd.to_datetime(itf_df['time'])
    agulhas_df['time'] = pd.to_datetime(agulhas_df['time'])
    amoc_df['time'] = pd.to_datetime(amoc_df['time'])
    
    # Set time as index
    itf_df.set_index('time', inplace=True)
    agulhas_df.set_index('time', inplace=True)
    amoc_df.set_index('time', inplace=True)
    
    # Find common time period
    start_time = max(itf_df.index.min(), agulhas_df.index.min(), amoc_df.index.min())
    end_time = min(itf_df.index.max(), agulhas_df.index.max(), amoc_df.index.max())
    
    print(f"Common period: {start_time} to {end_time}")
    
    # Align to common period
    itf_aligned = itf_df.loc[start_time:end_time]
    agulhas_aligned = agulhas_df.loc[start_time:end_time]
    amoc_aligned = amoc_df.loc[start_time:end_time]
    
    # Combine into single dataframe
    combined_df = pd.concat([itf_aligned, agulhas_aligned, amoc_aligned], axis=1)
    
    # Handle any missing values
    combined_df = combined_df.interpolate(method='linear', limit=3)
    combined_df = combined_df.dropna()
    
    # Save processed data
    combined_df.to_csv('../data/processed_data/aligned_ocean_transport_data.csv')
    
    return combined_df

# ============================================================================
# 2. CAUSALITY METRICS
# ============================================================================

def maximum_cross_correlation(x, y, max_lag=24):
    """
    Calculate maximum cross-correlation and optimal lag
    Positive lag means x leads y
    """
    # Standardize series
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    
    # Calculate cross-correlation
    correlations = []
    lags = range(-max_lag, max_lag + 1)
    
    for lag in lags:
        if lag < 0:
            corr = np.corrcoef(x[:lag], y[-lag:])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(x[lag:], y[:-lag])[0, 1]
        else:
            corr = np.corrcoef(x, y)[0, 1]
        correlations.append(corr)
    
    # Find maximum absolute correlation
    correlations = np.array(correlations)
    max_idx = np.argmax(np.abs(correlations))
    
    return correlations[max_idx], lags[max_idx]

def convergent_cross_mapping(x, y, lib_sizes=None, E=3, tau=1):
    """
    Simplified CCM implementation
    Tests if historical values of x help predict y
    """
    if lib_sizes is None:
        # Adaptive library sizes based on data length
        n = len(x)
        lib_sizes = [int(n * frac) for frac in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9] if int(n * frac) > E+2]
    
    # Create embedding
    def create_embedding(ts, E, tau):
        n = len(ts)
        if n < (E-1)*tau + 1:
            return None
        embed = np.zeros((n - (E-1)*tau, E))
        for i in range(E):
            embed[:, i] = ts[i*tau:n-(E-1-i)*tau]
        return embed
    
    # Normalize
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y_norm = (y - np.mean(y)) / (np.std(y) + 1e-10)
    
    # Create embeddings
    x_embed = create_embedding(x_norm, E, tau)
    y_embed = create_embedding(y_norm, E, tau)
    
    if x_embed is None or y_embed is None or len(x_embed) < E+2:
        return 0
    
    # CCM: use y embedding to predict x
    correlations = []
    for lib_size in lib_sizes:
        if lib_size > len(x_embed) - 1:
            continue
            
        # Random sampling
        pred_corrs = []
        for _ in range(5):  # Reduced bootstrap for speed
            idx = np.random.choice(len(x_embed), size=min(lib_size, len(x_embed)-1), replace=False)
            
            # Find nearest neighbors in y_embed to predict x
            predictions = []
            actuals = []
            
            for i in range(len(idx)):
                test_idx = idx[i]
                train_idx = np.delete(idx, i)  # Leave one out
                
                if len(train_idx) < E+1:
                    continue
                
                # Find E+1 nearest neighbors
                distances = np.sum((y_embed[train_idx] - y_embed[test_idx])**2, axis=1)
                nn_indices = np.argsort(distances)[:min(E+1, len(train_idx))]
                nn_idx = train_idx[nn_indices]
                
                # Weight by distance
                weights = np.exp(-distances[nn_indices])
                weights /= (np.sum(weights) + 1e-10)
                
                # Predict x from neighbors
                x_pred = np.sum(weights * x_norm[nn_idx])
                predictions.append(x_pred)
                actuals.append(x_norm[test_idx])
            
            if len(predictions) > 2:
                corr = np.corrcoef(predictions, actuals)[0, 1]
                if not np.isnan(corr):
                    pred_corrs.append(corr)
        
        if pred_corrs:
            correlations.append(np.mean(pred_corrs))
    
    # Return maximum correlation as CCM strength
    return max(correlations) if correlations else 0

def transfer_entropy(x, y, k=1, lag=1, bins=10):
    """
    Calculate transfer entropy from x to y
    TE(X→Y) = information about Y_future given X_past beyond Y_past
    """
    n = len(x)
    
    # Need enough data points
    if n < lag + k + 10:
        return 0
    
    # Discretize the data
    x_discrete = pd.cut(x, bins=bins, labels=False).astype(str)
    y_discrete = pd.cut(y, bins=bins, labels=False).astype(str)
    
    # Create past/future vectors with proper alignment
    # We need: y_past[t-k:t], x_past[t-k:t], y_future[t+lag]
    
    # Arrays for the analysis
    y_past_vec = []
    x_past_vec = []
    y_future_vec = []
    
    # Build the vectors
    for t in range(k, n - lag):
        # Past values
        y_past_str = '_'.join(y_discrete[t-k:t])
        x_past_str = '_'.join(x_discrete[t-k:t])
        y_future_str = y_discrete[t + lag - 1]
        
        y_past_vec.append(y_past_str)
        x_past_vec.append(x_past_str)
        y_future_vec.append(y_future_str)
    
    # Convert to arrays
    y_past_vec = np.array(y_past_vec)
    x_past_vec = np.array(x_past_vec)
    y_future_vec = np.array(y_future_vec)
    
    # Create joint past state
    joint_past_vec = np.array([f"{xp}|{yp}" for xp, yp in zip(x_past_vec, y_past_vec)])
    
    # Calculate mutual information terms
    # TE = I(Y_future; X_past | Y_past) ≈ I(Y_future; X_past, Y_past) - I(Y_future; Y_past)
    
    try:
        mi_joint = mutual_info_score(y_future_vec, joint_past_vec)
        mi_y_only = mutual_info_score(y_future_vec, y_past_vec)
        
        te = max(0, mi_joint - mi_y_only)
        
        # Normalize by entropy of Y_future
        unique_y = np.unique(y_future_vec)
        probs = np.array([np.sum(y_future_vec == val) / len(y_future_vec) for val in unique_y])
        h_y = -np.sum(probs * np.log2(probs + 1e-10))
        
        te_normalized = te / h_y if h_y > 0 else 0
        
        return te_normalized
        
    except Exception as e:
        # If calculation fails, return 0
        return 0

# ============================================================================
# 3. SIGNIFICANCE TESTING
# ============================================================================

def block_bootstrap_significance(x, y, metric_func, n_surrogates=1000, block_length=None):
    """
    Test significance using block bootstrap to preserve autocorrelation
    """
    n = len(x)
    
    # Estimate block length if not provided
    if block_length is None:
        # Use autocorrelation to estimate
        acf_x = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')[n-1:]
        acf_x = acf_x / acf_x[0]
        decorr_time = np.where(acf_x < 1/np.e)[0][0] if np.any(acf_x < 1/np.e) else 10
        block_length = max(5, min(30, decorr_time * 2))
    
    # Calculate original metric
    original_metric = metric_func(x, y)
    
    # Generate surrogates
    surrogate_metrics = []
    for _ in range(n_surrogates):
        # Block permutation
        n_blocks = n // block_length
        block_indices = np.random.permutation(n_blocks)
        
        # Reconstruct surrogate series
        y_surrogate = []
        for idx in block_indices:
            start = idx * block_length
            end = min(start + block_length, n)
            y_surrogate.extend(y[start:end])
        
        y_surrogate = np.array(y_surrogate[:n])
        
        # Calculate metric for surrogate
        surrogate_metric = metric_func(x, y_surrogate)
        surrogate_metrics.append(surrogate_metric)
    
    # Calculate p-value
    surrogate_metrics = np.array(surrogate_metrics)
    if original_metric > 0:
        p_value = np.mean(surrogate_metrics >= original_metric)
    else:
        p_value = np.mean(surrogate_metrics <= original_metric)
    
    return p_value, original_metric

# ============================================================================
# 4. MAIN ANALYSIS
# ============================================================================

def analyze_pathway(x, y, x_name, y_name, max_lag=24):
    """Analyze causal relationship between two time series"""
    
    results = {
        'pathway': f"{x_name} → {y_name}",
        'MCC': np.nan,
        'MCC_lag': np.nan,
        'MCC_pval': np.nan,
        'CCM': np.nan,
        'CCM_pval': np.nan,
        'TE': np.nan,
        'TE_pval': np.nan,
        'consensus': 0
    }
    
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 50:  # Too few points
        return results
    
    # 1. Maximum Cross-Correlation
    mcc, lag = maximum_cross_correlation(x_clean, y_clean, max_lag)
    results['MCC'] = mcc
    results['MCC_lag'] = lag
    
    # MCC significance
    p_val, _ = block_bootstrap_significance(
        x_clean, y_clean, 
        lambda a, b: abs(maximum_cross_correlation(a, b, max_lag)[0]),
        n_surrogates=500
    )
    results['MCC_pval'] = p_val
    
    # 2. Convergent Cross Mapping
    ccm = convergent_cross_mapping(x_clean, y_clean)
    results['CCM'] = ccm
    
    # CCM significance
    p_val, _ = block_bootstrap_significance(
        x_clean, y_clean,
        lambda a, b: convergent_cross_mapping(a, b),
        n_surrogates=200  # CCM is computationally intensive
    )
    results['CCM_pval'] = p_val
    
    # 3. Transfer Entropy
    te = transfer_entropy(x_clean, y_clean)
    results['TE'] = te
    
    # TE significance
    p_val, _ = block_bootstrap_significance(
        x_clean, y_clean,
        lambda a, b: transfer_entropy(a, b),
        n_surrogates=500
    )
    results['TE_pval'] = p_val
    
    # Calculate consensus score
    consensus = 0
    if results['MCC_pval'] < 0.05:
        consensus += 1
    if results['CCM_pval'] < 0.05:
        consensus += 1
    if results['TE_pval'] < 0.05:
        consensus += 1
    results['consensus'] = consensus
    
    return results

def main():
    """Run complete causality analysis"""
    
    print("Loading and aligning data...")
    data = load_and_align_data()
    
    # Define all pathways to test
    pathways = [
        # ITF → Agulhas
        ('itf_g', 'aghulas_box'),
        ('itf_g', 'aghulas_jet'),
        ('itf_t', 'aghulas_box'),
        ('itf_t', 'aghulas_jet'),
        ('itf_s', 'aghulas_box'),
        ('itf_s', 'aghulas_jet'),
        
        # Agulhas → AMOC
        ('aghulas_box', 'amoc'),
        ('aghulas_jet', 'amoc'),
        
        # ITF → AMOC (direct)
        ('itf_g', 'amoc'),
        ('itf_t', 'amoc'),
        ('itf_s', 'amoc'),
    ]
    
    # Analyze all pathways
    results_list = []
    
    print("\nAnalyzing pathways...")
    for x_name, y_name in pathways:
        print(f"  Analyzing {x_name} → {y_name}...")
        
        x = data[x_name].values
        y = data[y_name].values
        
        results = analyze_pathway(x, y, x_name, y_name)
        results_list.append(results)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Add significance stars
    def add_stars(row, metric):
        pval = row[f'{metric}_pval']
        val = row[metric]
        if pval < 0.01:
            return f"{val:.3f}**"
        elif pval < 0.05:
            return f"{val:.3f}*"
        else:
            return f"{val:.3f}"
    
    results_df['MCC_str'] = results_df.apply(lambda x: add_stars(x, 'MCC'), axis=1)
    results_df['CCM_str'] = results_df.apply(lambda x: add_stars(x, 'CCM'), axis=1)
    results_df['TE_str'] = results_df.apply(lambda x: add_stars(x, 'TE'), axis=1)
    
    # Sort by consensus score
    results_df = results_df.sort_values('consensus', ascending=False)
    
    # Save detailed results
    results_df.to_csv('../data/processed_data/causality_analysis_results.csv', index=False)
    
    # Create summary table
    summary_df = results_df[['pathway', 'MCC_str', 'MCC_lag', 'CCM_str', 'TE_str', 'consensus']]
    summary_df.columns = ['Pathway', 'MCC (r)', 'Lag (months)', 'CCM (ρ)', 'TE (bits)', 'Consensus']
    
    # Save summary table
    summary_df.to_csv('../data/processed_data/causality_summary_table.csv', index=False)
    
    # ========================================================================
    # 5. GENERATE INTERPRETATION REPORT
    # ========================================================================
    
    print("\nGenerating interpretation report...")
    
    report = []
    report.append("="*80)
    report.append("ROBUST OCEAN TRANSPORT CAUSALITY ANALYSIS")
    report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*80)
    report.append("")
    
    # Data summary
    report.append("DATA SUMMARY:")
    report.append(f"- Common analysis period: {data.index[0]} to {data.index[-1]}")
    report.append(f"- Number of months: {len(data)}")
    report.append(f"- Time series analyzed: ITF (G, T, S), Agulhas (Box, Jet), AMOC")
    report.append("")
    
    # Methods summary
    report.append("METHODS:")
    report.append("1. Maximum Cross-Correlation (MCC): Linear correlation at optimal lag")
    report.append("2. Convergent Cross Mapping (CCM): Nonlinear causality test")
    report.append("3. Transfer Entropy (TE): Information flow metric")
    report.append("- Significance: Block bootstrap (p<0.05*, p<0.01**)")
    report.append("- Consensus: Number of significant methods (0-3)")
    report.append("")
    
    # Key findings
    report.append("KEY FINDINGS:")
    report.append("")
    
    # Find dominant pathways
    dominant = results_df[results_df['consensus'] >= 2]
    
    if len(dominant) > 0:
        report.append("DOMINANT PATHWAYS (Consensus ≥ 2):")
        for _, row in dominant.iterrows():
            report.append(f"\n{row['pathway']}:")
            report.append(f"  - MCC: {row['MCC_str']} at lag {row['MCC_lag']} months")
            report.append(f"  - CCM: {row['CCM_str']}")
            report.append(f"  - TE: {row['TE_str']}")
            report.append(f"  - Consensus: {row['consensus']}/3 methods significant")
            
            # Physical interpretation
            if 'itf' in row['pathway'] and 'aghulas' in row['pathway']:
                if abs(row['MCC_lag']) >= 6 and abs(row['MCC_lag']) <= 12:
                    report.append(f"  - Physical: Lag of {abs(row['MCC_lag'])} months consistent with")
                    report.append("    Indian Ocean transit time for water masses")
            elif 'aghulas' in row['pathway'] and 'amoc' in row['pathway']:
                if abs(row['MCC_lag']) >= 6 and abs(row['MCC_lag']) <= 18:
                    report.append(f"  - Physical: Lag of {abs(row['MCC_lag'])} months consistent with")
                    report.append("    Agulhas leakage propagation to North Atlantic")
    
    report.append("")
    
    # Pathway rankings
    report.append("PATHWAY RANKINGS BY METRIC:")
    report.append("")
    
    # By MCC
    report.append("Strongest Cross-Correlations:")
    mcc_sorted = results_df.nlargest(3, 'MCC')[['pathway', 'MCC_str', 'MCC_lag']]
    for _, row in mcc_sorted.iterrows():
        report.append(f"  {row['pathway']}: {row['MCC_str']} at {row['MCC_lag']} months")
    
    report.append("")
    
    # By CCM
    report.append("Strongest Nonlinear Causality (CCM):")
    ccm_sorted = results_df.nlargest(3, 'CCM')[['pathway', 'CCM_str']]
    for _, row in ccm_sorted.iterrows():
        report.append(f"  {row['pathway']}: {row['CCM_str']}")
    
    report.append("")
    
    # By TE
    report.append("Strongest Information Flow (TE):")
    te_sorted = results_df.nlargest(3, 'TE')[['pathway', 'TE_str']]
    for _, row in te_sorted.iterrows():
        report.append(f"  {row['pathway']}: {row['TE_str']}")
    
    report.append("")
    
    # Physical mechanisms
    report.append("PHYSICAL INTERPRETATION:")
    report.append("")
    
    # Check for specific patterns
    itf_agulhas_dominant = dominant[dominant['pathway'].str.contains('itf') & 
                                   dominant['pathway'].str.contains('aghulas')]
    
    if len(itf_agulhas_dominant) > 0:
        report.append("ITF → Agulhas Connection:")
        for _, row in itf_agulhas_dominant.iterrows():
            component = row['pathway'].split(' → ')[0].split('_')[1].upper()
            agulhas_type = row['pathway'].split(' → ')[1].split('_')[1].title()
            
            if component == 'T':
                report.append(f"- Temperature component drives {agulhas_type} transport")
                report.append("  Mechanism: Thermal stratification affects Agulhas retroflection")
            elif component == 'S':
                report.append(f"- Salinity component drives {agulhas_type} transport")
                report.append("  Mechanism: Salinity controls water mass density and Indian Ocean")
                report.append("  thermocline depth, modulating Agulhas strength")
            elif component == 'G':
                report.append(f"- Geostrophic component drives {agulhas_type} transport")
                report.append("  Mechanism: Direct volume transport continuity")
    
    report.append("")
    
    agulhas_amoc_dominant = dominant[dominant['pathway'].str.contains('aghulas') & 
                                    dominant['pathway'].str.contains('amoc')]
    
    if len(agulhas_amoc_dominant) > 0:
        report.append("Agulhas → AMOC Connection:")
        for _, row in agulhas_amoc_dominant.iterrows():
            agulhas_type = row['pathway'].split(' → ')[0].split('_')[1].title()
            report.append(f"- {agulhas_type} transport influences AMOC")
            
            if 'box' in row['pathway'].lower():
                report.append("  Mechanism: Box transport captures full Agulhas leakage")
                report.append("  including rings and filaments entering Atlantic")
            else:
                report.append("  Mechanism: Jet transport represents core current strength")
    
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS FOR FURTHER ANALYSIS:")
    report.append("")
    
    if len(dominant) > 0:
        strongest_pathway = dominant.iloc[0]['pathway']
        report.append(f"1. Focus wavelet analysis on: {strongest_pathway}")
        report.append("   (Highest consensus score pathway)")
        
        # Check for chain
        if 'itf' in strongest_pathway and 'aghulas' in strongest_pathway:
            itf_component = strongest_pathway.split(' → ')[0]
            agulhas_component = strongest_pathway.split(' → ')[1]
            
            # Find corresponding Agulhas->AMOC
            agulhas_amoc = results_df[results_df['pathway'].str.contains(f"{agulhas_component} → amoc")]
            if len(agulhas_amoc) > 0 and agulhas_amoc.iloc[0]['consensus'] > 0:
                report.append(f"2. Examine full chain: {itf_component} → {agulhas_component} → amoc")
        
        report.append("3. Test stability during different climate phases (ENSO, IOD)")
        report.append("4. Examine seasonal modulation of identified pathways")
    
    report.append("")
    report.append("="*80)
    report.append("END OF REPORT")
    
    # Save report
    with open('../stats/ocean_transport_causality_interpretation.txt', 'w') as f:
        f.write('\n'.join(report))
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nDominant Pathways (Consensus ≥ 2):")
    if len(dominant) > 0:
        for _, row in dominant.iterrows():
            print(f"  {row['pathway']}: Consensus = {row['consensus']}/3")
    else:
        print("  No pathways with consensus ≥ 2")
    
    print("\nFiles saved:")
    print("  - ../data/processed_data/aligned_ocean_transport_data.csv")
    print("  - ../data/processed_data/causality_analysis_results.csv")
    print("  - ../data/processed_data/causality_summary_table.csv")
    print("  - ../stats/ocean_transport_causality_interpretation.txt")
    
    return results_df

if __name__ == "__main__":
    results = main()
