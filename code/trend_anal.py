#!/usr/bin/env python

"""
Ocean Current Theil-Sen Trend Analysis

Sandy H. S. Herho <sandy.herho@email.ucr.edu>
2025/05/29
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import TheilSenRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define directories
data_dir = Path('../data/raw_data')
processed_dir = Path('../data/processed_data')
stats_dir = Path('../stats')
figs_dir = Path('../figs')

# Create directories if they don't exist
processed_dir.mkdir(parents=True, exist_ok=True)
stats_dir.mkdir(parents=True, exist_ok=True)
figs_dir.mkdir(parents=True, exist_ok=True)

def parse_time_column(df, time_col='time'):
    """Parse time column and extract date information"""
    print(f"  Parsing time column...")
    
    # Try different date formats
    date_formats = ['%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y', '%Y-%m', '%Y/%m']
    
    for fmt in date_formats:
        try:
            df['datetime'] = pd.to_datetime(df[time_col], format=fmt)
            break
        except:
            continue
    else:
        try:
            df['datetime'] = pd.to_datetime(df[time_col])
        except:
            df['datetime'] = pd.to_datetime(df[time_col], errors='coerce')
    
    # Create decimal year for regression
    df['decimal_year'] = df['datetime'].dt.year + (df['datetime'].dt.dayofyear - 1) / 365.25
    return df

def calculate_simple_theil_sen(df, value_cols, time_col='decimal_year'):
    """Calculate Theil-Sen trends with p-values"""
    trend_results = {}
    
    for col in value_cols:
        print(f"    Calculating trend for {col}...")
        
        # Remove NaN values
        mask = ~(df[col].isna() | df[time_col].isna())
        clean_data = df[mask].copy()
        
        if len(clean_data) < 10:
            print(f"      Warning: Insufficient data for {col}")
            continue
            
        X = clean_data[time_col].values.reshape(-1, 1)
        y = clean_data[col].values
        
        # Fit Theil-Sen regression
        theil_sen = TheilSenRegressor(random_state=42, max_iter=300, tol=1e-2)
        theil_sen.fit(X, y)
        
        # Get trend statistics
        slope = theil_sen.coef_[0]
        intercept = theil_sen.intercept_
        
        # Calculate predictions
        y_pred = theil_sen.predict(X)
        
        # Mann-Kendall test for significance
        def mann_kendall_test(x):
            n = len(x)
            s = 0
            for i in range(n-1):
                s += np.sum(np.sign(x[i+1:] - x[i]))
            
            # Variance calculation
            var_s = n * (n - 1) * (2 * n + 5) / 18
            
            if var_s == 0:
                return s, 0, 1.0
                
            if s > 0:
                z = (s - 1) / np.sqrt(var_s)
            elif s < 0:
                z = (s + 1) / np.sqrt(var_s)
            else:
                z = 0
                
            # Two-tailed test
            p_mk = 2 * (1 - stats.norm.cdf(abs(z)))
            
            return s, z, p_mk
        
        mk_s, mk_z, mk_p = mann_kendall_test(clean_data[col])
        
        # Store results with p-values
        trend_results[col] = {
            'slope': slope,
            'intercept': intercept,
            'mann_kendall_p': mk_p,
            'significant': mk_p < 0.05,
            'start_year': clean_data[time_col].min(),
            'end_year': clean_data[time_col].max(),
            'n_points': len(clean_data),
            'data': clean_data,
            'predictions': y_pred
        }
        
        significance = "significant" if mk_p < 0.05 else "not significant"
        print(f"      ✓ {col}: slope = {slope:.6f} Sv/year ({significance})")
    
    return trend_results

def load_and_process_data():
    """Load and process all datasets"""
    
    print("Loading and processing data...")
    
    # Load ITF data
    print("Loading ITF data...")
    itf_df = pd.read_csv(data_dir / 'itf_ts.csv')
    itf_df = parse_time_column(itf_df)
    itf_trends = calculate_simple_theil_sen(itf_df, ['itf_g', 'itf_t', 'itf_s'])
    
    # Load Agulhas data
    print("\nLoading Agulhas data...")
    agulhas_df = pd.read_csv(data_dir / 'aghulas_ts.csv')
    agulhas_df = parse_time_column(agulhas_df)
    agulhas_trends = calculate_simple_theil_sen(agulhas_df, ['aghulas_box', 'aghulas_jet'])
    
    # Load AMOC data
    print("\nLoading AMOC data...")
    amoc_df = pd.read_csv(data_dir / 'amoc_ts.csv')
    amoc_df = parse_time_column(amoc_df)
    amoc_trends = calculate_simple_theil_sen(amoc_df, ['amoc'])
    
    return {
        'itf': itf_trends,
        'agulhas': agulhas_trends,
        'amoc': amoc_trends
    }

def create_theil_sen_trend_plot(datasets):
    """Create 3-row Theil-Sen trend plot with consistent time axes (1984-2023)"""
    
    print("Creating Theil-Sen trend plot...")
    
    # Define colors
    itf_colors = ['#E74C3C', '#2ECC71', '#3498DB']  # Red, Green, Blue
    agulhas_colors = ['#9B59B6', '#F39C12']  # Purple, Orange
    amoc_colors = ['#1ABC9C']  # Teal
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.subplots_adjust(hspace=0.1, bottom=0.15)
    
    # Define consistent time axis (1984-2023)
    start_year = 1984
    end_year = 2023
    
    # Collect legend info
    all_lines = []
    all_labels = []
    
    # Plot 1: ITF trends
    ax1 = axes[0]
    itf_vars = ['itf_g', 'itf_t', 'itf_s']
    itf_labels = ['ITF-G', 'ITF-T', 'ITF-S']
    
    for i, (var, label, color) in enumerate(zip(itf_vars, itf_labels, itf_colors)):
        if var in datasets['itf']:
            trend_data = datasets['itf'][var]
            data = trend_data['data']
            
            # Plot scatter points
            ax1.scatter(data['decimal_year'], data[var], alpha=0.5, color=color, s=20)
            
            # Plot trend line only over data period
            line = ax1.plot(data['decimal_year'], trend_data['predictions'], 
                          color=color, linewidth=3.0, label=label, alpha=0.9)[0]
            
            all_lines.append(line)
            all_labels.append(label)
    
    ax1.set_xlim(start_year, end_year)  # Consistent time axis
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    # No title
    
    # Plot 2: Agulhas trends
    ax2 = axes[1]
    agulhas_vars = ['aghulas_box', 'aghulas_jet']
    agulhas_labels = ['Agulhas Box', 'Agulhas Jet']
    
    for i, (var, label, color) in enumerate(zip(agulhas_vars, agulhas_labels, agulhas_colors)):
        if var in datasets['agulhas']:
            trend_data = datasets['agulhas'][var]
            data = trend_data['data']
            
            # Plot scatter points
            ax2.scatter(data['decimal_year'], data[var], alpha=0.5, color=color, s=20)
            
            # Plot trend line only over data period
            line = ax2.plot(data['decimal_year'], trend_data['predictions'], 
                          color=color, linewidth=3.0, label=label, alpha=0.9)[0]
            
            all_lines.append(line)
            all_labels.append(label)
    
    ax2.set_xlim(start_year, end_year)  # Consistent time axis
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    # No title
    
    # Plot 3: AMOC trends
    ax3 = axes[2]
    
    if 'amoc' in datasets['amoc']:
        trend_data = datasets['amoc']['amoc']
        data = trend_data['data']
        
        # Plot scatter points
        ax3.scatter(data['decimal_year'], data['amoc'], alpha=0.5, color=amoc_colors[0], s=20)
        
        # Plot trend line only over data period
        line = ax3.plot(data['decimal_year'], trend_data['predictions'], 
                      color=amoc_colors[0], linewidth=3.0, label='AMOC', alpha=0.9)[0]
        
        all_lines.append(line)
        all_labels.append('AMOC')
    
    ax3.set_xlim(start_year, end_year)  # Consistent time axis
    ax3.set_xlabel('Year', fontsize=20, fontweight='bold')
    ax3.set_ylabel('')
    # No title
    ax3.grid(True, alpha=0.3)
    
    # Add shared y-label
    fig.text(0, 0.5, 'Volumetric Transport [Sv]', va='center', rotation='vertical', 
             fontsize=20, fontweight='bold')
    
    # Style improvements
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.tick_params(labelsize=13)
        ax.set_ylabel('')
    
    # Create legend
    if all_lines:
        fig.legend(all_lines, all_labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                   ncol=6, frameon=True, fancybox=True, shadow=True, fontsize=11,
                   columnspacing=1.5, handletextpad=0.5)
    
    plt.tight_layout()
    return fig

def save_simple_trend_data(datasets):
    """Save trend data with p-values to CSV"""
    
    print("Saving trend data...")
    
    all_trends = []
    
    for dataset_name, trends in datasets.items():
        for var_name, trend_data in trends.items():
            all_trends.append({
                'dataset': dataset_name,
                'variable': var_name,
                'slope': trend_data['slope'],
                'intercept': trend_data['intercept'],
                'mann_kendall_p': trend_data['mann_kendall_p'],
                'significant': trend_data['significant'],
                'start_year': trend_data['start_year'],
                'end_year': trend_data['end_year'],
                'n_points': trend_data['n_points'],
                'trend_per_decade': trend_data['slope'] * 10  # Sv per decade
            })
    
    trends_df = pd.DataFrame(all_trends)
    trends_df.to_csv(processed_dir / 'theil_sen_trends_with_pvalues.csv', index=False)
    print("  ✓ Trend data saved")

def generate_simple_interpretation(datasets):
    """Generate trend interpretation with p-values"""
    
    print("Generating interpretation...")
    
    interpretation = []
    interpretation.append("OCEAN CURRENT THEIL-SEN TREND ANALYSIS WITH SIGNIFICANCE")
    interpretation.append("="*65)
    interpretation.append(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    interpretation.append("")
    interpretation.append("METHODOLOGY:")
    interpretation.append("- Theil-Sen estimator: Robust regression resistant to outliers")
    interpretation.append("- Mann-Kendall test: Non-parametric trend significance test")
    interpretation.append("- Significance level: p < 0.05")
    interpretation.append("- Time axis: All plots standardized to 1984-2023")
    interpretation.append("")
    
    # Collect all trends for summary
    significant_trends = []
    
    for dataset_name, trends in datasets.items():
        interpretation.append(f"\n{dataset_name.upper()} TRENDS:")
        interpretation.append("-" * 25)
        
        for var_name, trend_data in trends.items():
            interpretation.append(f"\n{var_name}:")
            interpretation.append(f"  Slope: {trend_data['slope']:.6f} Sv/year")
            interpretation.append(f"  Trend per decade: {trend_data['slope'] * 10:.4f} Sv")
            interpretation.append(f"  Mann-Kendall p-value: {trend_data['mann_kendall_p']:.6f}")
            interpretation.append(f"  Time period: {trend_data['start_year']:.1f} - {trend_data['end_year']:.1f}")
            interpretation.append(f"  Data points: {trend_data['n_points']}")
            
            direction = "increasing" if trend_data['slope'] > 0 else "decreasing"
            if trend_data['significant']:
                interpretation.append(f"  Interpretation: SIGNIFICANT {direction} trend (p < 0.05)")
                significant_trends.append(f"{dataset_name.upper()}_{var_name}: {direction}")
            else:
                interpretation.append(f"  Interpretation: NO significant trend detected (p ≥ 0.05)")
    
    # Summary section
    interpretation.append(f"\n\nSUMMARY:")
    interpretation.append("-" * 10)
    
    total_variables = sum(len(trends) for trends in datasets.values())
    n_significant = len(significant_trends)
    
    interpretation.append(f"Total variables analyzed: {total_variables}")
    interpretation.append(f"Variables with significant trends: {n_significant}")
    interpretation.append(f"Percentage with significant trends: {n_significant/total_variables*100:.1f}%")
    
    if significant_trends:
        interpretation.append(f"\nSignificant trends detected:")
        for trend in significant_trends:
            interpretation.append(f"  - {trend}")
    else:
        interpretation.append(f"\nNo significant trends detected at p < 0.05 level.")
    
    interpretation.append(f"\nNOTE:")
    interpretation.append("- All plots use consistent time axis (1984-2023)")
    interpretation.append("- Data gaps appear as blank periods")
    interpretation.append("- Trend lines only shown over available data periods")
    interpretation.append("- Theil-Sen regression is robust to outliers")
    
    # Save interpretation
    with open(stats_dir / 'theil_sen_interpretation_with_pvalues.txt', 'w') as f:
        f.write('\n'.join(interpretation))
    
    print("  ✓ Interpretation saved")

def main():
    """Main function"""
    print("Starting Theil-Sen Trend Analysis with P-values...")
    print("="*55)
    
    start_time = datetime.now()
    
    # Load and process data
    datasets = load_and_process_data()
    
    # Create and save trend plot
    print("\nCreating trend plot (1984-2023 time axis)...")
    fig = create_theil_sen_trend_plot(datasets)
    fig.savefig(figs_dir / 'theil_sen_trends_1984_2023.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    fig.savefig(figs_dir / 'theil_sen_trends_1984_2023.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("  ✓ Trend plot saved")
    
    # Save trend data
    save_simple_trend_data(datasets)
    
    # Generate interpretation
    generate_simple_interpretation(datasets)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\nTheil-Sen Analysis completed!")
    print(f"Computation time: {duration}")
    print("\nFiles created:")
    print("- ../figs/theil_sen_trends_1984_2023.png")
    print("- ../figs/theil_sen_trends_1984_2023.pdf")
    print("- ../data/processed_data/theil_sen_trends_with_pvalues.csv")
    print("- ../stats/theil_sen_interpretation_with_pvalues.txt")

if __name__ == "__main__":
    main()
