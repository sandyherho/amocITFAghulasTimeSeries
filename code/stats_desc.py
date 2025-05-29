#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ocean Current KDE Distribution Analysis
Creates KDE plots for ITF, Agulhas, and AMOC time series data
and provides comprehensive descriptive statistics.

Author: Sandy Herho <sandy.herho@email.ucr.edu>
Date: May 28, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

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
    # Try different date formats
    date_formats = ['%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y', '%Y-%m', '%Y/%m']
    
    for fmt in date_formats:
        try:
            df['datetime'] = pd.to_datetime(df[time_col], format=fmt)
            break
        except:
            continue
    else:
        # If no format works, try pandas automatic parsing
        try:
            df['datetime'] = pd.to_datetime(df[time_col])
        except:
            # If still fails, try to parse as year-month
            df['datetime'] = pd.to_datetime(df[time_col], errors='coerce')
    
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    return df

def calculate_descriptive_stats(df, value_cols):
    """Calculate comprehensive descriptive statistics"""
    stats_dict = {}
    
    for col in value_cols:
        # Remove NaN values
        data = df[col].dropna()
        
        # Basic descriptive statistics
        basic_stats = {
            'count': len(data),
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'range': data.max() - data.min(),
            'q25': data.quantile(0.25),
            'q75': data.quantile(0.75),
            'iqr': data.quantile(0.75) - data.quantile(0.25),
        }
        
        # Robust statistics
        robust_stats = {
            'mad': stats.median_abs_deviation(data),  # Median Absolute Deviation
            'trimmed_mean_10': stats.trim_mean(data, 0.1),  # 10% trimmed mean
            'trimmed_mean_20': stats.trim_mean(data, 0.2),  # 20% trimmed mean
        }
        
        # Distribution shape statistics
        shape_stats = {
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'cv': data.std() / data.mean() if data.mean() != 0 else np.nan,  # Coefficient of variation
        }
        
        # Percentiles
        percentiles = {
            'p01': data.quantile(0.01),
            'p05': data.quantile(0.05),
            'p10': data.quantile(0.10),
            'p90': data.quantile(0.90),
            'p95': data.quantile(0.95),
            'p99': data.quantile(0.99),
        }
        
        # Find extreme values and their dates
        min_idx = data.idxmin()
        max_idx = data.idxmax()
        
        extremes = {
            'min_value': data.min(),
            'min_date': df.loc[min_idx, 'datetime'] if 'datetime' in df.columns else df.loc[min_idx, 'time'],
            'max_value': data.max(),
            'max_date': df.loc[max_idx, 'datetime'] if 'datetime' in df.columns else df.loc[max_idx, 'time'],
        }
        
        # Combine all statistics
        stats_dict[col] = {
            **basic_stats,
            **robust_stats,
            **shape_stats,
            **percentiles,
            **extremes
        }
    
    return stats_dict

def load_and_process_data():
    """Load and process all datasets"""
    
    # Load ITF data
    print("Loading ITF data...")
    itf_df = pd.read_csv(data_dir / 'itf_ts.csv')
    itf_df = parse_time_column(itf_df)
    itf_stats = calculate_descriptive_stats(itf_df, ['itf_g', 'itf_t', 'itf_s'])
    
    # Load Agulhas data
    print("Loading Agulhas data...")
    agulhas_df = pd.read_csv(data_dir / 'aghulas_ts.csv')
    agulhas_df = parse_time_column(agulhas_df)
    agulhas_stats = calculate_descriptive_stats(agulhas_df, ['aghulas_box', 'aghulas_jet'])
    
    # Load AMOC data
    print("Loading AMOC data...")
    amoc_df = pd.read_csv(data_dir / 'amoc_ts.csv')
    amoc_df = parse_time_column(amoc_df)
    amoc_stats = calculate_descriptive_stats(amoc_df, ['amoc'])
    
    return {
        'itf': {'data': itf_df, 'stats': itf_stats},
        'agulhas': {'data': agulhas_df, 'stats': agulhas_stats},
        'amoc': {'data': amoc_df, 'stats': amoc_stats}
    }

def create_kde_plot(datasets):
    """Create publication-ready 3-row KDE plot"""
    
    # Define beautiful and highly distinguishable color palettes
    itf_colors = ['#E74C3C', '#2ECC71', '#3498DB']  # Red, Green, Blue
    agulhas_colors = ['#9B59B6', '#F39C12']  # Purple, Orange
    amoc_colors = ['#1ABC9C']  # Teal
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    fig.subplots_adjust(hspace=0.15, bottom=0.15)
    
    # Collect all line objects for combined legend
    all_lines = []
    all_labels = []
    
    # Plot 1: ITF KDE (3 variables)
    ax1 = axes[0]
    itf_vars = ['itf_g', 'itf_t', 'itf_s']
    itf_labels = ['ITF-G', 'ITF-T', 'ITF-S']
    
    for i, (var, label, color) in enumerate(zip(itf_vars, itf_labels, itf_colors)):
        data = datasets['itf']['data'][var].dropna()
        # Create KDE plot (line only, no fill)
        sns.kdeplot(data=data, ax=ax1, color=color, linewidth=3.0, alpha=0.8)
        # Create line for legend
        line = ax1.plot([], [], color=color, linewidth=3.0, label=label, alpha=0.8)[0]
        all_lines.append(line)
        all_labels.append(label)
    
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('')
    ax1.set_ylabel('')  # Remove ylabel after plotting
    
    # Plot 2: Agulhas KDE (2 variables)
    ax2 = axes[1]
    agulhas_vars = ['aghulas_box', 'aghulas_jet']
    agulhas_labels = ['Agulhas Box', 'Agulhas Jet']
    
    for i, (var, label, color) in enumerate(zip(agulhas_vars, agulhas_labels, agulhas_colors)):
        data = datasets['agulhas']['data'][var].dropna()
        # Create KDE plot (line only, no fill)
        sns.kdeplot(data=data, ax=ax2, color=color, linewidth=3.0, alpha=0.8)
        # Create line for legend
        line = ax2.plot([], [], color=color, linewidth=3.0, label=label, alpha=0.8)[0]
        all_lines.append(line)
        all_labels.append(label)
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('')
    ax2.set_ylabel('')  # Remove ylabel after plotting
    
    # Plot 3: AMOC KDE (1 variable)
    ax3 = axes[2]
    data = datasets['amoc']['data']['amoc'].dropna()
    # Create KDE plot (line only, no fill)
    sns.kdeplot(data=data, ax=ax3, color=amoc_colors[0], linewidth=3.0, alpha=0.8)
    # Create line for legend
    line = ax3.plot([], [], color=amoc_colors[0], linewidth=3.0, label='AMOC', alpha=0.8)[0]
    all_lines.append(line)
    all_labels.append('AMOC')
    
    ax3.set_xlabel('Volumetric Transport [Sv]', fontsize=20, fontweight='bold')
    ax3.set_ylabel('')  # Remove individual ylabel - only shared ylabel
    ax3.grid(True, alpha=0.3)
    
    # Add shared y-label (moved even more to the left and bigger)
    fig.text(0, 0.5, 'Density', va='center', rotation='vertical', 
             fontsize=20, fontweight='bold')
    
    # Style improvements
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.tick_params(labelsize=13)  # Bigger tick labels
    
    # Create combined legend at the bottom (positioned lower)
    fig.legend(all_lines, all_labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
               ncol=6, frameon=True, fancybox=True, shadow=True, fontsize=11,
               columnspacing=1.5, handletextpad=0.5)
    
    plt.tight_layout()
    return fig

def save_processed_data(datasets):
    """Save processed statistics data"""
    
    print("Saving processed statistics...")
    
    # Save ITF processed statistics
    itf_stats_df = pd.DataFrame(datasets['itf']['stats']).T
    itf_stats_df.to_csv(processed_dir / 'itf_descriptive_stats.csv')
    
    # Save Agulhas processed statistics
    agulhas_stats_df = pd.DataFrame(datasets['agulhas']['stats']).T
    agulhas_stats_df.to_csv(processed_dir / 'agulhas_descriptive_stats.csv')
    
    # Save AMOC processed statistics
    amoc_stats_df = pd.DataFrame(datasets['amoc']['stats']).T
    amoc_stats_df.to_csv(processed_dir / 'amoc_descriptive_stats.csv')

def generate_statistical_interpretation(datasets):
    """Generate comprehensive statistical interpretation"""
    
    print("Generating statistical interpretation...")
    
    interpretation = []
    interpretation.append("OCEAN CURRENT DISTRIBUTION ANALYSIS")
    interpretation.append("="*50)
    interpretation.append(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    interpretation.append("")
    
    # ITF Analysis
    interpretation.append("1. INDONESIAN THROUGHFLOW (ITF) - DISTRIBUTION ANALYSIS")
    interpretation.append("-" * 55)
    for var in ['itf_g', 'itf_t', 'itf_s']:
        stats = datasets['itf']['stats'][var]
        
        interpretation.append(f"\n{var.upper()}:")
        interpretation.append(f"  Basic Statistics:")
        interpretation.append(f"    Mean: {stats['mean']:.3f} Sv")
        interpretation.append(f"    Median: {stats['median']:.3f} Sv")
        interpretation.append(f"    Standard Deviation: {stats['std']:.3f} Sv")
        interpretation.append(f"    Range: {stats['range']:.3f} Sv ({stats['min']:.3f} to {stats['max']:.3f})")
        
        interpretation.append(f"  Robust Statistics:")
        interpretation.append(f"    Median Absolute Deviation: {stats['mad']:.3f} Sv")
        interpretation.append(f"    Interquartile Range (IQR): {stats['iqr']:.3f} Sv")
        interpretation.append(f"    10% Trimmed Mean: {stats['trimmed_mean_10']:.3f} Sv")
        interpretation.append(f"    20% Trimmed Mean: {stats['trimmed_mean_20']:.3f} Sv")
        
        interpretation.append(f"  Distribution Shape:")
        interpretation.append(f"    Skewness: {stats['skewness']:.3f}")
        interpretation.append(f"    Kurtosis: {stats['kurtosis']:.3f}")
        interpretation.append(f"    Coefficient of Variation: {stats['cv']*100:.2f}%")
        
        interpretation.append(f"  Extreme Values:")
        interpretation.append(f"    Maximum: {stats['max_value']:.3f} Sv on {stats['max_date']}")
        interpretation.append(f"    Minimum: {stats['min_value']:.3f} Sv on {stats['min_date']}")
        
        interpretation.append(f"  Percentiles:")
        interpretation.append(f"    5th-95th percentile range: {stats['p05']:.3f} to {stats['p95']:.3f} Sv")
        interpretation.append(f"    1st-99th percentile range: {stats['p01']:.3f} to {stats['p99']:.3f} Sv")
    
    # Agulhas Analysis
    interpretation.append("\n\n2. AGULHAS CURRENT SYSTEM - DISTRIBUTION ANALYSIS")
    interpretation.append("-" * 50)
    for var in ['aghulas_box', 'aghulas_jet']:
        stats = datasets['agulhas']['stats'][var]
        var_name = "Agulhas Box" if "box" in var else "Agulhas Jet"
        
        interpretation.append(f"\n{var_name}:")
        interpretation.append(f"  Basic Statistics:")
        interpretation.append(f"    Mean: {stats['mean']:.3f} Sv")
        interpretation.append(f"    Median: {stats['median']:.3f} Sv")
        interpretation.append(f"    Standard Deviation: {stats['std']:.3f} Sv")
        interpretation.append(f"    Range: {stats['range']:.3f} Sv ({stats['min']:.3f} to {stats['max']:.3f})")
        
        interpretation.append(f"  Robust Statistics:")
        interpretation.append(f"    Median Absolute Deviation: {stats['mad']:.3f} Sv")
        interpretation.append(f"    Interquartile Range (IQR): {stats['iqr']:.3f} Sv")
        interpretation.append(f"    10% Trimmed Mean: {stats['trimmed_mean_10']:.3f} Sv")
        interpretation.append(f"    20% Trimmed Mean: {stats['trimmed_mean_20']:.3f} Sv")
        
        interpretation.append(f"  Distribution Shape:")
        interpretation.append(f"    Skewness: {stats['skewness']:.3f}")
        interpretation.append(f"    Kurtosis: {stats['kurtosis']:.3f}")
        interpretation.append(f"    Coefficient of Variation: {stats['cv']*100:.2f}%")
        
        interpretation.append(f"  Extreme Values:")
        interpretation.append(f"    Maximum: {stats['max_value']:.3f} Sv on {stats['max_date']}")
        interpretation.append(f"    Minimum: {stats['min_value']:.3f} Sv on {stats['min_date']}")
        
        interpretation.append(f"  Percentiles:")
        interpretation.append(f"    5th-95th percentile range: {stats['p05']:.3f} to {stats['p95']:.3f} Sv")
        interpretation.append(f"    1st-99th percentile range: {stats['p01']:.3f} to {stats['p99']:.3f} Sv")
    
    # AMOC Analysis
    interpretation.append("\n\n3. ATLANTIC MERIDIONAL OVERTURNING CIRCULATION (AMOC) - DISTRIBUTION ANALYSIS")
    interpretation.append("-" * 75)
    stats = datasets['amoc']['stats']['amoc']
    
    interpretation.append(f"\nAMOC:")
    interpretation.append(f"  Basic Statistics:")
    interpretation.append(f"    Mean: {stats['mean']:.3f} Sv")
    interpretation.append(f"    Median: {stats['median']:.3f} Sv")
    interpretation.append(f"    Standard Deviation: {stats['std']:.3f} Sv")
    interpretation.append(f"    Range: {stats['range']:.3f} Sv ({stats['min']:.3f} to {stats['max']:.3f})")
    
    interpretation.append(f"  Robust Statistics:")
    interpretation.append(f"    Median Absolute Deviation: {stats['mad']:.3f} Sv")
    interpretation.append(f"    Interquartile Range (IQR): {stats['iqr']:.3f} Sv")
    interpretation.append(f"    10% Trimmed Mean: {stats['trimmed_mean_10']:.3f} Sv")
    interpretation.append(f"    20% Trimmed Mean: {stats['trimmed_mean_20']:.3f} Sv")
    
    interpretation.append(f"  Distribution Shape:")
    interpretation.append(f"    Skewness: {stats['skewness']:.3f}")
    interpretation.append(f"    Kurtosis: {stats['kurtosis']:.3f}")
    interpretation.append(f"    Coefficient of Variation: {stats['cv']*100:.2f}%")
    
    interpretation.append(f"  Extreme Values:")
    interpretation.append(f"    Maximum: {stats['max_value']:.3f} Sv on {stats['max_date']}")
    interpretation.append(f"    Minimum: {stats['min_value']:.3f} Sv on {stats['min_date']}")
    
    interpretation.append(f"  Percentiles:")
    interpretation.append(f"    5th-95th percentile range: {stats['p05']:.3f} to {stats['p95']:.3f} Sv")
    interpretation.append(f"    1st-99th percentile range: {stats['p01']:.3f} to {stats['p99']:.3f} Sv")
    
    # Comparative Analysis
    interpretation.append("\n\n4. COMPARATIVE ANALYSIS AND ROBUST STATISTICS")
    interpretation.append("-" * 50)
    
    # Collect all coefficient of variations for ranking
    cv_data = []
    for dataset_name, dataset in datasets.items():
        for var_name, var_stats in dataset['stats'].items():
            cv = var_stats['cv']*100
            cv_data.append((f"{dataset_name.upper()}_{var_name.upper()}", cv))
    
    cv_data.sort(key=lambda x: x[1], reverse=True)
    
    interpretation.append("\nVariability Ranking (by Coefficient of Variation):")
    for i, (var, cv) in enumerate(cv_data, 1):
        interpretation.append(f"  {i}. {var}: {cv:.2f}%")
    
    # Robust vs Classical Statistics Comparison
    interpretation.append("\nRobust vs Classical Statistics Analysis:")
    interpretation.append("- Trimmed means reduce the influence of extreme values")
    interpretation.append("- MAD is more robust to outliers than standard deviation")
    interpretation.append("- IQR provides outlier-resistant measure of spread")
    interpretation.append("- Skewness and kurtosis indicate distribution asymmetry and tail behavior")
    
    # Distribution Characteristics
    interpretation.append("\nDistribution Characteristics:")
    for dataset_name, dataset in datasets.items():
        for var_name, var_stats in dataset['stats'].items():
            skew = var_stats['skewness']
            kurt = var_stats['kurtosis']
            
            skew_desc = "symmetric" if abs(skew) < 0.5 else ("right-skewed" if skew > 0 else "left-skewed")
            kurt_desc = "normal" if abs(kurt) < 0.5 else ("heavy-tailed" if kurt > 0 else "light-tailed")
            
            interpretation.append(f"  {dataset_name.upper()}_{var_name.upper()}: {skew_desc}, {kurt_desc}")
    
    interpretation.append("\nPhysical Interpretation:")
    interpretation.append("- Distribution shapes reflect the underlying physical processes")
    interpretation.append("- Heavy-tailed distributions suggest occasional extreme events")
    interpretation.append("- Skewed distributions indicate asymmetric forcing mechanisms")
    interpretation.append("- High variability (CV) suggests strong external forcing")
    interpretation.append("- Extreme values often correspond to exceptional climate events")
    
    # Save interpretation
    with open(stats_dir / 'kde_distribution_interpretation.txt', 'w') as f:
        f.write('\n'.join(interpretation))
    
    return interpretation

def main():
    """Main analysis function"""
    print("Starting Ocean Current KDE Distribution Analysis...")
    print("="*60)
    
    # Load and process data
    datasets = load_and_process_data()
    
    # Create and save KDE plot
    print("Creating publication-ready KDE plot...")
    fig = create_kde_plot(datasets)
    fig.savefig(figs_dir / 'ocean_currents_kde_distributions.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    fig.savefig(figs_dir / 'ocean_currents_kde_distributions.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("Plot saved as '../figs/ocean_currents_kde_distributions.png' and '.pdf'")
    
    # Save processed data
    save_processed_data(datasets)
    
    # Generate and save statistical interpretation
    interpretation = generate_statistical_interpretation(datasets)
    
    print("\nKDE Distribution Analysis completed successfully!")
    print(f"Processed data saved to: {processed_dir}")
    print(f"Statistical interpretation saved to: {stats_dir}")
    print(f"Figures saved to: {figs_dir}")
    print("\nFiles created:")
    print("- ../figs/ocean_currents_kde_distributions.png")
    print("- ../figs/ocean_currents_kde_distributions.pdf")
    print("- ../data/processed_data/itf_descriptive_stats.csv")
    print("- ../data/processed_data/agulhas_descriptive_stats.csv")
    print("- ../data/processed_data/amoc_descriptive_stats.csv")
    print("- ../stats/kde_distribution_interpretation.txt")

if __name__ == "__main__":
    main()
