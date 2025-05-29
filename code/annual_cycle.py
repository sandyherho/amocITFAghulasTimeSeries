#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ocean Current Annual Cycle Analysis
Processes ITF, Agulhas, and AMOC time series data to calculate annual cycles
and create publication-ready visualizations.

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
    """Parse time column and extract month information"""
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

def calculate_annual_cycle(df, value_cols):
    """Calculate annual cycle statistics"""
    monthly_stats = {}
    
    for col in value_cols:
        monthly_data = df.groupby('month')[col].agg(['mean', 'std', 'count']).reset_index()
        monthly_data['se'] = monthly_data['std'] / np.sqrt(monthly_data['count'])
        monthly_stats[col] = monthly_data
    
    return monthly_stats

def load_and_process_data():
    """Load and process all datasets"""
    
    # Load ITF data
    print("Loading ITF data...")
    itf_df = pd.read_csv(data_dir / 'itf_ts.csv')
    itf_df = parse_time_column(itf_df)
    itf_stats = calculate_annual_cycle(itf_df, ['itf_g', 'itf_t', 'itf_s'])
    
    # Load Agulhas data
    print("Loading Agulhas data...")
    agulhas_df = pd.read_csv(data_dir / 'aghulas_ts.csv')
    agulhas_df = parse_time_column(agulhas_df)
    agulhas_stats = calculate_annual_cycle(agulhas_df, ['aghulas_box', 'aghulas_jet'])
    
    # Load AMOC data
    print("Loading AMOC data...")
    amoc_df = pd.read_csv(data_dir / 'amoc_ts.csv')
    amoc_df = parse_time_column(amoc_df)
    amoc_stats = calculate_annual_cycle(amoc_df, ['amoc'])
    
    return {
        'itf': {'data': itf_df, 'stats': itf_stats},
        'agulhas': {'data': agulhas_df, 'stats': agulhas_stats},
        'amoc': {'data': amoc_df, 'stats': amoc_stats}
    }

def create_publication_plot(datasets):
    """Create publication-ready 3-row plot"""
    
    # Define beautiful color palettes
    itf_colors = ['#FF6B35', '#4ECDC4', '#45B7D1']  # Orange, Teal, Blue
    agulhas_colors = ['#96CEB4', '#DDA0DD']  # Mint, Plum
    amoc_colors = ['#FF8A80']  # Coral
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    fig.subplots_adjust(hspace=0.15, bottom=0.15)
    
    months = range(1, 13)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Collect all line objects for combined legend
    all_lines = []
    all_labels = []
    
    # Plot 1: ITF (3 variables)
    ax1 = axes[0]
    itf_vars = ['itf_g', 'itf_t', 'itf_s']
    itf_labels = ['ITF-G', 'ITF-T', 'ITF-S']
    
    for i, (var, label, color) in enumerate(zip(itf_vars, itf_labels, itf_colors)):
        stats = datasets['itf']['stats'][var]
        line = ax1.plot(stats['month'], stats['mean'], 'o-', color=color, 
                linewidth=2.5, markersize=6, label=label, alpha=0.8)[0]
        ax1.fill_between(stats['month'], 
                        stats['mean'] - stats['se'], 
                        stats['mean'] + stats['se'], 
                        color=color, alpha=0.2)
        all_lines.append(line)
        all_labels.append(label)
    
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(months)
    ax1.set_xticklabels([])  # Remove x labels from top plot
    
    # Plot 2: Agulhas (2 variables)
    ax2 = axes[1]
    agulhas_vars = ['aghulas_box', 'aghulas_jet']
    agulhas_labels = ['Agulhas Box', 'Agulhas Jet']
    
    for i, (var, label, color) in enumerate(zip(agulhas_vars, agulhas_labels, agulhas_colors)):
        stats = datasets['agulhas']['stats'][var]
        line = ax2.plot(stats['month'], stats['mean'], 'o-', color=color, 
                linewidth=2.5, markersize=6, label=label, alpha=0.8)[0]
        ax2.fill_between(stats['month'], 
                        stats['mean'] - stats['se'], 
                        stats['mean'] + stats['se'], 
                        color=color, alpha=0.2)
        all_lines.append(line)
        all_labels.append(label)
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(months)
    ax2.set_xticklabels([])  # Remove x labels from middle plot
    
    # Plot 3: AMOC (1 variable)
    ax3 = axes[2]
    stats = datasets['amoc']['stats']['amoc']
    line = ax3.plot(stats['month'], stats['mean'], 'o-', color=amoc_colors[0], 
            linewidth=2.5, markersize=6, label='AMOC', alpha=0.8)[0]
    ax3.fill_between(stats['month'], 
                    stats['mean'] - stats['se'], 
                    stats['mean'] + stats['se'], 
                    color=amoc_colors[0], alpha=0.2)
    all_lines.append(line)
    all_labels.append('AMOC')
    
    ax3.set_xlabel('Month', fontsize=20, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(months)
    ax3.set_xticklabels(month_labels, fontsize=12, fontweight='bold')  # Bigger month labels only on bottom plot
    
    # Add shared y-label 
    fig.text(0.0, 0.5, 'Volumetric Transport [Sv]', va='center', rotation='vertical', 
             fontsize=20, fontweight='bold')
    
    # Style improvements
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.tick_params(labelsize=11)
    
    # Create combined legend at the bottom (positioned lower)
    fig.legend(all_lines, all_labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
               ncol=6, frameon=True, fancybox=True, shadow=True, fontsize=11,
               columnspacing=1.5, handletextpad=0.5)
    
    plt.tight_layout()
    return fig

def save_processed_data(datasets):
    """Save processed monthly statistics"""
    
    print("Saving processed data...")
    
    # Save ITF processed data
    itf_processed = pd.DataFrame()
    for var in ['itf_g', 'itf_t', 'itf_s']:
        stats = datasets['itf']['stats'][var]
        for col in ['mean', 'std', 'se']:
            itf_processed[f'{var}_{col}'] = stats[col]
    itf_processed['month'] = range(1, 13)
    itf_processed.to_csv(processed_dir / 'itf_annual_cycle.csv', index=False)
    
    # Save Agulhas processed data
    agulhas_processed = pd.DataFrame()
    for var in ['aghulas_box', 'aghulas_jet']:
        stats = datasets['agulhas']['stats'][var]
        for col in ['mean', 'std', 'se']:
            agulhas_processed[f'{var}_{col}'] = stats[col]
    agulhas_processed['month'] = range(1, 13)
    agulhas_processed.to_csv(processed_dir / 'agulhas_annual_cycle.csv', index=False)
    
    # Save AMOC processed data
    amoc_processed = datasets['amoc']['stats']['amoc'].copy()
    amoc_processed.to_csv(processed_dir / 'amoc_annual_cycle.csv', index=False)

def generate_statistical_interpretation(datasets):
    """Generate comprehensive statistical interpretation"""
    
    print("Generating statistical interpretation...")
    
    interpretation = []
    interpretation.append("OCEAN CURRENT ANNUAL CYCLE ANALYSIS")
    interpretation.append("="*50)
    interpretation.append(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    interpretation.append("")
    
    # ITF Analysis
    interpretation.append("1. INDONESIAN THROUGHFLOW (ITF)")
    interpretation.append("-" * 35)
    for var in ['itf_g', 'itf_t', 'itf_s']:
        stats = datasets['itf']['stats'][var]
        annual_mean = stats['mean'].mean()
        seasonal_amplitude = stats['mean'].max() - stats['mean'].min()
        max_month = stats.loc[stats['mean'].idxmax(), 'month']
        min_month = stats.loc[stats['mean'].idxmin(), 'month']
        
        interpretation.append(f"\n{var.upper()}:")
        interpretation.append(f"  Annual mean: {annual_mean:.3f} Sv")
        interpretation.append(f"  Seasonal amplitude: {seasonal_amplitude:.3f} Sv")
        interpretation.append(f"  Maximum transport: Month {max_month} ({stats['mean'].max():.3f} Sv)")
        interpretation.append(f"  Minimum transport: Month {min_month} ({stats['mean'].min():.3f} Sv)")
        interpretation.append(f"  Coefficient of variation: {(stats['mean'].std()/annual_mean)*100:.2f}%")
    
    # Agulhas Analysis
    interpretation.append("\n\n2. AGULHAS CURRENT SYSTEM")
    interpretation.append("-" * 30)
    for var in ['aghulas_box', 'aghulas_jet']:
        stats = datasets['agulhas']['stats'][var]
        annual_mean = stats['mean'].mean()
        seasonal_amplitude = stats['mean'].max() - stats['mean'].min()
        max_month = stats.loc[stats['mean'].idxmax(), 'month']
        min_month = stats.loc[stats['mean'].idxmin(), 'month']
        
        var_name = "Agulhas Box" if "box" in var else "Agulhas Jet"
        interpretation.append(f"\n{var_name}:")
        interpretation.append(f"  Annual mean: {annual_mean:.3f} Sv")
        interpretation.append(f"  Seasonal amplitude: {seasonal_amplitude:.3f} Sv")
        interpretation.append(f"  Maximum transport: Month {max_month} ({stats['mean'].max():.3f} Sv)")
        interpretation.append(f"  Minimum transport: Month {min_month} ({stats['mean'].min():.3f} Sv)")
        interpretation.append(f"  Coefficient of variation: {(stats['mean'].std()/annual_mean)*100:.2f}%")
    
    # AMOC Analysis
    interpretation.append("\n\n3. ATLANTIC MERIDIONAL OVERTURNING CIRCULATION (AMOC)")
    interpretation.append("-" * 55)
    stats = datasets['amoc']['stats']['amoc']
    annual_mean = stats['mean'].mean()
    seasonal_amplitude = stats['mean'].max() - stats['mean'].min()
    max_month = stats.loc[stats['mean'].idxmax(), 'month']
    min_month = stats.loc[stats['mean'].idxmin(), 'month']
    
    interpretation.append(f"\nAMOC:")
    interpretation.append(f"  Annual mean: {annual_mean:.3f} Sv")
    interpretation.append(f"  Seasonal amplitude: {seasonal_amplitude:.3f} Sv")
    interpretation.append(f"  Maximum transport: Month {max_month} ({stats['mean'].max():.3f} Sv)")
    interpretation.append(f"  Minimum transport: Month {min_month} ({stats['mean'].min():.3f} Sv)")
    interpretation.append(f"  Coefficient of variation: {(stats['mean'].std()/annual_mean)*100:.2f}%")
    
    # Summary and interpretation
    interpretation.append("\n\n4. SUMMARY AND PHYSICAL INTERPRETATION")
    interpretation.append("-" * 40)
    interpretation.append("\nSeasonal Variability Ranking (by coefficient of variation):")
    
    # Calculate CV for all variables
    cv_data = []
    for dataset_name, dataset in datasets.items():
        for var_name, var_stats in dataset['stats'].items():
            annual_mean = var_stats['mean'].mean()
            cv = (var_stats['mean'].std()/annual_mean)*100
            cv_data.append((f"{dataset_name.upper()}_{var_name.upper()}", cv))
    
    cv_data.sort(key=lambda x: x[1], reverse=True)
    for i, (var, cv) in enumerate(cv_data, 1):
        interpretation.append(f"  {i}. {var}: {cv:.2f}%")
    
    interpretation.append("\nPhysical Interpretation:")
    interpretation.append("- Seasonal cycles reflect monsoon forcing, thermal stratification changes,")
    interpretation.append("  and wind stress patterns in each respective ocean basin.")
    interpretation.append("- Higher coefficients of variation indicate stronger seasonal modulation")
    interpretation.append("  relative to the annual mean transport.")
    interpretation.append("- Peak transport months often correspond to seasonal wind maxima")
    interpretation.append("  and thermal forcing patterns in each region.")
    
    # Save interpretation
    with open(stats_dir / 'annual_cycle_interpretation.txt', 'w') as f:
        f.write('\n'.join(interpretation))
    
    return interpretation

def main():
    """Main analysis function"""
    print("Starting Ocean Current Annual Cycle Analysis...")
    print("="*50)
    
    # Load and process data
    datasets = load_and_process_data()
    
    # Create and save plot
    print("Creating publication-ready plot...")
    fig = create_publication_plot(datasets)
    fig.savefig(figs_dir / 'ocean_currents_annual_cycle.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    fig.savefig(figs_dir / 'ocean_currents_annual_cycle.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("Plot saved as '../figs/ocean_currents_annual_cycle.png' and '.pdf'")
    
    # Save processed data
    save_processed_data(datasets)
    
    # Generate and save statistical interpretation
    interpretation = generate_statistical_interpretation(datasets)
    
    print("\nAnalysis completed successfully!")
    print(f"Processed data saved to: {processed_dir}")
    print(f"Statistical interpretation saved to: {stats_dir}")
    print(f"Figures saved to: {figs_dir}")
    print("\nFiles created:")
    print("- ../figs/ocean_currents_annual_cycle.png")
    print("- ../figs/ocean_currents_annual_cycle.pdf")
    print("- ../data/processed_data/itf_annual_cycle.csv")
    print("- ../data/processed_data/agulhas_annual_cycle.csv")
    print("- ../data/processed_data/amoc_annual_cycle.csv")
    print("- ../stats/annual_cycle_interpretation.txt")

if __name__ == "__main__":
    main()
