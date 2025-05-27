#!/usr/bin/env python
"""
Ocean Transport Time Series Visualization

Author: Sandy Herho <sandy.herho@email.ucr.edu>
Date: May 26, 2025

This script creates a professional 3-panel time series plot of ocean circulation
transport data including Indonesian Throughflow (ITF), Aghulas Current System,
and Atlantic Meridional Overturning Circulation (AMOC).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyleoclim as pyleo
import os


def load_data():
    """Load ocean transport time series data from CSV files."""
    aghulas = pd.read_csv("../data/raw_data/aghulas_ts.csv")
    itf = pd.read_csv("../data/raw_data/itf_ts.csv")
    amoc = pd.read_csv("../data/raw_data/amoc_ts.csv")
    
    # Convert time columns to datetime
    aghulas['time'] = pd.to_datetime(aghulas['time'])
    itf['time'] = pd.to_datetime(itf['time'])
    amoc['time'] = pd.to_datetime(amoc['time'])
    
    # Print column names for debugging
    print("Aghulas columns:", aghulas.columns.tolist())
    print("ITF columns:", itf.columns.tolist())
    print("AMOC columns:", amoc.columns.tolist())
    
    return aghulas, itf, amoc


def setup_plot_style():
    """Configure matplotlib style settings for professional appearance."""
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'


def create_transport_plot(aghulas, itf, amoc):
    """
    Create a 3-panel time series plot of ocean transport data.
    
    Parameters:
    -----------
    aghulas : pd.DataFrame
        Aghulas current transport data
    itf : pd.DataFrame
        Indonesian Throughflow transport data
    amoc : pd.DataFrame
        AMOC transport data
    """
    # Create directory if it doesn't exist
    os.makedirs("../figs", exist_ok=True)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Setup plot style
    setup_plot_style()
    
    # Define professional colors
    colors_itf = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    colors_aghulas = ['#C73E1D', '#592E83']  # Red, Purple
    color_amoc = '#0B6E4F'  # Green
    
    # Common time limits
    time_min = min(itf['time']) 
    time_max = max(amoc['time']) 
    
    # Plot 1: ITF data
    ax1 = axes[0]
    line1 = ax1.plot(itf['time'], itf["itf_g"], linewidth=2, color=colors_itf[0], alpha=0.9)
    line2 = ax1.plot(itf['time'], itf["itf_s"], linewidth=2, color=colors_itf[1], alpha=0.9)
    line3 = ax1.plot(itf['time'], itf["itf_t"], linewidth=2, color=colors_itf[2], alpha=0.9)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax1.set_xlim([time_min, time_max])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Aghulas data
    ax2 = axes[1]
    line4 = ax2.plot(aghulas['time'], aghulas["aghulas_box"], linewidth=2, color=colors_aghulas[0], alpha=0.9)
    line5 = ax2.plot(aghulas['time'], aghulas["aghulas_jet"], linewidth=2, color=colors_aghulas[1], alpha=0.9)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax2.set_xlim([time_min, time_max])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Plot 3: AMOC data
    ax3 = axes[2]
    line6 = ax3.plot(amoc['time'], amoc["amoc"], linewidth=2, color=color_amoc, alpha=0.9)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    ax3.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax3.set_xlim([time_min, time_max])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Single labels for entire figure
    fig.text(0.04, 0.5, 'Volumetric Transport [Sv]', va='center', rotation='vertical', 
             fontsize=16, fontweight='bold')
    fig.text(0.5, 0.02, 'Time [years C.E.]', ha='center', fontsize=16, fontweight='bold')
    
    # Create combined legend below plots
    lines = [line1[0], line2[0], line3[0], line4[0], line5[0], line6[0]]
    labels = ['ITF-G', 'ITF-S', 'ITF-T', 'Aghulas Box', 'Aghulas Jet', 'AMOC']
    
    # Position legend below the plots
    legend = fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
                       ncol=6, fontsize=14, frameon=True, fancybox=True, shadow=True,
                       edgecolor='black', facecolor='white', framealpha=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.95)
    
    # Save the figure
    plt.savefig("../figs/ocean_transport_timeseries.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # Display the plot
    plt.show()
    
    print("Plot saved successfully to ../figs/ocean_transport_timeseries.png")


if __name__ == "__main__":
    # Load data
    aghulas, itf, amoc = load_data()
    
    # Create and save the plot
    create_transport_plot(aghulas, itf, amoc)
