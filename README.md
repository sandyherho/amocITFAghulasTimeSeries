# Supplementary Materials: Strengthening ITF and Weakening AMOC: Statistical Analysis of Ocean Transport Variability

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2023b-orange)](https://www.mathworks.com/products/matlab.html)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](http://www.wtfpl.net/about/)
[![DOI](https://zenodo.org/badge/990993766.svg)](https://doi.org/10.5281/zenodo.15567489)

## Research Overview

This repository contains the complete analytical framework for investigating multi-decadal trends and causal relationships in major ocean circulation systems. Our study employs robust statistical methods to quantify variability in the Indonesian Throughflow (ITF), Agulhas Current system, and Atlantic Meridional Overturning Circulation (AMOC) spanning 1984-2023.

### Key Scientific Findings

- **ITF Strengthening**: Geostrophic and salinity components show statistically significant increases of 0.79 and 0.28 Sv decade⁻¹ respectively (p < 0.05)
- **AMOC Weakening**: Robust decline of -1.61 Sv decade⁻¹ (p < 0.0001)
- **ITF-Agulhas Coupling**: Four dominant causal pathways identified with lag times of 0-18 months
- **Decoupled Systems**: No direct statistical connections between Indo-Pacific and Atlantic sectors

## Repository Structure

```
amocITFAghulasTimeSeries/
│
├── code/                      # Analysis scripts
│   ├── extract.py            # Raw data extraction and preprocessing
│   ├── merged_data.py        # Dataset alignment and merging
│   ├── stats_desc.py         # Descriptive statistics and distributions
│   ├── annual_cycle.py       # Seasonal variability analysis
│   ├── trend_anal.py         # Theil-Sen trend estimation
│   ├── causality.py          # Multi-method causality analysis
│   ├── ts_plot.py            # Time series visualization
│   ├── map.py                # Geographic visualization (PyGMT)
│   └── wtc_itf_agulhas.m     # Wavelet coherence analysis (MATLAB)
│
├── data/
│   ├── raw_data/             # Original datasets 
│   └── processed_data/       # Processed analysis outputs
│
├── figs/                     # Publication-ready figures
├── stats/                    # Statistical interpretations
└── README.md                 # This file
```

## Installation & Requirements

### Python Environment Setup

#### Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv ocean_transport_env
source ocean_transport_env/bin/activate  # On Windows: ocean_transport_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install scientific computing stack
pip install numpy pandas scipy scikit-learn
pip install matplotlib seaborn
pip install xarray netCDF4
pip install pygmt
```

#### System-wide Installation (Alternative)
```bash
# Install dependencies with pip
pip3 install --user numpy pandas scipy scikit-learn matplotlib seaborn
pip3 install --user xarray netCDF4 pyleoclim pygmt
```

### MATLAB Requirements
- MATLAB R2023b or later
- Wavelet Toolbox
- Signal Processing Toolbox (recommended)

### Verifying Installation
```python
# Test Python installation
python -c "import numpy, pandas, scipy, sklearn, matplotlib, seaborn, xarray, netCDF4; print('All packages imported successfully')"
```

## Methodological Framework

### 1. Data Processing Pipeline
```bash
# Extract raw ocean transport data
python code/extract.py

# Merge and align time series
python code/merged_data.py
```

### 2. Statistical Analysis Suite

#### Descriptive Statistics & Distributions
```bash
python code/stats_desc.py
```
Generates comprehensive distributional analysis including:
- Robust statistics (MAD, trimmed means)
- Distribution shape parameters (skewness, kurtosis)
- Extreme value identification

#### Seasonal Cycle Analysis
```bash
python code/annual_cycle.py
```
Quantifies:
- Monthly climatological means
- Seasonal amplitude and phase
- Coefficient of variation

#### Trend Detection
```bash
python code/trend_anal.py
```
Implements:
- Theil-Sen robust regression
- Mann-Kendall significance testing
- Decadal trend estimation

### 3. Causality Analysis Framework
```bash
python code/causality.py
```

Multi-method approach combining:
- **Maximum Cross-Correlation (MCC)**: Linear lagged relationships
- **Convergent Cross Mapping (CCM)**: Nonlinear dynamical coupling
- **Transfer Entropy (TE)**: Information flow quantification
- **Block Bootstrap**: Significance testing preserving autocorrelation

### 4. Wavelet Coherence Analysis
```matlab
% In MATLAB
cd code
wtc_itf_agulhas
```

Time-frequency decomposition revealing:
- Scale-dependent coupling patterns
- Phase relationships
- Episodic coherence during climate events

## Key Hypotheses & Results

### Primary Hypotheses
1. **H₁**: Contemporary ocean circulation exhibits basin-specific trends driven by regional forcing
   - **Result**: Confirmed - ITF strengthening linked to Maritime Continent rainfall; AMOC weakening to North Atlantic convection

2. **H₂**: Statistical teleconnections exist between Indo-Pacific and Atlantic circulation systems
   - **Result**: Partially supported - Strong ITF-Agulhas coupling detected; no direct ITF/Agulhas-AMOC connections

3. **H₃**: Ocean gateway dynamics respond coherently to climate forcing on interannual timescales
   - **Result**: Confirmed - Annual-scale coherence (0.87-1.30 years) dominates ITF-Agulhas relationships

### Statistical Significance Summary
| Pathway | MCC (r) | Lag (months) | CCM (ρ) | Consensus |
|---------|---------|--------------|---------|-----------|
| ITF-G → Agulhas Box | -0.280** | -18 | 0.237* | 2/3 |
| ITF-G → Agulhas Jet | 0.264** | -1 | 0.302** | 2/3 |
| ITF-T → Agulhas Box | 0.223* | 0 | 0.241** | 2/3 |
| ITF-S → Agulhas Box | 0.265* | -12 | 0.193* | 2/3 |

\* p < 0.05, ** p < 0.01

## Data Sources

- **ITF Data**: [doi:10.12157/IOCAS.20221214.001](http://doi.org/10.12157/IOCAS.20221214.001)
- **AMOC Data**: [doi:10.48670/moi-00232](https://doi.org/10.48670/moi-00232)
- **Agulhas Data**: [Beal-Agulhas Dataset](https://beal-agulhas.earth.miami.edu/data-and-products/index.html)

## Reproducing Results

### Complete Analysis Pipeline
```bash
# Activate virtual environment (if using)
source ocean_transport_env/bin/activate

# 1. Data preparation
python code/extract.py
python code/merged_data.py

# 2. Statistical analyses
python code/stats_desc.py
python code/annual_cycle.py
python code/trend_anal.py
python code/causality.py

# 3. Visualization
python code/ts_plot.py
python code/map.py

# 4. Wavelet analysis (in MATLAB)
matlab -batch "cd code; wtc_itf_agulhas"
```

### Troubleshooting Common Issues

#### PyGMT Installation
If PyGMT installation fails, install GMT first:
```bash
# On Ubuntu/Debian
sudo apt-get install gmt gmt-dcw gmt-gshhg

# On macOS with Homebrew
brew install gmt
```

#### Memory Issues
For large datasets, increase Python's memory allocation:
```bash
export PYTHONUNBUFFERED=1
ulimit -s unlimited
```

## Scientific Contributions

1. **Quantitative Evidence**: First comprehensive statistical characterization of simultaneous ITF strengthening and AMOC weakening
2. **Methodological Innovation**: Multi-method causality framework combining linear and nonlinear approaches
3. **Physical Insights**: Demonstrates regional forcing dominance over global-scale coupling in contemporary ocean circulation

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{herho2025ocean,
  title={{Strengthening ITF and Weakening AMOC: Time Series Evidence of 
         Trends and Causal Pathways to Agulhas Variability}},
  author={Herho, S. H. S. and Anwar, I. P. and Herho, K. E. P. 
          and Cahyarini, S. Y.},
  journal={xxxxxx},
  year={202x},
  doi=xx.xxxx/xxxxx}
}
```

## Acknowledgments

This research was supported by the Dean's Distinguished Fellowship from the College of Natural and Agricultural Sciences (CNAS) at the University of California, Riverside (2023).

## License

This project is released under the [WTFPL](http://www.wtfpl.net/) - Do What The F*ck You Want To Public License.
