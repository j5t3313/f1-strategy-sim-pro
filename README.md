# 🏎️ F1 Strategy Simulator Pro 🏎️

A professional Formula 1 race strategy analysis tool that uses advanced tire modeling and Monte Carlo simulation to analyze and compare different pit stop strategies across all 2025 F1 circuits.

## 🏆 Key Features

### 🔧 Advanced Tire Modeling
- **5-Point Curve System**: Professional tire modeling with fully editable parameters
  - Point 1: Compound baseline performance (relative to SOFT)
  - Point 2: Tire warmup characteristics and penalties
  - Point 3: Linear degradation rate per lap
  - Point 4: Performance cliff detection and wear life limits
  - Point 5: Maximum usable stint length before severe penalties
- **Bayesian Tire Models**: MCMC-based statistical models trained on real F1 telemetry data
- **Physics-Based Fallback**: Compound-specific degradation models with exponential penalties

### 🏁 Complete Circuit Coverage
- **All 24 F1 Circuits**: Complete 2025 calendar support with circuit-specific parameters
- **Auto-Scaling**: Strategies automatically adjust for different race distances
- **Circuit-Specific Defaults**: Realistic base pace and pit loss times for each track
- **Fuel Consumption**: Accurate fuel load effects throughout the race

### ⚙️ Flexible Strategy Configuration
- **9 Pre-defined Strategies**: Common 1-stop and 2-stop strategies with realistic compound sequences
- **Custom Strategy Editor**: Modify stint lengths and tire compounds for existing strategies
- **Advanced Strategy Builder**: COMING SOON
- **Custom Tire Allocation**: Define specific tire sets with age and usage history
- **Strategy Validation**: Automatic checking for tire availability and lap count accuracy

### 📊 Comprehensive Analysis & Export
- **Monte Carlo Simulation**: 100-2000 simulations per strategy for robust statistical analysis
- **Performance Distributions**: Box plots, histograms, and cumulative probability charts
- **Head-to-Head Analysis**: Win probability matrices between all strategies
- **Risk Assessment**: Standard deviation and percentile-based uncertainty metrics
- **Multiple Export Formats**:
  - **CSV**: Summary statistics and detailed simulation results
  - **JSON**: Complete structured data including raw results
  - **PDF**: Professional multi-page reports with visualizations and analysis

### 🎯 Professional Interface
- **Tabbed Interface**: Separate Setup and Results areas for clean workflow
- **Progress Tracking**: Real-time simulation progress with status updates
- **Interactive Visualizations**: Plotly charts for tire curve configuration
- **One-Click Reset**: Complete simulator reset to default state

## 🚀 Quick Start

### Online (Recommended)
Visit the live app: https://f1-strategy-sim.streamlit.app

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/j5t3313/f1-strategy-simulator.git
   cd f1-strategy-simulator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run appv9.py
   ```

## 📖 How to Use

### 1. Circuit Selection
Choose from any of the 24 F1 circuits in the 2025 calendar. Each circuit has:
- Specific lap counts and distances
- Realistic base pace and pit loss defaults
- Fuel consumption calculations

### 2. Strategy Selection
Select one or more strategies to compare:
- **Default Strategies**: Pre-configured 1-stop and 2-stop options
- **Custom Editor**: Modify existing strategies with different stint lengths
- **Advanced Builder**: Create new strategies with custom pit stop counts

### 3. Tire Model Configuration
Choose your preferred tire modeling approach:
- **5-Point Curves**: Professional system with 5 editable parameters per compound
- **Bayesian Models**: Real F1 data-based models with uncertainty quantification
- **Default Physics**: Simple compound-specific degradation models

### 4. Analysis Parameters
Configure simulation settings:
- **Base Pace**: Circuit-specific lap time defaults (adjustable)
- **Pit Loss**: Circuit-specific pit stop penalties (adjustable)
- **Simulations**: 100-2000 Monte Carlo runs (more = higher accuracy, longer runtime)
- **Tire Allocation**: Optional custom tire sets with age specifications

### 5. Results Analysis
Comprehensive output includes:
- **Summary Statistics**: Median, mean, standard deviation, percentiles
- **Performance Visualization**: Distribution plots, box plots, cumulative probability
- **Head-to-Head Matrices**: Win rates between all strategy pairs
- **Risk Analysis**: Time penalties and uncertainty metrics
- **Export Options**: CSV, JSON, and PDF formats

## 🔬 Technical Implementation

### Tire Performance Modeling
The simulator employs a hierarchical tire modeling approach:

1. **5-Point Empirical Curves** (Primary):
   - Compound offset calculation relative to SOFT baseline
   - Warmup penalty modeling for initial laps (decreasing effect)
   - Linear degradation throughout stint with compound-specific rates
   - Performance cliff detection at wear life threshold
   - Severe penalties beyond maximum usable stint length

2. **Bayesian Statistical Models** (Fallback):
   - MCMC inference using NUTS sampling for posterior estimation
   - Linear degradation: `laptime ~ Normal(α + β × (stint_lap + tire_age), σ)`
   - Single posterior sample per calculation for stochastic behavior
   - Trained on fuel-corrected lap times from real F1 telemetry

3. **Physics-Based Models** (Final Fallback):
   - Compound-specific degradation rates and baseline offsets
   - Exponential penalties for extended stints beyond typical windows
   - Tire age effects from previous usage

### Simulation Engine
- **Fuel Correction**: `Laptime(corrected) = Laptime - (Remaining_Laps × Fuel_Per_Lap × Weight_Effect)`
- **Stochastic Elements**: Gaussian noise for lap-to-lap variability and base pace variation
- **Monte Carlo Framework**: User-configurable simulation counts with progress tracking
- **Strategy Validation**: Automatic tire availability checking and lap count verification

### Data Management
- **Circuit Database**: Complete 2025 F1 calendar with accurate parameters
- **Model Caching**: Efficient storage and retrieval of Bayesian model parameters
- **Session State**: Persistent configuration across user interactions
- **Export Pipeline**: Structured data preparation for multiple output formats

## 📋 System Requirements

### Core Dependencies
- **Python 3.8+**
- **Streamlit**: Web application framework
- **NumPy/Pandas**: Numerical analysis and data manipulation
- **Matplotlib/Plotly**: Visualization and interactive charts
- **JAX/NumPyro**: High-performance Bayesian modeling (optional)

### Optional Dependencies
- **FastF1**: Official F1 data API for enhanced tire models
- **ReportLab**: Advanced PDF generation capabilities

## ⚠️ Important Disclaimers

### Model Limitations
This simulator is designed for educational and research purposes. Actual F1 race strategies depend on numerous factors not fully captured:

- **Environmental Variables**: Weather conditions, track temperature, surface evolution
- **Human Factors**: Driver performance variations, mistakes, racecraft decisions
- **Race Dynamics**: Safety car deployments, VSC periods, red flag situations
- **Strategic Gaming**: Real-time reactive decisions and competitor interactions
- **Technical Issues**: Reliability problems, damage, setup compromises
- **Regulatory Factors**: Penalty risks, technical regulations, tire pressure monitoring

### Technical Assumptions
- **Linear Degradation**: Models may miss complex thermal and compound behaviors
- **Perfect Execution**: Assumes optimal pit timing without human error
- **Clean Air Racing**: No modeling of dirty air, overtaking difficulty, or track position value
- **Data Currency**: Based on 2024 data which may not reflect 2025 car/tire characteristics
- **Weather Independence**: Dry conditions only (no wet/intermediate tire modeling)

### Usage Guidelines
- Results should not be used for actual racing decisions, betting, or gambling
- Models provide statistical trends rather than exact predictions
- Consider multiple strategies and risk tolerance in decision-making
- Validate results against domain expertise and current F1 knowledge

## 📞 Support & Contributing

### Questions & Issues
For technical questions, feature requests, or bug reports:
- **Email**: jessica.5t3313@gmail.com
- **GitHub Issues**: Create detailed issue reports with reproduction steps

### Contributing
Contributions welcome for:
- Enhanced tire modeling algorithms
- Additional circuit-specific parameters
- User interface improvements
- Documentation updates
- Performance optimizations

## 📄 License

This project is for educational and research purposes. F1 data is accessed through official APIs under their respective terms of service. See LICENSE file for complete terms.

---

**Enjoy professional F1 strategy analysis! 🏁**
