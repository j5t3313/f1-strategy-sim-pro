import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import os
import pickle
from pathlib import Path
import json
import io
from datetime import datetime
warnings.filterwarnings('ignore')

# Import the empirical analyzer
try:
    from tireModelv2 import EmpiricalTireAnalyzer
    EMPIRICAL_AVAILABLE = True
except ImportError:
    EMPIRICAL_AVAILABLE = False
    print("EmpiricalTireAnalyzer not available - quality dashboard will be disabled")

# set page config
st.set_page_config(
    page_title="F1 Strategy Simulator Pro",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 
st.markdown("""
<style>
/* Make tab labels larger */
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.2rem;
    font-weight: 600;
}

/* Make tab panels larger */
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem;
}

/* Make tab bar taller */
.stTabs [data-baseweb="tab-list"] {
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
}

/* Make headers smaller */
h1 {
    font-size: 2rem !important;
}

h2 {
    font-size: 1.5rem !important;
}

h3 {
    font-size: 1.2rem !important;
}

/* Progress bar  */
.stProgress > div > div > div > div {
    background-color: #ff6b6b;
}

/* Smaller run button */
.small-button {
    font-size: 0.9rem !important;
    padding: 0.3rem 1rem !important;
    height: auto !important;
}
</style>
""", unsafe_allow_html=True)
class F1FivePointTireEditor:
    """
    5-point tire curve editor
    All curves based off the SOFT compound as reference
    """
    
    def __init__(self):
        # Initialize session state for tire curves if not exists
        if 'f1_tire_curves' not in st.session_state:
            st.session_state.f1_tire_curves = self._get_default_curves()
    
    def _get_default_curves(self):
        """Default 5-point curves based on typical values"""
        return {
            'SOFT': {
                'compound_offset': 0.0,      # Point 1: Reference compound
                'warmup_effect': 0.8,        # Point 2: Tire warmup penalty (seconds)
                'linear_degradation': 0.035, # Point 3: Linear deg rate (s/lap)
                'wear_life_start': 18,       # Point 4: When high deg starts (lap)
                'max_usable_laps': 25        # Point 5: Can't use beyond this
            },
            'MEDIUM': {
                'compound_offset': 0.3,      # Point 1: 0.3s slower than SOFT
                'warmup_effect': 0.6,        # Point 2: Less warmup penalty
                'linear_degradation': 0.020, # Point 3: Slower degradation
                'wear_life_start': 28,       # Point 4: Longer before high deg
                'max_usable_laps': 40        # Point 5: Can run much longer
            },
            'HARD': {
                'compound_offset': 0.8,      # Point 1: 0.8s slower than SOFT
                'warmup_effect': 1.0,        # Point 2: Most warmup penalty
                'linear_degradation': 0.012, # Point 3: Slowest degradation
                'wear_life_start': 35,       # Point 4: Very long before high deg
                'max_usable_laps': 50        # Point 5: Can run much longer
            }
        }
    
    def render_tire_curve_editor(self):
        """Render the 5-point tire curve editor interface"""
        st.subheader("🔧 5-Point Tire Curve Editor 🔧")
        
        st.markdown("""
        **5-Point Curve Tire Modeling**: Adjust the 5 key parameters that define tire performance.
        All curves are relative to the SOFT compound as the reference.
        """)
        
        # Get current curves
        curves = st.session_state.f1_tire_curves
        
        # Create tabs for each compound
        soft_tab, medium_tab, hard_tab = st.tabs(['SOFT (Reference)', 'MEDIUM', 'HARD'])
        
        with soft_tab:
            self._render_compound_editor('SOFT', curves['SOFT'], is_reference=True)
        
        with medium_tab:
            self._render_compound_editor('MEDIUM', curves['MEDIUM'])
        
        with hard_tab:
            self._render_compound_editor('HARD', curves['HARD'])
        
        # Controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Reset to Defaults"):
                st.session_state.f1_tire_curves = self._get_default_curves()
                st.rerun()
        
        with col2:
            if st.button("Export Curves"):
                self._export_curves()
        
        # Visualization
        st.subheader("📊 Tire Performance Visualization 📊")
        self._render_curve_visualization()
    
    def _render_compound_editor(self, compound, curve_data, is_reference=False):
        """Render editor for a specific compound's 5-point curve"""
        
        if is_reference:
            st.info("SOFT is the reference compound. All other compounds are relative to SOFT.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Performance Points:**")
            
            # Point 1: Compound Offset
            if is_reference:
                st.metric("Point 1: Compound Offset", "0.0s (Reference)")
                curve_data['compound_offset'] = 0.0
            else:
                offset = st.slider(
                    f"Point 1: Compound Offset (s)",
                    0.0, 2.0, 
                    float(curve_data['compound_offset']), 
                    0.1,
                    key=f"{compound}_offset",
                    help="How much slower than SOFT compound"
                )
                # Only update if the value changed
                if offset != curve_data['compound_offset']:
                    curve_data['compound_offset'] = offset
            
            # Point 2: Warmup Effect
            warmup = st.slider(
                f"Point 2: Warmup Effect (s)",
                0.0, 2.0, 
                float(curve_data['warmup_effect']), 
                0.1,
                key=f"{compound}_warmup",
                help="Time penalty when tires are cold"
            )
            if warmup != curve_data['warmup_effect']:
                curve_data['warmup_effect'] = warmup
            
            # Point 3: Linear Degradation
            linear_deg = st.slider(
                f"Point 3: Linear Degradation (s/lap)",
                0.005, 0.100, 
                float(curve_data['linear_degradation']), 
                0.005,
                key=f"{compound}_linear",
                help="Steady degradation rate per lap"
            )
            if linear_deg != curve_data['linear_degradation']:
                curve_data['linear_degradation'] = linear_deg
        
        with col2:
            st.write("**Durability Points:**")
            
            # Point 4: Wear Life Start
            wear_start = st.slider(
                f"Point 4: Wear Life Start (lap)",
                5, 60, 
                int(curve_data['wear_life_start']),
                key=f"{compound}_wear_start",
                help="When high degradation phase begins"
            )
            if wear_start != curve_data['wear_life_start']:
                curve_data['wear_life_start'] = wear_start
            
            # Point 5: Max Usable Laps
            max_laps = st.slider(
                f"Point 5: Max Usable Laps",
                10, 70, 
                int(curve_data['max_usable_laps']),
                key=f"{compound}_max_laps",
                help="Cannot be used beyond this point"
            )
            if max_laps != curve_data['max_usable_laps']:
                curve_data['max_usable_laps'] = max_laps
        
        # Show current curve summary
        st.write("**Curve Summary:**")
        summary_cols = st.columns(5)
        
        with summary_cols[0]:
            st.metric("Offset", f"+{curve_data['compound_offset']:.1f}s")
        with summary_cols[1]:
            st.metric("Warmup", f"{curve_data['warmup_effect']:.1f}s")
        with summary_cols[2]:
            st.metric("Deg Rate", f"{curve_data['linear_degradation']:.3f}s/lap")
        with summary_cols[3]:
            st.metric("Wear Life", f"Lap {curve_data['wear_life_start']}")
        with summary_cols[4]:
            st.metric("Max Stint", f"{curve_data['max_usable_laps']} laps")
    
    def calculate_f1_tire_performance(self, compound, stint_lap, base_laptime):
        """Calculate tire performance using 5-point curve system"""
        curves = st.session_state.f1_tire_curves
        curve = curves[compound]
        
        # base laptime
        laptime = base_laptime
        
        # Point 1: Compound offset (relative to SOFT)
        laptime += curve['compound_offset']
        
        # Point 2: Warmup effect (first 3 laps)
        if stint_lap <= 3:
            warmup_factor = (4 - stint_lap) / 3  # Decreases from 1.0 to 0.0
            laptime += curve['warmup_effect'] * warmup_factor
        
        # Point 3: Linear deg
        laptime += curve['linear_degradation'] * stint_lap
        
        # Point 4: Wear life (high deg phase)
        if stint_lap > curve['wear_life_start']:
            wear_laps = stint_lap - curve['wear_life_start']
            # Exponential increase in deg
            high_deg = 0.1 * (wear_laps ** 1.8)
            laptime += high_deg
        
        # Point 5: Beyond max usable (severe penalty)
        if stint_lap > curve['max_usable_laps']:
            danger_laps = stint_lap - curve['max_usable_laps']
            laptime += 5.0 + danger_laps * 2.0  
        
        return laptime
    
    def _render_curve_visualization(self):
        """Render interactive tire curve visualization"""
        
        # Controls for visualization
        col1, col2, col3 = st.columns(3)
        
        with col1:
            base_pace = st.number_input("Base Pace (s)", 60.0, 120.0, 80.0, 0.5)
        with col2:
            max_laps = st.number_input("Max Laps to Show", 10, 80, 50, 5)
        with col3:
            show_zones = st.checkbox("Show Performance Zones", True)
        
        # plot
        fig = go.Figure()
        
        colors = {'SOFT': '#FF4444', 'MEDIUM': '#FFB347', 'HARD': '#808080'}
        
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            laps = list(range(1, max_laps + 1))
            laptimes = []
            
            for lap in laps:
                laptime = self.calculate_f1_tire_performance(compound, lap, base_pace)
                laptimes.append(laptime)
            
            # Main performance curve
            fig.add_trace(
                go.Scatter(
                    x=laps, 
                    y=laptimes,
                    mode='lines+markers',
                    name=compound,
                    line=dict(color=colors[compound], width=3),
                    hovertemplate=f'{compound}<br>Lap: %{{x}}<br>Time: %{{y:.3f}}s<extra></extra>'
                )
            )
        
        fig.update_layout(
            title="5-Point Tire Performance Curves",
            xaxis_title="Stint Lap",
            yaxis_title="Lap Time (s)",
            height=600,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance comparison table
        self._render_performance_table(base_pace)
    
    def _render_performance_table(self, base_pace):
        """Render performance comparison table"""
        st.write("**Performance Comparison:**")
        
        comparison_laps = [1, 5, 10, 15, 20, 25, 30]
        
        data = []
        for lap in comparison_laps:
            row = {'Lap': lap}
            for compound in ['SOFT', 'MEDIUM', 'HARD']:
                laptime = self.calculate_f1_tire_performance(compound, lap, base_pace)
                row[compound] = f"{laptime:.3f}s"
            data.append(row)
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
    
    def _export_curves(self):
        """Export current tire curves"""
        curves = st.session_state.f1_tire_curves
        
        # Create export data
        export_data = {
            'f1_tire_curves': curves,
            'export_timestamp': pd.Timestamp.now().isoformat(),
            'curve_type': '5_point_f1_system'
        }
        
        # Convert to JSON for download
        json_str = json.dumps(export_data, indent=2)
        
        st.download_button(
            label="Download 5-Point Tire Curves",
            data=json_str,
            file_name="f1_tire_curves.json",
            mime="application/json"
        )
def render_custom_strategy_editor(sim, circuit, selected_strategies):
    """Render custom strategy editor with lap customization"""
    
    st.header("Custom Strategy Configuration")
    
    use_custom_strategies = st.checkbox("Enable Custom Strategy Editor")
    
    if use_custom_strategies:
        if 'custom_strategies' not in st.session_state:
            st.session_state.custom_strategies = {}
        
        circuit_laps = sim.circuits[circuit]['laps']
        
        for strategy_name in selected_strategies:
            st.subheader(f"Edit {strategy_name}")
            
            # Initialize custom strategy if not exists with circuit-scaled values
            if strategy_name not in st.session_state.custom_strategies:
                base_strategy = ALL_STRATEGIES[strategy_name]
                
                # Apply circuit scaling to base strategy
                total_original_laps = sum(stint['laps'] for stint in base_strategy)
                scale_factor = circuit_laps / total_original_laps
                
                scaled_strategy = []
                remaining_laps = circuit_laps
                
                for i, stint in enumerate(base_strategy):
                    if i == len(base_strategy) - 1:
                        laps = remaining_laps
                    else:
                        scaled_laps = stint['laps'] * scale_factor
                        laps = max(1, round(scaled_laps))
                        laps = min(laps, remaining_laps - (len(base_strategy) - i - 1))
                        remaining_laps -= laps
                    
                    scaled_strategy.append({'compound': stint['compound'], 'laps': laps})
                
                st.session_state.custom_strategies[strategy_name] = scaled_strategy
            
            custom_strategy = st.session_state.custom_strategies[strategy_name]
            
            cols = st.columns(len(custom_strategy))
            
            for i, (col, stint) in enumerate(zip(cols, custom_strategy)):
                with col:
                    st.write(f"**Stint {i+1}**")
                    
                    # Compound selector
                    compound = st.selectbox(
                        "Compound",
                        ["SOFT", "MEDIUM", "HARD"],
                        index=["SOFT", "MEDIUM", "HARD"].index(stint['compound']),
                        key=f"{strategy_name}_stint_{i}_compound"
                    )
                    
                    # Lap count selector
                    laps = st.number_input(
                        "Laps",
                        min_value=1,
                        max_value=circuit_laps,
                        value=stint['laps'],
                        key=f"{strategy_name}_stint_{i}_laps"
                    )
                    
                    # Update the custom strategy
                    custom_strategy[i] = {'compound': compound, 'laps': laps}
            
            # Show total laps
            total_laps = sum(stint['laps'] for stint in custom_strategy)
            
            if total_laps != circuit_laps:
                st.warning(f"Total laps: {total_laps} (Circuit has {circuit_laps} laps)")
            else:
                st.success(f"Total laps: {total_laps} ✓")
    
    return use_custom_strategies

def render_advanced_strategy_editor(sim, circuit, selected_strategies):
    """Render advanced strategy editor with add/remove stint functionality"""
    
    st.header("Advanced Strategy Builder")
    
    use_advanced_strategies = st.checkbox("Enable Advanced Strategy Builder")
    
    if use_advanced_strategies:
        if 'advanced_strategies' not in st.session_state:
            st.session_state.advanced_strategies = {}
        
        circuit_laps = sim.circuits[circuit]['laps']
        
        for strategy_name in selected_strategies:
            with st.expander(f"Edit {strategy_name}", expanded=True):
                
                # Initialize if not exists with circuit-scaled values
                if strategy_name not in st.session_state.advanced_strategies:
                    base_strategy = ALL_STRATEGIES[strategy_name]
                    
                    # Apply circuit scaling to base strategy
                    total_original_laps = sum(stint['laps'] for stint in base_strategy)
                    scale_factor = circuit_laps / total_original_laps
                    
                    scaled_strategy = []
                    remaining_laps = circuit_laps
                    
                    for i, stint in enumerate(base_strategy):
                        if i == len(base_strategy) - 1:
                            laps = remaining_laps
                        else:
                            scaled_laps = stint['laps'] * scale_factor
                            laps = max(1, round(scaled_laps))
                            laps = min(laps, remaining_laps - (len(base_strategy) - i - 1))
                            remaining_laps -= laps
                        
                        scaled_strategy.append({'compound': stint['compound'], 'laps': laps})
                    
                    st.session_state.advanced_strategies[strategy_name] = scaled_strategy
                
                strategy = st.session_state.advanced_strategies[strategy_name]
                
                # Add/Remove stint buttons
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    if st.button(f"Add Stint", key=f"add_{strategy_name}"):
                        strategy.append({'compound': 'MEDIUM', 'laps': 10})
                        st.rerun()
                
                with col2:
                    if len(strategy) > 1 and st.button(f"Remove Stint", key=f"remove_{strategy_name}"):
                        strategy.pop()
                        st.rerun()
                
                # Render each stint
                for i, stint in enumerate(strategy):
                    stint_col1, stint_col2, stint_col3 = st.columns([2, 2, 1])
                    
                    with stint_col1:
                        compound = st.selectbox(
                            f"Stint {i+1} Compound",
                            ["SOFT", "MEDIUM", "HARD"],
                            index=["SOFT", "MEDIUM", "HARD"].index(stint['compound']),
                            key=f"adv_{strategy_name}_stint_{i}_compound"
                        )
                        stint['compound'] = compound
                    
                    with stint_col2:
                        laps = st.number_input(
                            f"Stint {i+1} Laps",
                            min_value=1,
                            max_value=circuit_laps,
                            value=stint['laps'],
                            key=f"adv_{strategy_name}_stint_{i}_laps"
                        )
                        stint['laps'] = laps
                    
                    with stint_col3:
                        if i == 0:
                            st.metric("Pit Stops", len(strategy) - 1)
                
                # Auto-balance button
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Auto-Balance Laps", key=f"balance_{strategy_name}"):
                        laps_per_stint = circuit_laps // len(strategy)
                        remainder = circuit_laps % len(strategy)
                        
                        for i, stint in enumerate(strategy):
                            stint['laps'] = laps_per_stint + (1 if i < remainder else 0)
                        st.rerun()
                
                # Strategy summary with dynamic naming
                total_laps = sum(stint['laps'] for stint in strategy)
                strategy_str = " → ".join([f"{stint['laps']}{stint['compound'][0]}" for stint in strategy])
                
                # Generate dynamic strategy name based on number of stints
                num_stops = len(strategy) - 1
                compounds_str = "-".join([stint['compound'][0] for stint in strategy])
                dynamic_name = f"{num_stops}-stop ({compounds_str})"
                
                if total_laps == circuit_laps:
                    st.success(f"**{dynamic_name}:** {strategy_str} (Total: {total_laps} laps)")
                else:
                    st.error(f"**{dynamic_name}:** {strategy_str} (Total: {total_laps} laps - Need {circuit_laps})")
    
    return use_advanced_strategies

# predefined strategies
ALL_STRATEGIES = {
    "1-stop (M-H)": [{"compound": "MEDIUM", "laps": 20}, {"compound": "HARD", "laps": 24}],
    "1-stop (S-H)": [{"compound": "SOFT", "laps": 15}, {"compound": "HARD", "laps": 29}],
    "2-stop (M-M-H)": [{"compound": "MEDIUM", "laps": 15}, {"compound": "MEDIUM", "laps": 15}, {"compound": "HARD", "laps": 14}],
    "2-stop (M-M-S)": [{"compound": "MEDIUM", "laps": 16}, {"compound": "MEDIUM", "laps": 16}, {"compound": "SOFT", "laps": 12}],
    "2-stop (S-M-H)": [{"compound": "SOFT", "laps": 12}, {"compound": "MEDIUM", "laps": 16}, {"compound": "HARD", "laps": 16}],
    "2-stop (H-M-S)": [{"compound": "HARD", "laps": 16}, {"compound": "MEDIUM", "laps": 16}, {"compound": "SOFT", "laps": 12}],
    "1-stop (H-M)": [{"compound": "HARD", "laps": 25}, {"compound": "MEDIUM", "laps": 19}],
    "1-stop (H-S)": [{"compound": "HARD", "laps": 31}, {"compound": "SOFT", "laps": 13}],
    "2-stop (M-H-M)": [{"compound": "MEDIUM", "laps": 12}, {"compound": "HARD", "laps": 18}, {"compound": "MEDIUM", "laps": 14}]
}
class F1StrategySimulator:
    def __init__(self, models_dir="prebuilt_models_v2"):
        self.models_dir = Path(models_dir)
        
        # 2025 F1 Calendar
        self.circuit_data = [
            ('Australia', {'laps': 58, 'distance_km': 5.278, 'gp_name': 'Australian Grand Prix'}),
            ('China', {'laps': 56, 'distance_km': 5.451, 'gp_name': 'Chinese Grand Prix'}),
            ('Japan', {'laps': 53, 'distance_km': 5.807, 'gp_name': 'Japanese Grand Prix'}),
            ('Bahrain', {'laps': 57, 'distance_km': 5.412, 'gp_name': 'Bahrain Grand Prix'}),
            ('Saudi Arabia', {'laps': 50, 'distance_km': 6.174, 'gp_name': 'Saudi Arabian Grand Prix'}),
            ('Miami', {'laps': 57, 'distance_km': 5.41, 'gp_name': 'Miami Grand Prix'}),
            ('Imola', {'laps': 63, 'distance_km': 4.909, 'gp_name': 'Emilia Romagna Grand Prix'}),
            ('Monaco', {'laps': 78, 'distance_km': 3.337, 'gp_name': 'Monaco Grand Prix'}),
            ('Spain', {'laps': 66, 'distance_km': 4.655, 'gp_name': 'Spanish Grand Prix'}),
            ('Canada', {'laps': 70, 'distance_km': 4.361, 'gp_name': 'Canadian Grand Prix'}),
            ('Austria', {'laps': 71, 'distance_km': 4.318, 'gp_name': 'Austrian Grand Prix'}),
            ('Britain', {'laps': 52, 'distance_km': 5.891, 'gp_name': 'British Grand Prix'}),
            ('Belgium', {'laps': 44, 'distance_km': 7.004, 'gp_name': 'Belgian Grand Prix'}),
            ('Hungary', {'laps': 70, 'distance_km': 4.381, 'gp_name': 'Hungarian Grand Prix'}),
            ('Netherlands', {'laps': 72, 'distance_km': 4.259, 'gp_name': 'Dutch Grand Prix'}),
            ('Italy', {'laps': 53, 'distance_km': 5.793, 'gp_name': 'Italian Grand Prix'}),
            ('Azerbaijan', {'laps': 51, 'distance_km': 6.003, 'gp_name': 'Azerbaijan Grand Prix'}),
            ('Singapore', {'laps': 62, 'distance_km': 4.940, 'gp_name': 'Singapore Grand Prix'}),
            ('United States', {'laps': 56, 'distance_km': 5.513, 'gp_name': 'United States Grand Prix'}),
            ('Mexico', {'laps': 71, 'distance_km': 4.304, 'gp_name': 'Mexico City Grand Prix'}),
            ('Brazil', {'laps': 71, 'distance_km': 4.309, 'gp_name': 'São Paulo Grand Prix'}),
            ('Las Vegas', {'laps': 50, 'distance_km': 6.201, 'gp_name': 'Las Vegas Grand Prix'}),
            ('Qatar', {'laps': 57, 'distance_km': 5.380, 'gp_name': 'Qatar Grand Prix'}),
            ('Abu Dhabi', {'laps': 58, 'distance_km': 5.281, 'gp_name': 'Abu Dhabi Grand Prix'})
        ]
        
        # convert to dict for easy lookup
        self.circuits = {name: data for name, data in self.circuit_data}
        
        # Bayesian models disabled in this version - using 5-point curves or default physics only
        
        # Initialize tire editor
        if 'f1_tire_editor' not in st.session_state:
            st.session_state.f1_tire_editor = F1FivePointTireEditor()
    
    def calculate_fuel_consumption(self, circuit):
        """Calculate fuel consumption per lap"""
        laps = self.circuits[circuit]['laps']
        return 105.0 / laps
        
    def calculate_fuel_corrected_laptime(self, raw_laptime, lap_number, total_laps, 
                                       fuel_consumption_per_lap, weight_effect=0.03):
        """Fuel corrected laptime calculation"""
        remaining_laps = total_laps - lap_number
        fuel_correction = remaining_laps * fuel_consumption_per_lap * weight_effect
        return raw_laptime - fuel_correction
    
    def calculate_tire_performance(self, compound, stint_lap, tire_age, base_laptime, circuit_name):
        """Calculate tire performance - use 5-point curves if enabled, otherwise default physics"""
        
        # Check if forcing default physics model
        if hasattr(st.session_state, 'force_default_physics') and st.session_state.force_default_physics:
            # Skip all other models and go directly to default physics
            return self._calculate_default_physics_model(compound, stint_lap, tire_age, base_laptime)
        
        # Check if tire curves are enabled
        if hasattr(st.session_state, 'use_f1_curves') and st.session_state.use_f1_curves:
            return st.session_state.f1_tire_editor.calculate_f1_tire_performance(
                compound, stint_lap + tire_age, base_laptime
            )
        
        # Fallback to default physics model 
        return self._calculate_default_physics_model(compound, stint_lap, tire_age, base_laptime)
    
    def _calculate_default_physics_model(self, compound, stint_lap, tire_age, base_laptime):
        """Calculate tire performance using default physics model"""
        
        compound_degradation = {
            'SOFT': 0.08,       
            'MEDIUM': 0.04,   
            'HARD': 0.02      
        }
        
        compound_offsets = {
            'SOFT': -0.8,     
            'MEDIUM': 0.0,    
            'HARD': +0.5      
        }
        
        # Calculate degradation
        deg_rate = compound_degradation.get(compound, 0.05)
        offset = compound_offsets.get(compound, 0.0)
        
        # Age effect from previous usage
        age_effect = tire_age * 0.02
        
        # Stint degradation (linear + slight exponential for long stints)
        stint_degradation = deg_rate * stint_lap
        if stint_lap > 20:
            stint_degradation += 0.02 * (stint_lap - 20) ** 1.5
        
        calculated_laptime = base_laptime + offset + age_effect + stint_degradation
        
        return calculated_laptime
    
    def validate_tire_allocation(self, strategy, tire_allocation):
        if not tire_allocation:
            return True, "Using fresh tires"
            
        required_sets = {}
        for stint in strategy:
            compound = stint['compound']
            required_sets[compound] = required_sets.get(compound, 0) + 1
            
        for compound, needed in required_sets.items():
            available = len([t for t in tire_allocation if t['compound'] == compound])
            if available < needed:
                return False, f"Need {needed} {compound} sets, only have {available}"
                
        return True, "Strategy is valid"
    
    def assign_tires_to_strategy(self, strategy, tire_allocation):
        if not tire_allocation:
            return [{'compound': stint['compound'], 'laps': stint['laps'], 'tire_age': 0} 
                   for stint in strategy]
            
        tire_sets = {compound: [] for compound in ['SOFT', 'MEDIUM', 'HARD']}
        for tire in tire_allocation:
            tire_sets[tire['compound']].append(tire)
        
        for compound in tire_sets:
            tire_sets[compound].sort(key=lambda x: x['age_laps'])
        
        assigned_strategy = []
        for stint in strategy:
            compound = stint['compound']
            if tire_sets[compound]:
                tire_set = tire_sets[compound].pop(0)
                assigned_strategy.append({
                    'compound': compound,
                    'laps': stint['laps'],
                    'tire_age': tire_set['age_laps']
                })
            else:
                raise ValueError(f"No more {compound} tire sets available.")
                
        return assigned_strategy

    def simulate_race_strategy(self, circuit, strategy, tire_allocation=None, base_pace=80.0, 
                             pit_loss=22.0, num_simulations=1000, progress_bar=None, status_text=None):
        circuit_info = self.circuits[circuit]
        total_laps = circuit_info['laps']
        fuel_per_lap = self.calculate_fuel_consumption(circuit)
        
        is_valid, message = self.validate_tire_allocation(strategy, tire_allocation)
        if not is_valid:
            raise ValueError(f"Invalid strategy: {message}")
        
        assigned_strategy = self.assign_tires_to_strategy(strategy, tire_allocation)
        
        results = []
        
        for sim in range(num_simulations):
            # Update progress bar and status text
            if progress_bar and sim % 50 == 0:
                progress = sim / num_simulations
                progress_bar.progress(progress)
                if status_text:
                    status_text.text(f"Running simulation {sim + 1}/{num_simulations}...")
                
            race_time = 0
            current_lap = 1
            sim_base_pace = base_pace + np.random.normal(0, 0.5)
            
            for stint_idx, stint in enumerate(assigned_strategy):
                compound = stint['compound']
                stint_length = stint['laps']
                tire_age = stint['tire_age']
                
                remaining_laps = total_laps - current_lap + 1
                stint_length = min(stint_length, remaining_laps)
                
                for stint_lap in range(1, stint_length + 1):
                    if current_lap > total_laps:
                        break
                    
                    raw_laptime = self.calculate_tire_performance(
                        compound, stint_lap, tire_age, sim_base_pace, circuit
                    )
                    
                    laptime = self.calculate_fuel_corrected_laptime(
                        raw_laptime, current_lap, total_laps, fuel_per_lap
                    )
                    
                    laptime += np.random.normal(0, 0.2)
                    race_time += laptime
                    current_lap += 1
                    
                if stint_idx < len(assigned_strategy) - 1:
                    race_time += pit_loss
                        
            results.append(race_time)
        
        # Final progress update
        if progress_bar:
            progress_bar.progress(1.0)
        if status_text:
            status_text.text("Simulation complete!")
            
        return np.array(results)
    # Helper functions
def create_results_export(results, circuit, analysis_params):
    """Create exportable results data"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare summary statistics
    summary_data = []
    for name, times in results.items():
        summary_data.append({
            "Strategy": name,
            "Median_Time": round(np.median(times), 1),
            "Mean_Time": round(np.mean(times), 1),
            "Std_Dev": round(np.std(times), 1),
            "5th_Percentile": round(np.percentile(times, 5), 1),
            "95th_Percentile": round(np.percentile(times, 95), 1),
            "Range": round(np.percentile(times, 95) - np.percentile(times, 5), 1),
            "Best_Time": round(np.min(times), 1),
            "Worst_Time": round(np.max(times), 1)
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values("Median_Time")
    
    # Prepare detailed results
    detailed_data = []
    for strategy_name, times in results.items():
        for i, time in enumerate(times):
            detailed_data.append({
                "Strategy": strategy_name,
                "Simulation": i + 1,
                "Race_Time": round(time, 3)
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    
    # Prepare metadata
    metadata = {
        "Export_Timestamp": timestamp,
        "Circuit": circuit,
        "Analysis_Parameters": analysis_params,
        "Tire_Model": "5-Point Curves" if analysis_params.get('use_f1_curves', False) else "Default Models",
        "Summary_Statistics": summary_df.to_dict('records'),
        "Strategy_Count": len(results),
        "Simulations_Per_Strategy": len(list(results.values())[0]) if results else 0
    }
    
    return summary_df, detailed_df, metadata

def export_results_csv(results, circuit, analysis_params):
    """Export results as CSV"""
    summary_df, detailed_df, metadata = create_results_export(results, circuit, analysis_params)
    
    # Create CSV buffer
    output = io.StringIO()
    
    # Write metadata
    output.write("# F1 Strategy Analysis Results\n")
    output.write(f"# Circuit: {circuit}\n")
    output.write(f"# Export Time: {metadata['Export_Timestamp']}\n")
    output.write(f"# Tire Model: {metadata['Tire_Model']}\n")
    output.write(f"# Strategies: {metadata['Strategy_Count']}\n")
    output.write(f"# Simulations: {metadata['Simulations_Per_Strategy']}\n")
    output.write("\n")
    
    # Write summary
    output.write("# SUMMARY STATISTICS\n")
    summary_df.to_csv(output, index=False)
    output.write("\n")
    
    # Write detailed results
    output.write("# DETAILED SIMULATION RESULTS\n")
    detailed_df.to_csv(output, index=False)
    
    return output.getvalue()

def export_results_json(results, circuit, analysis_params):
    """Export results as JSON"""
    summary_df, detailed_df, metadata = create_results_export(results, circuit, analysis_params)
    
    export_data = {
        "metadata": metadata,
        "summary_statistics": summary_df.to_dict('records'),
        "detailed_results": detailed_df.to_dict('records'),
        "raw_simulation_data": {name: times.tolist() for name, times in results.items()}
    }
    
    return json.dumps(export_data, indent=2)

def create_performance_plot(results, circuit):
    """Create performance visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Strategy Performance - {circuit}', fontsize=16, fontweight='bold')
    
    colors = ['steelblue', 'sandybrown', 'crimson', 'seagreen', 'indianred']
    
    # performance Distribution
    for i, (name, times) in enumerate(results.items()):
        color = colors[i % len(colors)]
        ax1.hist(times, bins=30, alpha=0.6, label=name, color=color, density=True)
        median = np.median(times)
        ax1.axvline(median, color=color, linestyle='--', linewidth=2)
                          
    ax1.set_xlabel("Race Time (seconds)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("Performance Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # box plot
    data_for_box = [results[name] for name in results.keys()]
    labels_for_box = list(results.keys())
    
    box_plot = ax2.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    for i, patch in enumerate(box_plot['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)
        
    ax2.set_ylabel("Race Time (seconds)")
    ax2.set_title("Performance Spread")
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # performance metrics
    strategies = list(results.keys())
    x_pos = np.arange(len(strategies))
    
    medians = [np.median(results[s]) for s in strategies]
    p95s = [np.percentile(results[s], 95) for s in strategies]
    
    ax3.bar(x_pos, medians, alpha=0.8, color='lightblue', label='Median')
    ax3.bar(x_pos, p95s, alpha=0.5, color='lightcoral', label='95th Percentile')
    
    ax3.set_xlabel("Strategy")
    ax3.set_ylabel("Race Time (seconds)")
    ax3.set_title("Performance Comparison")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(strategies, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # cumulative distribution
    for i, (strategy, times) in enumerate(results.items()):
        sorted_times = np.sort(times)
        cumulative_prob = np.arange(1, len(times) + 1) / len(times)
        ax4.plot(sorted_times, cumulative_prob, label=strategy, 
                color=colors[i % len(colors)], linewidth=2)
    
    ax4.set_xlabel("Race Time (seconds)")
    ax4.set_ylabel("Cumulative Probability")
    ax4.set_title("Cumulative Distribution")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    return fig

def run_strategy_analysis(sim, circuit, selected_strategies, tire_allocation, base_pace, pit_loss, num_sims, progress_bar=None, status_text=None):
    """Run the strategy analysis with custom or default strategies and progress tracking"""
    circuit_info = sim.circuits[circuit]
    circuit_laps = circuit_info['laps']
    
    # Use custom strategies if available, otherwise use defaults
    if hasattr(st.session_state, 'advanced_strategies') and st.session_state.advanced_strategies:
        strategies_to_use = st.session_state.advanced_strategies
        use_scaling = False  # Don't scale advanced strategies
    elif hasattr(st.session_state, 'custom_strategies') and st.session_state.custom_strategies:
        strategies_to_use = st.session_state.custom_strategies
        use_scaling = False  # Don't scale custom strategies
    else:
        strategies_to_use = {name: ALL_STRATEGIES[name] for name in selected_strategies}
        use_scaling = True  # Scale default strategies
    
    # Adjust strategies for circuit (only if using default strategies)
    adjusted_strategies = {}
    for strategy_name in selected_strategies:
        if strategy_name in strategies_to_use:
            strategy = strategies_to_use[strategy_name]
            
            if use_scaling:
                total_original_laps = sum(stint['laps'] for stint in strategy)
                scale_factor = circuit_laps / total_original_laps
                
                adjusted_strategy = []
                remaining_laps = circuit_laps
                
                for i, stint in enumerate(strategy):
                    if i == len(strategy) - 1:
                        laps = remaining_laps
                    else:
                        scaled_laps = stint['laps'] * scale_factor
                        laps = max(1, round(scaled_laps))
                        laps = min(laps, remaining_laps - (len(strategy) - i - 1))
                        remaining_laps -= laps
                        
                    adjusted_strategy.append({'compound': stint['compound'], 'laps': laps})
                
                adjusted_strategies[strategy_name] = adjusted_strategy
            else:
                # Use custom strategy as-is
                adjusted_strategies[strategy_name] = strategy
    
    # Run simulations with progress tracking
    results = {}
    total_strategies = len(adjusted_strategies)
    
    for strategy_idx, (name, strategy) in enumerate(adjusted_strategies.items()):
        if status_text:
            status_text.text(f"Analyzing {name}... ({strategy_idx + 1}/{total_strategies})")
        
        try:
            times = sim.simulate_race_strategy(
                circuit, strategy, tire_allocation, base_pace, pit_loss, num_sims,
                progress_bar=progress_bar, status_text=status_text
            )
            results[name] = times
                
        except ValueError as e:
            st.error(f"{name}: {e}")
            continue
    
    return results
def create_pdf_report(results, circuit, analysis_params):
    """Create PDF report with all tables and details"""
    from matplotlib.backends.backend_pdf import PdfPages
    from io import BytesIO
    
    # Create a BytesIO buffer for the PDF
    buffer = BytesIO()
    
    # Create the PDF with multiple pages
    with PdfPages(buffer) as pdf:
        # Page 1: Summary and Overview
        fig1 = plt.figure(figsize=(11, 8.5))
        fig1.suptitle(f'F1 Strategy Analysis Report - {circuit}', fontsize=16, fontweight='bold')
        
        ax = fig1.add_subplot(111)
        ax.axis('off')
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tire_model = "5-Point Curves" if analysis_params.get('use_f1_curves', False) else "Default Models"
        
        header_text = f"""F1 Strategy Simulator Pro - Analysis Report

Circuit: {circuit}
Generated: {timestamp}
Tire Model: {tire_model}
Base Pace: {analysis_params.get('base_pace', 'N/A')}s
Pit Loss: {analysis_params.get('pit_loss', 'N/A')}s
Simulations: {analysis_params.get('num_sims', 'N/A')} per strategy
Strategies Analyzed: {len(results)}"""
        
        ax.text(0.05, 0.95, header_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace')
        
        # Summary table
        summary_data = []
        for name, times in results.items():
            summary_data.append({
                "Strategy": name,
                "Median": f"{np.median(times):.1f}s",
                "Mean": f"{np.mean(times):.1f}s",
                "Std Dev": f"{np.std(times):.1f}s",
                "Best": f"{np.min(times):.1f}s",
                "95th %ile": f"{np.percentile(times, 95):.1f}s"
            })
        
        df_summary = pd.DataFrame(summary_data).sort_values("Median")
        
        table_text = "STRATEGY PERFORMANCE SUMMARY\n"
        table_text += "=" * 70 + "\n"
        table_text += f"{'Strategy':<20} {'Median':<10} {'Mean':<10} {'Std Dev':<10} {'Best':<10} {'95th %ile':<10}\n"
        table_text += "-" * 70 + "\n"
        
        for _, row in df_summary.iterrows():
            table_text += f"{row['Strategy']:<20} {row['Median']:<10} {row['Mean']:<10} {row['Std Dev']:<10} {row['Best']:<10} {row['95th %ile']:<10}\n"
        
        ax.text(0.05, 0.65, table_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        best_strategy = df_summary.iloc[0]['Strategy']
        most_consistent = df_summary.loc[df_summary['Std Dev'].str.replace('s', '').astype(float).idxmin(), 'Strategy']
        
        insights_text = f"""KEY INSIGHTS

Fastest Strategy: {best_strategy}
Most Consistent: {most_consistent}

The analysis shows performance differences between strategies based on
{analysis_params.get('num_sims', 'N/A')} Monte Carlo simulations per strategy."""
        
        ax.text(0.05, 0.25, insights_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top')
        
        pdf.savefig(fig1, bbox_inches='tight')
        plt.close(fig1)
        
        # Page 2: Head-to-Head Win Rates and Risk Analysis
        fig2 = plt.figure(figsize=(11, 8.5))
        fig2.suptitle('Head-to-Head Analysis', fontsize=16, fontweight='bold')
        
        ax2a = fig2.add_subplot(2, 1, 1)
        ax2b = fig2.add_subplot(2, 1, 2)
        ax2a.axis('off')
        ax2b.axis('off')
        
        # Head-to-Head Win Rates
        strategy_names = list(results.keys())
        win_rates_text = "HEAD-TO-HEAD WIN RATES\n"
        win_rates_text += "=" * 80 + "\n\n"
        
        # Create abbreviated strategy names
        abbrev_names = {}
        for name in strategy_names:
            if "1-stop" in name:
                compounds = name.split("(")[1].split(")")[0]
                abbrev_names[name] = f"1S-{compounds}"
            elif "2-stop" in name:
                compounds = name.split("(")[1].split(")")[0]
                abbrev_names[name] = f"2S-{compounds}"
            else:
                abbrev_names[name] = name[:8]
        
        # Header row
        win_rates_text += f"{'Strategy':<12}"
        for strategy in strategy_names:
            win_rates_text += f"{abbrev_names[strategy]:<9}"
        win_rates_text += "\n" + "-" * (12 + len(strategy_names) * 9) + "\n"
        
        # Data rows
        for strategy1 in strategy_names:
            win_rates_text += f"{abbrev_names[strategy1]:<12}"
            for strategy2 in strategy_names:
                if strategy1 == strategy2:
                    win_rates_text += f"{'—':<9}"
                else:
                    wins = np.sum(results[strategy1] < results[strategy2])
                    total = len(results[strategy1])
                    win_rate = (wins / total) * 100
                    win_rates_text += f"{win_rate:.1f}%{'':<5}"
            win_rates_text += "\n"
        
        ax2a.text(0.05, 0.95, win_rates_text, transform=ax2a.transAxes, fontsize=8,
                 verticalalignment='top', fontfamily='monospace')
        
        # Risk Analysis
        risk_text = "RISK ANALYSIS\n"
        risk_text += "=" * 50 + "\n\n"
        
        fastest_median = min(np.median(times) for times in results.values())
        
        risk_text += f"{'Strategy':<20} {'Time Penalty':<15} {'Risk (±s)':<10}\n"
        risk_text += "-" * 50 + "\n"
        
        strategy_times = [(name, times) for name, times in results.items()]
        strategy_times.sort(key=lambda x: np.median(x[1]))
        
        for strategy_name, times in strategy_times:
            penalty = np.median(times) - fastest_median
            risk = np.std(times)
            risk_text += f"{strategy_name[:19]:<20} +{penalty:.1f}s{'':<10} {risk:.1f}\n"
        
        ax2b.text(0.05, 0.95, risk_text, transform=ax2b.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace')
        
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)
        # Page 3: Strategy Configurations
        fig3 = plt.figure(figsize=(11, 8.5))
        fig3.suptitle('Strategy Configuration Details', fontsize=16, fontweight='bold')
        
        ax3 = fig3.add_subplot(111)
        ax3.axis('off')
        
        # Get circuit information for accurate lap counts
        circuit_laps = 70  # default fallback
        if 'simulator' in st.session_state:
            sim = st.session_state.simulator
            if circuit in sim.circuits:
                circuit_laps = sim.circuits[circuit]['laps']
        
        strategy_text = "STRATEGY CONFIGURATIONS\n"
        strategy_text += "=" * 50 + "\n\n"
        strategy_text += f"Circuit: {circuit} ({circuit_laps} laps)\n\n"
        
        # Get the strategies used in analysis
        custom_strategies = getattr(st.session_state, 'custom_strategies', None)
        advanced_strategies = getattr(st.session_state, 'advanced_strategies', None)
        
        if advanced_strategies:
            strategies_to_show = advanced_strategies
            strategy_text += "ADVANCED CUSTOM STRATEGIES:\n\n"
        elif custom_strategies:
            strategies_to_show = custom_strategies
            strategy_text += "CUSTOM STRATEGIES:\n\n"
        else:
            # Generate scaled default strategies to match what was analyzed
            strategies_to_show = {}
            for strategy_name in results.keys():
                if strategy_name in ALL_STRATEGIES:
                    base_strategy = ALL_STRATEGIES[strategy_name]
                    total_original_laps = sum(stint['laps'] for stint in base_strategy)
                    scale_factor = circuit_laps / total_original_laps
                    
                    scaled_strategy = []
                    remaining_laps = circuit_laps
                    
                    for i, stint in enumerate(base_strategy):
                        if i == len(base_strategy) - 1:
                            laps = remaining_laps
                        else:
                            scaled_laps = stint['laps'] * scale_factor
                            laps = max(1, round(scaled_laps))
                            laps = min(laps, remaining_laps - (len(base_strategy) - i - 1))
                            remaining_laps -= laps
                        
                        scaled_strategy.append({'compound': stint['compound'], 'laps': laps})
                    
                    strategies_to_show[strategy_name] = scaled_strategy
            
            strategy_text += "DEFAULT STRATEGIES (Circuit-Scaled):\n\n"
        
        for strategy_name, strategy in strategies_to_show.items():
            strategy_text += f"{strategy_name}:\n"
            total_laps = sum(stint['laps'] for stint in strategy)
            
            for i, stint in enumerate(strategy):
                strategy_text += f"  Stint {i+1}: {stint['laps']} laps on {stint['compound']}\n"
            
            strategy_text += f"  Total Laps: {total_laps}\n"
            strategy_text += f"  Pit Stops: {len(strategy) - 1}\n\n"
        
        # Add tire allocation if used
        tire_allocation = analysis_params.get('tire_allocation')
        if tire_allocation:
            strategy_text += "\nCUSTOM TIRE ALLOCATION:\n"
            strategy_text += "=" * 30 + "\n"
            
            # Group tires by compound
            tire_summary = {}
            for tire in tire_allocation:
                compound = tire['compound']
                age = tire['age_laps']
                if compound not in tire_summary:
                    tire_summary[compound] = []
                tire_summary[compound].append(age)
            
            for compound in ['SOFT', 'MEDIUM', 'HARD']:
                if compound in tire_summary:
                    ages = tire_summary[compound]
                    strategy_text += f"\n{compound} TIRES:\n"
                    strategy_text += f"  Total Sets: {len(ages)}\n"
                    strategy_text += f"  Ages (laps): {', '.join(map(str, ages))}\n"
                    strategy_text += f"  Fresh Sets: {sum(1 for age in ages if age == 0)}\n"
                    strategy_text += f"  Used Sets: {sum(1 for age in ages if age > 0)}\n"
        
        ax3.text(0.05, 0.95, strategy_text, transform=ax3.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace')
        
        pdf.savefig(fig3, bbox_inches='tight')
        plt.close(fig3)
        
        # Page 4: Performance visualization plots
        fig4 = create_performance_plot(results, circuit)
        pdf.savefig(fig4, bbox_inches='tight')
        plt.close(fig4)
        
        # Page 5: Tire curve information (if F1 curves used)
        if analysis_params.get('use_f1_curves', False):
            fig5 = plt.figure(figsize=(11, 8.5))
            fig5.suptitle('Tire Curve Configuration', fontsize=16, fontweight='bold')
            fig5.subplots_adjust(hspace=0.6)
            
            # Split into text and plot
            ax5a = fig5.add_subplot(2, 1, 1)
            ax5b = fig5.add_subplot(2, 1, 2)
            ax5a.axis('off')
            
            curves = st.session_state.f1_tire_curves
            curve_text = "5-POINT TIRE CURVE PARAMETERS\n"
            curve_text += "=" * 50 + "\n\n"
            
            for compound in ['SOFT', 'MEDIUM', 'HARD']:
                curve = curves[compound]
                curve_text += f"{compound} COMPOUND:\n"
                curve_text += f"  Point 1 - Compound Offset: +{curve['compound_offset']:.1f}s\n"
                curve_text += f"  Point 2 - Warmup Effect: {curve['warmup_effect']:.1f}s\n"
                curve_text += f"  Point 3 - Linear Degradation: {curve['linear_degradation']:.3f}s/lap\n"
                curve_text += f"  Point 4 - Wear Life Start: Lap {curve['wear_life_start']}\n"
                curve_text += f"  Point 5 - Max Usable Laps: {curve['max_usable_laps']}\n\n"
            
            ax5a.text(0.05, 0.95, curve_text, transform=ax5a.transAxes, fontsize=10,
                     verticalalignment='top', fontfamily='monospace')
            
            # Add tire curve plot
            base_pace = analysis_params.get('base_pace', 80.0)
            max_laps = 50
            
            colors = {'SOFT': '#FF4444', 'MEDIUM': '#FFB347', 'HARD': '#808080'}
            
            for compound in ['SOFT', 'MEDIUM', 'HARD']:
                laps = list(range(1, max_laps + 1))
                laptimes = []
                
                for lap in laps:
                    # Use the same calculation as in the F1FivePointTireEditor
                    curve = curves[compound]
                    laptime = base_pace
                    
                    # Point 1: Compound offset
                    laptime += curve['compound_offset']
                    
                    # Point 2: Warmup effect (first 3 laps)
                    if lap <= 3:
                        warmup_factor = (4 - lap) / 3
                        laptime += curve['warmup_effect'] * warmup_factor
                    
                    # Point 3: Linear deg
                    laptime += curve['linear_degradation'] * lap
                    
                    # Point 4: Wear life
                    if lap > curve['wear_life_start']:
                        wear_laps = lap - curve['wear_life_start']
                        high_deg = 0.1 * (wear_laps ** 1.8)
                        laptime += high_deg
                    
                    # Point 5: Beyond max usable
                    if lap > curve['max_usable_laps']:
                        danger_laps = lap - curve['max_usable_laps']
                        laptime += 5.0 + danger_laps * 2.0
                    
                    laptimes.append(laptime)
                
                ax5b.plot(laps, laptimes, color=colors[compound], linewidth=2, label=compound)
            
            ax5b.set_xlabel('Stint Lap')
            ax5b.set_ylabel('Lap Time (s)')
            ax5b.set_title('5-Point Tire Performance Curves')
            ax5b.legend()
            ax5b.grid(True, alpha=0.3)
            
            pdf.savefig(fig5, bbox_inches='tight')
            plt.close(fig5)
    
    buffer.seek(0)
    return buffer.getvalue()
def main():
    st.title("🏎️ F1 Strategy Simulator Pro 🏎️")
    
    # Initialize simulator
    if 'simulator' not in st.session_state:
        st.session_state.simulator = F1StrategySimulator()
    
    sim = st.session_state.simulator
    
    # Sidebar configuration
    st.sidebar.header("🏁 Race Configuration")
    
    # Circuit selection in sidebar
    circuit_names = ["Select Race"] + [name for name, _ in sim.circuit_data]
    
    # Use a reset counter to force widget reset
    reset_key = getattr(st.session_state, 'reset_counter', 0)
    
    circuit = st.sidebar.selectbox(
        "Select Circuit", 
        circuit_names, 
        key=f"circuit_selector_{reset_key}"
    )
    
    if circuit != "Select Race":
        circuit_info = sim.circuits[circuit]
        st.sidebar.info(f"""
        **{circuit} GP**
        - {circuit_info['laps']} laps
        - {circuit_info['distance_km']:.3f} km/lap
        - {sim.calculate_fuel_consumption(circuit):.2f} kg fuel/lap
        """)
    
    # Strategy selection in sidebar
    selected_strategies = st.sidebar.multiselect(
        "Select Strategies",
        list(ALL_STRATEGIES.keys()),
        default=[],
        key=f"strategies_selector_{reset_key}"
    )
    
    # Tire allocation in sidebar
    use_custom_tires = st.sidebar.checkbox(
        "Custom Tire Allocation",
        key=f"custom_tires_{reset_key}"
    )
    tire_allocation = None
    
    if use_custom_tires:
        st.sidebar.write("**Tire Sets Available:**")
        tire_allocation = []
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            num_sets = st.sidebar.number_input(
                f"{compound} sets", 
                0, 8, 2, 
                key=f"{compound}_sets_{reset_key}"
            )
            for i in range(num_sets):
                age = st.sidebar.number_input(
                    f"{compound} set {i+1} age", 
                    0, 50, 0, 
                    key=f"{compound}_{i}_{reset_key}"
                )
                tire_allocation.append({'compound': compound, 'age_laps': age})
    
    # Circuit and simulation parameters in sidebar
    if circuit != "Select Race" and selected_strategies:
        st.sidebar.markdown("---")
        st.sidebar.header("Analysis Parameters")
        
        # Circuit-specific defaults
        circuit_base_paces = {
            'Australia': 82.0, 'China': 95.0, 'Japan': 92.0, 'Bahrain': 93.0,
            'Saudi Arabia': 90.0, 'Miami': 91.0, 'Imola': 77.0, 'Monaco': 75.0,
            'Spain': 78.0, 'Canada': 75.0, 'Austria': 67.0, 'Britain': 88.0,
            'Hungary': 78.0, 'Belgium': 107.0, 'Netherlands': 72.0, 'Italy': 83.0,
            'Azerbaijan': 102.0, 'Singapore': 95.0, 'United States': 96.0,
            'Mexico': 78.0, 'Brazil': 72.0, 'Las Vegas': 85.0, 'Qatar': 84.0,
            'Abu Dhabi': 87.0
        }
        
        circuit_pit_losses = {
            'Australia': 21.5, 'China': 20.8, 'Japan': 20.2, 'Bahrain': 19.8,
            'Saudi Arabia': 22.1, 'Miami': 18.5, 'Imola': 21.8, 'Monaco': 16.2,
            'Spain': 21.4, 'Canada': 15.8, 'Austria': 18.9, 'Britain': 20.5,
            'Hungary': 22.8, 'Belgium': 23.2, 'Netherlands': 20.1, 'Italy': 15.9,
            'Azerbaijan': 21.7, 'Singapore': 22.5, 'United States': 20.3,
            'Mexico': 21.1, 'Brazil': 19.4, 'Las Vegas': 19.6, 'Qatar': 20.7,
            'Abu Dhabi': 21.3
        }
        
        default_pace = circuit_base_paces.get(circuit, 80.0)
        base_pace = st.sidebar.slider(
            "Base Pace (s)", 60.0, 120.0, default_pace, 0.1,
            help=f"Typical lap time for {circuit}. Default: {default_pace}s",
            key=f"base_pace_{reset_key}"
        )
        
        default_pit_loss = circuit_pit_losses.get(circuit, 22.0)
        pit_loss = st.sidebar.slider(
            "Pit Loss (s)", 15.0, 35.0, default_pit_loss, 0.1,
            help=f"Time penalty for pit stop at {circuit}. Default: {default_pit_loss}s",
            key=f"pit_loss_{reset_key}"
        )
        
        num_sims = st.sidebar.slider(
            "Number of Simulations", 100, 2000, 1000, 100,
            key=f"num_sims_{reset_key}"
        )
        
        # Show current tire model status in sidebar
        st.sidebar.subheader("Current Tire Model")
        tire_model_type_current = getattr(st.session_state, 'tire_model_type', 'Default Physics Models')
        
        if getattr(st.session_state, 'use_f1_curves', False):
            st.sidebar.success("Using 5-Point Curves")
        else:
            st.sidebar.info("Using Default Physics Models")
        
        # RUN ANALYSIS BUTTON in sidebar
        st.sidebar.markdown("---")
        if st.sidebar.button("Run Strategy Analysis", type="primary", use_container_width=True):
            # Store parameters for results tab
            st.session_state.analysis_params = {
                'circuit': circuit,
                'selected_strategies': selected_strategies,
                'base_pace': base_pace,
                'pit_loss': pit_loss,
                'num_sims': num_sims,
                'tire_allocation': tire_allocation,
                'tire_model_type': tire_model_type_current,
                'use_f1_curves': getattr(st.session_state, 'use_f1_curves', False)
            }
            
            # Run the analysis with progress tracking
            with st.spinner("Running Monte Carlo simulations..."):
                # Create progress bar and status text
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                results = run_strategy_analysis(sim, circuit, selected_strategies, 
                                              tire_allocation, base_pace, pit_loss, num_sims,
                                              progress_bar=progress_bar, status_text=status_text)
                
                if results:
                    st.session_state.results = results
                    st.session_state.results_ready = True
                    st.session_state.analysis_complete_message = True
    
    # Check if race is selected
    race_selected = circuit != "Select Race"
    
    # Main content area
    if not race_selected:
        # Show welcome page
        st.markdown("""
        ## 🏁 Welcome to the F1 Strategy Simulator Pro! 🏁
        
        This tool helps you analyze and compare different pit stop strategies for Formula 1 races using advanced tire modeling and Monte Carlo simulation.
        
        ### How to Use:
        
        1. 📍 **Select a Circuit** 📍
           - Choose from the 2025 F1 calendar in the sidebar
           - Each circuit has unique characteristics that affect strategy
        
        2. 📊 **Choose Strategies** 📊
           - Pick one or more strategies to compare
           - Options include 1-stop and 2-stop strategies with different tire compounds
           - S = Soft, M = Medium, H = Hard tires
           - Continue with default stint selections or make edits using the Custom Strategy Editor
        
        3. 🏁 **Configure Tire Model (Optional)** 🏁
           - **5-Point Curves**: Professional tire modeling with editable parameters
             - Point 1: Compound baseline performance (relative to SOFT)
             - Point 2: Tire warmup characteristics
             - Point 3: Linear degradation rate
             - Point 4: Performance cliff detection
             - Point 5: Maximum usable stint length
           - **Default Physics**: Simple physics-based degradation models
        
        4. ⚙️ **Adjust Parameters** ⚙️
           - **Base Pace**: Typical lap time for the circuit (auto-selected per circuit, but adjustable)
           - **Pit Loss**: Time penalty for each pit stop (auto-selected per circuit, but adjustable)
           - **Simulations**: Number of Monte Carlo runs (more = more accurate, fewer = faster)
        
        5. 🏎️ **Optional: Custom Tire Allocation** 🏎️
           - Set specific tire sets and their ages in the sidebar
                    
        6. 🏁 **Run Analysis** 🏁
           - Click "Run Analysis" to simulate thousands of race scenarios
           - View performance distributions, risk analysis, and head-to-head comparisons
        
        #### Description:
        The F1 Strategy Simulator employs a dual-modal tire modeling framework combining
        editable 5-point curve analysis and physics-based fallback models with Monte Carlo simulation
        for race strategy optimization.
                    
        The primary tire performance model implements a user-configurable 5-point curve system
        with adjustable parameters: (1) compound baseline performance relative to SOFT tires,
        (2) tire warmup penalty duration and magnitude for cold tires, (3) linear degradation
        rate per lap during normal operation, (4) performance cliff onset lap where high
        degradation begins, and (5) maximum usable stint length beyond which severe penalties
        apply. Users can interactively adjust these parameters through sliders to model
        different tire characteristics and circuit conditions.
                    
        When 5-point curves are not selected, the system falls back to physics-based tire
        models using compound-specific degradation rates and compound performance offsets.
        The physics model applies linear degradation with exponential penalties for extended
        stints beyond typical operating windows, plus age-based performance penalties for
        used tire sets.
                    
        Fuel correction applies weight-based transformations accounting for decreasing
        fuel load throughout the race, where fuel consumption is calculated from total
        fuel allocation divided by circuit lap count. The Monte Carlo engine executes
        user-configurable simulation counts while adding appropriate Gaussian noise for
        lap-to-lap variability and base pace variation between simulations.
                    
        The system incorporates tire allocation optimization with age-based performance
        penalties, circuit-specific pit loss timing, and strategy validation against
        available tire sets. Circuit adaptation automatically scales strategy lap counts
        proportionally to race distance while maintaining compound sequences. Output
        analysis generates empirical probability distributions from Monte Carlo samples,
        providing percentile-based risk metrics, median performance comparisons, and
        head-to-head win rate matrices for strategy evaluation under uncertainty.
        
        **Get started by selecting a circuit from the sidebar!** 👈
        """)
        
    elif not selected_strategies:
        # Show strategy selection instructions
        st.subheader(f"🏁 {circuit} Grand Prix Selected")
        
        circuit_info = sim.circuits[circuit]
        st.info(f"""
        **Circuit Info:** {circuit_info['laps']} laps • {circuit_info['distance_km']:.3f} km per lap • {sim.calculate_fuel_consumption(circuit):.2f} kg fuel per lap
        """)
        
        st.markdown("""
        ### Next Steps:
        1. **Select Strategies** from the sidebar to compare
        2. **Configure tire model** if needed (optional)
        3. **Run the analysis** to see results
        """)
        
    else:
        # Show main analysis interface with tabs
        setup_tab, results_tab = st.tabs(["🔧  Setup", "📈  Results"])
        
        with setup_tab:
            # Show analysis complete message at top if analysis was just completed
            if getattr(st.session_state, 'analysis_complete_message', False):
                st.success("Analysis complete! Check the Results tab.")
                st.session_state.analysis_complete_message = False  # Clear the message
            
            # Tire Model Configuration
            st.header("Tire Model Configuration")
            
            tire_model_type = st.radio(
                "Select Tire Model",
                ["5-Point Curves (Editable)", "Default Physics Models"],
                key="tire_model_type"
            )
            
            if tire_model_type == "5-Point Curves (Editable)":
                st.session_state.use_f1_curves = True
                st.session_state.force_default_physics = False
                st.info("Using 5-Point Curves - Configure below")
                
                # Render the tire curve editor
                st.session_state.f1_tire_editor.render_tire_curve_editor()
                
            else:  # Default Physics Models
                st.session_state.use_f1_curves = False
                st.session_state.force_default_physics = True
                st.info("Using Default Physics-Based Models")
            
            # Strategy editors
            strategy_editor_type = st.radio(
                "Strategy Configuration",
                ["Use Default Strategies", "Custom Strategy Editor"],
                key="strategy_editor_type"
            )
            
            if strategy_editor_type == "Custom Strategy Editor":
                use_custom = render_custom_strategy_editor(sim, circuit, selected_strategies)
            elif strategy_editor_type == "Advanced Strategy Builder":
                use_advanced = render_advanced_strategy_editor(sim, circuit, selected_strategies)
            else:
                # Show selected strategies preview with scaling
                st.header("Selected Strategies")
                circuit_info = sim.circuits[circuit]
                circuit_laps = circuit_info['laps']
                
                for strategy_name in selected_strategies:
                    strategy = ALL_STRATEGIES[strategy_name]
                    total_original_laps = sum(stint['laps'] for stint in strategy)
                    scale_factor = circuit_laps / total_original_laps
                    
                    # Apply scaling logic (same as run_strategy_analysis)
                    scaled_strategy = []
                    remaining_laps = circuit_laps
                    
                    for i, stint in enumerate(strategy):
                        if i == len(strategy) - 1:
                            # Last stint gets all remaining laps
                            laps = remaining_laps
                        else:
                            # Scale intermediate stints proportionally
                            scaled_laps = stint['laps'] * scale_factor
                            laps = max(1, round(scaled_laps))
                            laps = min(laps, remaining_laps - (len(strategy) - i - 1))
                            remaining_laps -= laps
                        
                        scaled_strategy.append({'compound': stint['compound'], 'laps': laps})
                    
                    # Show scaled strategy
                    scaled_stints = []
                    for stint in scaled_strategy:
                        scaled_stints.append(f"{stint['laps']}{stint['compound'][0]}")
                    
                    strategy_str = " → ".join(scaled_stints)
                    total_scaled = sum(stint['laps'] for stint in scaled_strategy)
                    st.write(f"**{strategy_name}:** {strategy_str} (Total: {total_scaled} laps)")
        
        with results_tab:
            if not getattr(st.session_state, 'results_ready', False):
                st.info("Configure your analysis in the Setup tab and run it from the sidebar to see results here")
            else:
                results = st.session_state.results
                params = st.session_state.analysis_params
                
                st.header(f"🏁 Strategy Analysis Results - {params['circuit']}")
                
                # Analysis summary
                modeling_type = "5-Point Curves" if params.get('use_f1_curves', False) else "Default Physics Models"
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Circuit", params['circuit'])
                with col2:
                    st.metric("Strategies", len(results))
                with col3:
                    st.metric("Simulations", params['num_sims'])
                with col4:
                    st.metric("Tire Model", modeling_type)
                
                # Results display in sub-tabs
                summary_tab, performance_tab, export_tab = st.tabs(["📊  Summary", "📈  Performance Analysis", "📁  Export"])
                
                with summary_tab:
                    # prepare summary data
                    summary_data = []
                    for name, times in results.items():
                        summary_data.append({
                            "Strategy": name,
                            "Median (s)": round(np.median(times), 1),
                            "Mean (s)": round(np.mean(times), 1),
                            "Std Dev (s)": round(np.std(times), 1),
                            "5th %ile": round(np.percentile(times, 5), 1),
                            "95th %ile": round(np.percentile(times, 95), 1),
                            "Range": round(np.percentile(times, 95) - np.percentile(times, 5), 1)
                        })
                    
                    df = pd.DataFrame(summary_data).sort_values("Median (s)")
                    st.dataframe(df, use_container_width=True)
                    
                    # Head-to-Head Win Rates
                    st.subheader("⚔️ Head-to-Head Win Rates")
                    
                    strategy_names = list(results.keys())
                    win_rates = {}
                    
                    for i, strategy1 in enumerate(strategy_names):
                        win_rates[strategy1] = {}
                        for j, strategy2 in enumerate(strategy_names):
                            if i == j:
                                win_rates[strategy1][strategy2] = "—"
                            else:
                                wins = np.sum(results[strategy1] < results[strategy2])
                                total = len(results[strategy1])
                                win_rate = (wins / total) * 100
                                win_rates[strategy1][strategy2] = f"{win_rate:.1f}%"
                    
                    win_rates_df = pd.DataFrame(win_rates).T
                    st.dataframe(win_rates_df, use_container_width=True)
                    
                    # Risk Analysis
                    st.subheader("🎯 Risk Analysis")
                    
                    risk_data = []
                    fastest_median = df.iloc[0]['Median (s)']
                    
                    for _, row in df.iterrows():
                        strategy = row['Strategy']
                        penalty = row['Median (s)'] - fastest_median
                        risk = row['Std Dev (s)']
                        
                        risk_data.append({
                            "Strategy": strategy,
                            "Time Penalty (s)": f"+{penalty:.1f}",
                            "Risk (±s)": risk
                        })
                    
                    risk_df = pd.DataFrame(risk_data)
                    st.dataframe(risk_df, use_container_width=True)
                    
                    # Final recommendations
                    best = df.iloc[0]
                    most_consistent = df.loc[df['Std Dev (s)'].idxmin()]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🏆 Fastest Strategy", best['Strategy'], f"{best['Median (s)']}s")
                    with col2:
                        st.metric("🎯 Most Consistent", most_consistent['Strategy'], f"±{most_consistent['Std Dev (s)']}s")
                
                with performance_tab:
                    fig = create_performance_plot(results, params['circuit'])
                    st.pyplot(fig)
                
                with export_tab:
                    st.subheader("Export Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # CSV Export
                        csv_data = export_results_csv(results, params['circuit'], params)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"f1_strategy_analysis_{params['circuit']}_{timestamp}.csv"
                        
                        st.download_button(
                            label="Export as CSV",
                            data=csv_data,
                            file_name=filename,
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # JSON Export
                        json_data = export_results_json(results, params['circuit'], params)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"f1_strategy_analysis_{params['circuit']}_{timestamp}.json"
                        
                        st.download_button(
                            label="Export as JSON",
                            data=json_data,
                            file_name=filename,
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    with col3:
                        # PDF Export
                        pdf_data = create_pdf_report(results, params['circuit'], params)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"f1_strategy_analysis_{params['circuit']}_{timestamp}.pdf"
                        
                        st.download_button(
                            label="Export as PDF",
                            data=pdf_data,
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                    # Export summary
                    st.write("**Export includes:**")
                    st.write("- **CSV**: Summary statistics + detailed results with metadata")
                    st.write("- **JSON**: Complete structured data including raw simulation results")
                    st.write("- **PDF**: Professional report with visualizations and analysis")
                    st.write("- Tire curve settings (if F1 curves used)")
                    st.write("- Custom tire allocation details (if used)")
                    st.write("- Custom strategy configurations (if used)")
    
    # Reset button - appears anywhere except start page
    if race_selected or selected_strategies:
        st.markdown("---")
        if st.button("🔄 Reset Simulator", type="secondary"):
            # Increment reset counter to force widget reset
            st.session_state.reset_counter = getattr(st.session_state, 'reset_counter', 0) + 1
            
            # Clear ALL other session state to reset everything to defaults
            keys_to_keep = ['reset_counter']
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep:
                    del st.session_state[key]
            
            # Clear query params and force complete reset
            st.query_params.clear()
            st.rerun()

if __name__ == "__main__":

    main()
