#!/usr/bin/env python3
"""
Modified tireModel.py that preserves raw stint data for empirical 5-point curve analysis
Saves both Bayesian models AND the raw data needed for empirical parameter extraction
"""

import pickle
import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
warnings.filterwarnings('ignore')

# FastF1 and Bayesian modeling imports
try:
    import fastf1
    import jax.numpy as jnp
    import jax.random as random
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False
    exit(1)

class F1ModelPrebuilder:
    def __init__(self, output_dir="prebuilt_models_v2"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Cache directory for FastF1
        cache_dir = '.f1_cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        fastf1.Cache.enable_cache(cache_dir)
        
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
        
        self.circuits = {name: data for name, data in self.circuit_data}
    
    def calculate_fuel_corrected_laptime(self, raw_laptime, lap_number, total_laps, 
                                       fuel_consumption_per_lap, weight_effect=0.03):
        """Fuel corrected laptime calculation"""
        remaining_laps = total_laps - lap_number
        fuel_correction = remaining_laps * fuel_consumption_per_lap * weight_effect
        return raw_laptime - fuel_correction
    
    def load_and_process_f1_data(self, circuit_name):
        """Load and process F1 data for a circuit, preserving all raw stint data"""
        print(f"  Loading 2024 data for {circuit_name}...")
        
        circuit_info = self.circuits.get(circuit_name)
        if not circuit_info:
            return None
            
        try:
            # Load the specific GP's 2024 race data
            session = fastf1.get_session(2024, circuit_info['gp_name'], 'R')
            session.load()
            
            # Process laps with all available information
            laps = session.laps
            
            # Get comprehensive stint data
            stint_columns = ["Driver", "Stint", "Compound", "LapNumber", "LapTime", 
                           "Sector1Time", "Sector2Time", "Sector3Time", 
                           "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
                           "TyreLife", "FreshTyre", "TrackStatus"]
            
            # Only include columns that exist
            available_columns = [col for col in stint_columns if col in laps.columns]
            stints = laps[available_columns].copy()
            
            # Convert lap time to seconds
            stints["LapTime_s"] = stints["LapTime"].dt.total_seconds()
            stints.dropna(subset=["LapTime_s"], inplace=True)
            
            # Calculate stint lap (lap within each stint)
            stints["StintLap"] = stints.groupby(["Driver", "Stint"]).cumcount() + 1
            
            # Filter valid racing data
            stints = stints[
                (stints["LapTime_s"] > 60) &      # Reasonable lap times
                (stints["LapTime_s"] < 300) &     # No extremely slow laps
                (stints["StintLap"] <= 50) &      # Reasonable stint lengths
                (stints["Compound"].isin(['SOFT', 'MEDIUM', 'HARD']))  # Valid compounds
            ]
            
            if len(stints) == 0:
                print(f"    No valid data found for {circuit_name}")
                return None
            
            # Apply fuel correction
            circuit_laps = circuit_info['laps']
            fuel_per_lap = 105.0 / circuit_laps #110 kg/ num laps w/ 5kg in reserve
            weight_effect = 0.03
            
            stints['LapTime_FC'] = stints.apply(
                lambda row: self.calculate_fuel_corrected_laptime(
                    row['LapTime_s'], row['LapNumber'], circuit_laps, fuel_per_lap, weight_effect
                ), axis=1
            )
            
            # Add additional analysis columns
            stints['LapNumber_Normalized'] = stints['LapNumber'] / circuit_laps
            
            # Calculate tire age (if TyreLife is available)
            if 'TyreLife' in stints.columns:
                stints['TireAge'] = stints['TyreLife'] - stints['StintLap'] + 1
                stints['TireAge'] = stints['TireAge'].clip(lower=0)
            else:
                stints['TireAge'] = 0
            
            # Add stint analysis flags
            stints['IsFirstLap'] = (stints['StintLap'] == 1)
            stints['IsEarlyStint'] = (stints['StintLap'] <= 5)
            stints['IsMidStint'] = (stints['StintLap'] >= 6) & (stints['StintLap'] <= 20)
            stints['IsLateStint'] = (stints['StintLap'] > 20)
            
            print(f"    Processed {len(stints)} laps from {stints['Driver'].nunique()} drivers")
            return stints
            
        except Exception as e:
            print(f"    Error loading data for {circuit_info['gp_name']}: {str(e)}")
            return None
    
    def build_bayesian_tire_model(self, compound_data, circuit_name, compound):
        """Build Bayesian tire model from compound data"""
        print(f"    Building {compound} model...")
        
        if len(compound_data) < 15:
            print(f"      Insufficient data for {compound} ({len(compound_data)} laps)")
            return None
        
        try:
            x_data = jnp.array(compound_data['StintLap'].values, dtype=jnp.float32)
            y_data = jnp.array(compound_data['LapTime_FC'].values, dtype=jnp.float32)
            
            def model(x, y=None):
                alpha = numpyro.sample("alpha", dist.Normal(80, 5))
                beta = numpyro.sample("beta", dist.Normal(0.03, 0.02))
                sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))
                mu = alpha + beta * x
                numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

            kernel = NUTS(model)
            mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, progress_bar=False)
            mcmc.run(random.PRNGKey(0), x_data, y_data)
            
            # Extract samples for serialization
            samples = mcmc.get_samples()
            samples_dict = {
                'alpha': np.array(samples['alpha']),
                'beta': np.array(samples['beta']),
                'sigma': np.array(samples['sigma'])
            }
            
            model_data = {
                'samples': samples_dict,
                'n_observations': len(compound_data),
                'circuit': circuit_name,
                'compound': compound,
                'x_range': [float(x_data.min()), float(x_data.max())],
                'y_range': [float(y_data.min()), float(y_data.max())],
                'data_summary': {
                    'mean_stint_length': float(x_data.mean()),
                    'max_stint_length': float(x_data.max()),
                    'mean_laptime': float(y_data.mean()),
                    'laptime_std': float(y_data.std()),
                    'stint_count': len(compound_data.groupby(['Driver', 'Stint']))
                }
            }
            
            print(f"      Built {compound} model with {len(compound_data)} observations")
            return model_data
            
        except Exception as e:
            print(f"      Error building {compound} model: {str(e)}")
            return None
    
    def perform_empirical_curve_analysis(self, processed_data, circuit_name):
        """
        Perform empirical 5-point curve analysis on raw stint data
        """
        print(f"    Performing empirical curve analysis for {circuit_name}...")
        
        empirical_curves = {}
        
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            compound_data = processed_data[processed_data['Compound'] == compound]
            
            if len(compound_data) < 30:  # Need sufficient data
                print(f"      Insufficient data for {compound} empirical analysis")
                continue
            
            curve_analysis = self._analyze_five_point_curve(compound_data, compound, circuit_name)
            if curve_analysis:
                empirical_curves[compound] = curve_analysis
        
        return empirical_curves
    
    def _analyze_five_point_curve(self, compound_data, compound, circuit_name):
        """
        Extract 5-point curve parameters from stint data
        """
        try:
            # Point 1: Compound baseline (laps 5-10 median)
            baseline_data = compound_data[
                (compound_data['StintLap'] >= 5) & 
                (compound_data['StintLap'] <= 10)
            ]
            
            if len(baseline_data) > 5:
                compound_baseline = baseline_data['LapTime_FC'].median()
                baseline_std = baseline_data['LapTime_FC'].std()
            else:
                compound_baseline = compound_data['LapTime_FC'].median()
                baseline_std = compound_data['LapTime_FC'].std()
            
            # Point 2: Warmup analysis
            warmup_analysis = self._analyze_warmup_pattern(compound_data, compound_baseline)
            
            # Point 3: Linear deg analysis
            linear_analysis = self._analyze_linear_degradation(compound_data)
            
            # Point 4: Performance cliff detection
            cliff_analysis = self._detect_performance_cliff(compound_data)
            
            # Point 5: Maximum stint analysis
            stint_analysis = self._analyze_stint_lengths(compound_data)
            
            # Compile curve parameters
            curve_params = {
                # Point 1: Baseline performance
                'baseline_laptime': float(compound_baseline),
                'baseline_std': float(baseline_std),
                'baseline_sample_size': len(baseline_data),
                
                # Point 2: Warmup characteristics
                'warmup_penalty': warmup_analysis.get('penalty', 0.0),
                'warmup_laps': warmup_analysis.get('laps', 1),
                'warmup_sample_size': warmup_analysis.get('sample_size', 0),
                'warmup_confidence': warmup_analysis.get('confidence', 0.0),
                
                # Point 3: Linear deg
                'linear_deg_rate': linear_analysis.get('rate', 0.0),
                'linear_r_squared': linear_analysis.get('r_squared', 0.0),
                'linear_p_value': linear_analysis.get('p_value', 1.0),
                'linear_sample_size': linear_analysis.get('sample_size', 0),
                
                # Point 4: Performance cliff
                'cliff_start_lap': cliff_analysis.get('cliff_start_lap', 999),
                'cliff_steepness': cliff_analysis.get('cliff_steepness', 0.0),
                'cliff_confidence': cliff_analysis.get('detection_confidence', 0.0),
                
                # Point 5: Stint length analysis
                'practical_max_stint': stint_analysis.get('practical_maximum', 25),
                'observed_max_stint': stint_analysis.get('absolute_maximum', 25),
                'stint_length_range': stint_analysis.get('observed_range', [1, 25]),
                'stint_count': stint_analysis.get('stint_count', 0),
                
                # Metadata
                'total_laps_analyzed': len(compound_data),
                'driver_count': compound_data['Driver'].nunique(),
                'circuit_name': circuit_name,
                'compound': compound,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
            
            return curve_params
            
        except Exception as e:
            print(f"      Error in curve analysis for {compound}: {e}")
            return None
    
    def _analyze_warmup_pattern(self, compound_data, baseline_laptimes):
        """Analyze tire warmup from actual stint data"""
        warmup_deltas = {}
        
        # Group by driver and stint to analyze warmup patterns
        for (driver, stint), stint_data in compound_data.groupby(['Driver', 'Stint']):
            stint_data = stint_data.sort_values('StintLap')
            
            # Only analyze stints with enough laps
            if len(stint_data) >= 5:
                for _, lap_row in stint_data.iterrows():
                    stint_lap = lap_row['StintLap']
                    if stint_lap <= 5:  # Only first 5 laps
                        laptime = lap_row['LapTime_FC']
                        delta = laptime - baseline_laptimes
                        
                        if stint_lap not in warmup_deltas:
                            warmup_deltas[stint_lap] = []
                        warmup_deltas[stint_lap].append(delta)
        
        if not warmup_deltas:
            return {'penalty': 0, 'laps': 1, 'sample_size': 0, 'confidence': 0}
        
        # Find warmup completion point
        warmup_laps = 1
        max_penalty = 0
        
        for lap in sorted(warmup_deltas.keys()):
            if lap <= 5:
                lap_deltas = warmup_deltas[lap]
                median_delta = np.median(lap_deltas)
                
                if median_delta > 0.1:  # Still significantly slower than baseline
                    warmup_laps = lap
                    max_penalty = max(max_penalty, median_delta)
        
        # Calculate confidence based on sample size and consistency
        total_samples = sum(len(deltas) for deltas in warmup_deltas.values())
        confidence = min(1.0, total_samples / 50.0)  # Full confidence with 50+ samples
        
        return {
            'penalty': max_penalty,
            'laps': warmup_laps,
            'sample_size': total_samples,
            'confidence': confidence
        }
    
    def _analyze_linear_degradation(self, compound_data):
        """Analyze linear degradation in main stint portion"""
        # Focus on middle stint (laps 6-25) for linear analysis
        linear_data = compound_data[
            (compound_data['StintLap'] >= 6) & 
            (compound_data['StintLap'] <= 25)
        ].copy()
        
        if len(linear_data) < 10:
            return {'rate': 0, 'r_squared': 0, 'sample_size': 0, 'p_value': 1.0}
        
        try:
            x = linear_data['StintLap'].values
            y = linear_data['LapTime_FC'].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            return {
                'rate': max(0, slope),  # Deg should be positive
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_error': std_err,
                'sample_size': len(linear_data)
            }
        except Exception as e:
            return {'rate': 0, 'r_squared': 0, 'sample_size': 0, 'p_value': 1.0}
    
    def _detect_performance_cliff(self, compound_data):
        """Detect performance cliff using change point detection"""
        # Calculate median performance per stint lap
        lap_performance = compound_data.groupby('StintLap')['LapTime_FC'].agg(['median', 'count']).reset_index()
        lap_performance = lap_performance[lap_performance['count'] >= 3]  # Need multiple samples
        
        if len(lap_performance) < 10:
            return {'cliff_start_lap': 999, 'cliff_steepness': 0, 'detection_confidence': 0}
        
        laps = lap_performance['StintLap'].values
        times = lap_performance['median'].values
        
        # Calculate deg rate over rolling window
        window_size = 3
        degradation_rates = []
        
        for i in range(window_size, len(times)):
            window_laps = laps[i-window_size:i+1]
            window_times = times[i-window_size:i+1]
            
            try:
                slope, _, _, _, _ = stats.linregress(window_laps, window_times)
                degradation_rates.append((laps[i], slope))
            except:
                continue
        
        if len(degradation_rates) < 5:
            return {'cliff_start_lap': 999, 'cliff_steepness': 0, 'detection_confidence': 0}
        
        # Detect significant increase in deg rate
        rates_df = pd.DataFrame(degradation_rates, columns=['lap', 'rate'])
        
        cliff_lap = 999
        cliff_steepness = 0
        
        for i in range(3, len(rates_df)-2):
            early_rate = rates_df.iloc[:i]['rate'].median()
            late_rate = rates_df.iloc[i:]['rate'].median()
            
            # Look for significant increase in deg
            if late_rate > early_rate * 1.5 and early_rate > 0.005:  # 50% increase, minimum baseline
                cliff_lap = int(rates_df.iloc[i]['lap'])
                cliff_steepness = late_rate - early_rate
                break
        
        confidence = min(1.0, len(degradation_rates) / 15.0)
        
        return {
            'cliff_start_lap': cliff_lap,
            'cliff_steepness': max(0, cliff_steepness),
            'detection_confidence': confidence
        }
    
    def _analyze_stint_lengths(self, compound_data):
        """Analyze stint length patterns"""
        stint_lengths = compound_data.groupby(['Driver', 'Stint'])['StintLap'].max()
        
        if len(stint_lengths) == 0:
            return {'practical_maximum': 25, 'observed_range': [1, 25], 'stint_count': 0}
        
        practical_max = int(np.percentile(stint_lengths, 90))  # 90th percentile
        absolute_max = int(stint_lengths.max())
        observed_min = int(stint_lengths.min())
        
        return {
            'practical_maximum': practical_max,
            'absolute_maximum': absolute_max,
            'observed_range': [observed_min, absolute_max],
            'stint_count': len(stint_lengths),
            'mean_stint_length': float(stint_lengths.mean()),
            'median_stint_length': float(stint_lengths.median())
        }
    
    def save_comprehensive_circuit_data(self, circuit_name, processed_data, bayesian_models, empirical_curves):
        """Save comprehensive circuit data including raw stint data"""
        circuit_file = self.output_dir / f"{circuit_name.lower().replace(' ', '_')}_models.pkl"
        
        # Prepare processed data for storage (sample for size management)
        if len(processed_data) > 1000:
            # Save a representative sample + all key analysis data
            sampled_data = processed_data.sample(n=1000, random_state=42)
            data_storage = {
                'sample_data': sampled_data.to_dict('records'),
                'full_data_summary': {
                    'total_rows': len(processed_data),
                    'compounds': processed_data['Compound'].value_counts().to_dict(),
                    'drivers': processed_data['Driver'].unique().tolist(),
                    'stint_lap_range': [int(processed_data['StintLap'].min()), int(processed_data['StintLap'].max())],
                    'laptime_range': [float(processed_data['LapTime_FC'].min()), float(processed_data['LapTime_FC'].max())]
                }
            }
        else:
            # Save all data if dataset is small
            data_storage = {
                'full_data': processed_data.to_dict('records'),
                'full_data_summary': {
                    'total_rows': len(processed_data),
                    'compounds': processed_data['Compound'].value_counts().to_dict(),
                    'drivers': processed_data['Driver'].unique().tolist(),
                    'stint_lap_range': [int(processed_data['StintLap'].min()), int(processed_data['StintLap'].max())],
                    'laptime_range': [float(processed_data['LapTime_FC'].min()), float(processed_data['LapTime_FC'].max())]
                }
            }
        
        circuit_data = {
            'circuit_name': circuit_name,
            'bayesian_models': bayesian_models,
            'empirical_curves': empirical_curves,
            'stint_data': data_storage,
            'analysis_metadata': {
                'total_laps_processed': len(processed_data),
                'compounds_analyzed': list(processed_data['Compound'].unique()),
                'drivers_count': processed_data['Driver'].nunique(),
                'bayesian_model_count': len(bayesian_models),
                'empirical_curve_count': len(empirical_curves),
                'data_quality_score': self._calculate_data_quality_score(processed_data, empirical_curves)
            },
            'created_timestamp': pd.Timestamp.now().isoformat(),
            'model_version': '2.0_with_empirical_analysis'
        }
        
        with open(circuit_file, 'wb') as f:
            pickle.dump(circuit_data, f)
        
        print(f"  Saved comprehensive data to {circuit_file}")
        return circuit_data
    
    def _calculate_data_quality_score(self, processed_data, empirical_curves):
        """Calculate overall data quality score (0-1)"""
        quality_factors = []
        
        # Sample size factor
        total_laps = len(processed_data)
        size_score = min(1.0, total_laps / 500.0)  # Full score at 500+ laps
        quality_factors.append(size_score)
        
        # Compound coverage factor
        compounds_available = len(processed_data['Compound'].unique())
        compound_score = compounds_available / 3.0  # Full score with all 3 compounds
        quality_factors.append(compound_score)
        
        # Stint length diversity factor
        stint_lengths = processed_data.groupby(['Driver', 'Stint'])['StintLap'].max()
        if len(stint_lengths) > 0:
            length_diversity = len(stint_lengths[stint_lengths >= 10]) / max(1, len(stint_lengths))
            quality_factors.append(length_diversity)
        
        # Empirical analysis success factor
        if empirical_curves:
            analysis_success = len(empirical_curves) / 3.0  # Full score with all 3 compounds
            quality_factors.append(analysis_success)
        
        return float(np.mean(quality_factors))
    
    def prebuild_comprehensive_models(self):
        """Build comprehensive models with both Bayesian and empirical analysis"""
        print("Starting comprehensive F1 tire model building process...")
        print(f"Output directory: {self.output_dir.absolute()}")
        
        success_count = 0
        total_bayesian_models = 0
        total_empirical_curves = 0
        
        for circuit_name, circuit_info in self.circuits.items():
            print(f"\nProcessing {circuit_name} ({circuit_info['gp_name']})...")
            
            # Load and process data with full detail preservation
            processed_data = self.load_and_process_f1_data(circuit_name)
            
            if processed_data is None:
                print(f"  Skipping {circuit_name} - no data available")
                continue
            
            # Build Bayesian models
            bayesian_models = {}
            for compound in ['SOFT', 'MEDIUM', 'HARD']:
                compound_data = processed_data[processed_data['Compound'] == compound]
                if len(compound_data) > 15:  # Minimum for Bayesian model
                    model = self.build_bayesian_tire_model(compound_data, circuit_name, compound)
                    if model is not None:
                        bayesian_models[compound] = model
                        total_bayesian_models += 1
            
            # Perform empirical curve analysis
            empirical_curves = self.perform_empirical_curve_analysis(processed_data, circuit_name)
            total_empirical_curves += len(empirical_curves)
            
            # Save comprehensive data
            self.save_comprehensive_circuit_data(
                circuit_name, processed_data, bayesian_models, empirical_curves
            )
            
            success_count += 1
            print(f"  ✅ Built {len(bayesian_models)} Bayesian + {len(empirical_curves)} empirical models")
        
        print(f"\n🏁 Comprehensive model building complete!")
        print(f"Successfully processed {success_count}/{len(self.circuits)} circuits")
        print(f"Built {total_bayesian_models} Bayesian models")
        print(f"Built {total_empirical_curves} empirical curve sets")
        print(f"Models saved in: {self.output_dir.absolute()}")
        
        # Create summary
        self.create_comprehensive_summary(success_count, total_bayesian_models, total_empirical_curves)
    
    def create_comprehensive_summary(self, success_count, total_bayesian, total_empirical):
        """Create comprehensive summary of all built models"""
        summary = {
            'build_timestamp': pd.Timestamp.now().isoformat(),
            'model_version': '2.0_with_empirical_analysis',
            'circuits_processed': success_count,
            'total_circuits': len(self.circuits),
            'total_bayesian_models': total_bayesian,
            'total_empirical_curves': total_empirical,
            'model_files': [],
            'quality_metrics': {
                'avg_data_quality': 0,
                'high_quality_circuits': 0,
                'medium_quality_circuits': 0,
                'low_quality_circuits': 0
            }
        }
        
        quality_scores = []
        
        # Analyze each model file
        for model_file in self.output_dir.glob("*_models.pkl"):
            try:
                with open(model_file, 'rb') as f:
                    data = pickle.load(f)
                
                file_info = {
                    'filename': model_file.name,
                    'circuit': data['circuit_name'],
                    'bayesian_compounds': list(data.get('bayesian_models', {}).keys()),
                    'empirical_compounds': list(data.get('empirical_curves', {}).keys()),
                    'model_version': data.get('model_version', '1.0'),
                    'data_quality_score': data.get('analysis_metadata', {}).get('data_quality_score', 0)
                }
                
                # Add detailed metrics if available
                if 'analysis_metadata' in data:
                    metadata = data['analysis_metadata']
                    file_info.update({
                        'total_laps': metadata.get('total_laps_processed', 0),
                        'driver_count': metadata.get('drivers_count', 0),
                        'compounds_available': metadata.get('compounds_analyzed', [])
                    })
                
                summary['model_files'].append(file_info)
                
                # Track quality metrics
                quality_score = file_info['data_quality_score']
                quality_scores.append(quality_score)
                
                if quality_score > 0.8:
                    summary['quality_metrics']['high_quality_circuits'] += 1
                elif quality_score > 0.5:
                    summary['quality_metrics']['medium_quality_circuits'] += 1
                else:
                    summary['quality_metrics']['low_quality_circuits'] += 1
                    
            except Exception as e:
                print(f"Error reading {model_file}: {e}")
                continue
        
        if quality_scores:
            summary['quality_metrics']['avg_data_quality'] = float(np.mean(quality_scores))
        
        summary_file = self.output_dir / "comprehensive_models_summary.pkl"
        with open(summary_file, 'wb') as f:
            pickle.dump(summary, f)
        
        print(f"Comprehensive summary saved to: {summary_file}")
        
        # Print quality summary
        print(f"\nData Quality Summary:")
        print(f"  Average quality score: {summary['quality_metrics']['avg_data_quality']:.2f}")
        print(f"  High quality circuits: {summary['quality_metrics']['high_quality_circuits']}")
        print(f"  Medium quality circuits: {summary['quality_metrics']['medium_quality_circuits']}")
        print(f"  Low quality circuits: {summary['quality_metrics']['low_quality_circuits']}")

class EmpiricalTireAnalyzer:
    """
    Loads and analyzes the comprehensive model data for empirical tire performance
    """
    
    def __init__(self, models_dir="prebuilt_models"):
        self.models_dir = Path(models_dir)
        self.circuit_data = {}
        self.empirical_curves = {}
        self.bayesian_models = {}
        self._load_comprehensive_data()
    
    def _load_comprehensive_data(self):
        """Load all comprehensive model data"""
        if not self.models_dir.exists():
            print("Models directory not found")
            return
        
        for model_file in self.models_dir.glob("*_models.pkl"):
            try:
                with open(model_file, 'rb') as f:
                    data = pickle.load(f)
                
                circuit_name = data['circuit_name']
                self.circuit_data[circuit_name] = data
                
                # Extract empirical curves
                if 'empirical_curves' in data:
                    self.empirical_curves[circuit_name] = data['empirical_curves']
                
                # Extract Bayesian models
                if 'bayesian_models' in data:
                    for compound, model_data in data['bayesian_models'].items():
                        model_key = f"{circuit_name}_{compound}"
                        self.bayesian_models[model_key] = model_data
                
                print(f"Loaded comprehensive data for {circuit_name}")
                
            except Exception as e:
                print(f"Error loading {model_file}: {e}")
    
    def calculate_empirical_tire_performance(self, compound, stint_lap, tire_age, base_laptime, circuit_name):
        """
        Calculate tire performance using empirical 5-point curves
        """
        if circuit_name not in self.empirical_curves:
            return self._fallback_bayesian_calculation(compound, stint_lap, tire_age, base_laptime, circuit_name)
        
        if compound not in self.empirical_curves[circuit_name]:
            return self._fallback_bayesian_calculation(compound, stint_lap, tire_age, base_laptime, circuit_name)
        
        curve = self.empirical_curves[circuit_name][compound]
        effective_lap = stint_lap + tire_age
        
        # Start with base laptime
        laptime = base_laptime
        
        # Point 1: Use empirical baseline adjustment
        if 'baseline_laptime' in curve:
            # Adjust base laptime to match empirical baseline
            baseline_adjustment = curve['baseline_laptime'] - base_laptime
            # Only apply if the adjustment is reasonable (within 5 seconds)
            if abs(baseline_adjustment) < 5.0:
                laptime = curve['baseline_laptime']
        
        # Point 2: Warmup penalty
        if stint_lap <= curve.get('warmup_laps', 1):
            warmup_factor = 1 - (stint_lap - 1) / max(1, curve.get('warmup_laps', 1))
            laptime += curve.get('warmup_penalty', 0) * warmup_factor
        
        # Point 3: Linear deg
        linear_rate = curve.get('linear_deg_rate', 0)
        laptime += linear_rate * effective_lap
        
        # Point 4: Performance cliff
        cliff_start = curve.get('cliff_start_lap', 999)
        if effective_lap > cliff_start:
            cliff_laps = effective_lap - cliff_start
            cliff_steepness = curve.get('cliff_steepness', 0)
            laptime += cliff_steepness * (cliff_laps ** 1.5)
        
        # Point 5: Maximum stint penalty
        max_stint = curve.get('practical_max_stint', 30)
        if effective_lap > max_stint:
            danger_laps = effective_lap - max_stint
            laptime += 1.0 * danger_laps  # Linear penalty for exceeding practical max
        
        return laptime
    
    def _fallback_bayesian_calculation(self, compound, stint_lap, tire_age, base_laptime, circuit_name):
        """Fallback to Bayesian model if empirical curves unavailable"""
        model_key = f"{circuit_name}_{compound}"
        
        if model_key not in self.bayesian_models:
            # Final fallback - simple deg
            return base_laptime + 0.02 * (stint_lap + tire_age)
        
        model_data = self.bayesian_models[model_key]
        samples = model_data['samples']
        
        effective_stint_lap = stint_lap + tire_age
        
        # Sample from posterior
        sample_idx = np.random.choice(len(samples['alpha']))
        alpha_sample = float(samples['alpha'][sample_idx])
        beta_sample = float(samples['beta'][sample_idx])
        sigma_sample = float(samples['sigma'][sample_idx])
        
        # Calculate prediction
        mu = alpha_sample + beta_sample * effective_stint_lap
        prediction = mu + np.random.normal(0, sigma_sample)
        
        return float(prediction)
    
    def get_curve_quality_info(self, circuit_name, compound):
        """Get quality information for empirical curves"""
        if circuit_name not in self.empirical_curves:
            return None
        
        if compound not in self.empirical_curves[circuit_name]:
            return None
        
        curve = self.empirical_curves[circuit_name][compound]
        
        quality_info = {
            'total_laps_analyzed': curve.get('total_laps_analyzed', 0),
            'driver_count': curve.get('driver_count', 0),
            'warmup_confidence': curve.get('warmup_confidence', 0),
            'linear_r_squared': curve.get('linear_r_squared', 0),
            'linear_p_value': curve.get('linear_p_value', 1),
            'cliff_confidence': curve.get('cliff_confidence', 0),
            'stint_count': curve.get('stint_count', 0),
            'has_bayesian_fallback': f"{circuit_name}_{compound}" in self.bayesian_models
        }
        
        # Calculate overall quality score
        quality_factors = []
        
        # Sample size quality
        if quality_info['total_laps_analyzed'] > 100:
            quality_factors.append(1.0)
        elif quality_info['total_laps_analyzed'] > 50:
            quality_factors.append(0.7)
        else:
            quality_factors.append(0.3)
        
        # Linear model quality
        if quality_info['linear_r_squared'] > 0.7 and quality_info['linear_p_value'] < 0.05:
            quality_factors.append(1.0)
        elif quality_info['linear_r_squared'] > 0.4:
            quality_factors.append(0.6)
        else:
            quality_factors.append(0.2)
        
        # Analysis completeness
        analysis_completeness = sum([
            1 if curve.get('warmup_confidence', 0) > 0.5 else 0,
            1 if curve.get('cliff_confidence', 0) > 0.5 else 0,
            1 if curve.get('stint_count', 0) > 5 else 0
        ]) / 3.0
        quality_factors.append(analysis_completeness)
        
        quality_info['overall_quality_score'] = np.mean(quality_factors)
        
        return quality_info
    
    def get_available_circuits_and_compounds(self):
        """Get list of available circuits and compounds with quality info"""
        availability = {}
        
        for circuit_name in self.empirical_curves.keys():
            circuit_info = {
                'empirical_compounds': [],
                'bayesian_compounds': [],
                'quality_scores': {}
            }
            
            # Check empirical curves
            for compound in ['SOFT', 'MEDIUM', 'HARD']:
                if compound in self.empirical_curves[circuit_name]:
                    circuit_info['empirical_compounds'].append(compound)
                    quality = self.get_curve_quality_info(circuit_name, compound)
                    if quality:
                        circuit_info['quality_scores'][compound] = quality['overall_quality_score']
                
                # Check Bayesian models
                model_key = f"{circuit_name}_{compound}"
                if model_key in self.bayesian_models:
                    circuit_info['bayesian_compounds'].append(compound)
            
            availability[circuit_name] = circuit_info
        
        return availability
    
    def export_curve_parameters(self, circuit_name, compound):
        """Export curve parameters for external analysis"""
        if circuit_name not in self.empirical_curves:
            return None
        
        if compound not in self.empirical_curves[circuit_name]:
            return None
        
        curve = self.empirical_curves[circuit_name][compound]
        quality = self.get_curve_quality_info(circuit_name, compound)
        
        export_data = {
            'circuit': circuit_name,
            'compound': compound,
            'curve_parameters': curve,
            'quality_metrics': quality,
            'export_timestamp': pd.Timestamp.now().isoformat()
        }
        
        return export_data

if __name__ == "__main__":
    if not DEPENDENCIES_AVAILABLE:
        print("Required dependencies not available!")
        print("Please install: pip install fastf1 jax numpyro")
        exit(1)
    
    # Build comprehensive models
    prebuilder = F1ModelPrebuilder()
    prebuilder.prebuild_comprehensive_models()
    
    print("\n" + "="*50)
    print("Testing empirical analyzer...")
    
    # Test the empirical analyzer
    analyzer = EmpiricalTireAnalyzer()
    
    # Show availability
    availability = analyzer.get_available_circuits_and_compounds()
    
    print(f"\nAvailable circuits and compounds:")
    for circuit, info in availability.items():
        print(f"\n{circuit}:")
        print(f"  Empirical: {info['empirical_compounds']}")
        print(f"  Bayesian: {info['bayesian_compounds']}")
        if info['quality_scores']:
            avg_quality = np.mean(list(info['quality_scores'].values()))
            print(f"  Avg Quality: {avg_quality:.2f}")
    
    # Test calculation if data available
    if availability:
        test_circuit = list(availability.keys())[0]
        test_compound = 'SOFT'
        
        if test_compound in availability[test_circuit]['empirical_compounds']:
            print(f"\nTesting {test_compound} at {test_circuit}:")
            
            for lap in [1, 5, 10, 15, 20]:
                laptime = analyzer.calculate_empirical_tire_performance(
                    test_compound, lap, 0, 80.0, test_circuit
                )
                print(f"  Lap {lap}: {laptime:.3f}s")
            
            # Show quality info
            quality = analyzer.get_curve_quality_info(test_circuit, test_compound)
            if quality:
                print(f"\nQuality metrics:")
                print(f"  Overall score: {quality['overall_quality_score']:.2f}")
                print(f"  Sample size: {quality['total_laps_analyzed']}")
                print(f"  Linear R²: {quality['linear_r_squared']:.3f}")
                print(f"  Warmup confidence: {quality['warmup_confidence']:.2f}")
                print(f"  Cliff confidence: {quality['cliff_confidence']:.2f}")
                

class EmpiricalTireAnalyzer:
    """
    Loads and analyzes the comprehensive model data for empirical tire performance
    """
    
    def __init__(self, models_dir="prebuilt_models"):
        self.models_dir = Path(models_dir)
        self.circuit_data = {}
        self.empirical_curves = {}
        self.bayesian_models = {}
        self._load_comprehensive_data()
    
    def _load_comprehensive_data(self):
        """Load all comprehensive model data"""
        if not self.models_dir.exists():
            print("Models directory not found")
            return
        
        for model_file in self.models_dir.glob("*_models.pkl"):
            try:
                with open(model_file, 'rb') as f:
                    data = pickle.load(f)
                
                circuit_name = data['circuit_name']
                self.circuit_data[circuit_name] = data
                
                # Extract empirical curves
                if 'empirical_curves' in data:
                    self.empirical_curves[circuit_name] = data['empirical_curves']
                
                # Extract Bayesian models
                if 'bayesian_models' in data:
                    for compound, model_data in data['bayesian_models'].items():
                        model_key = f"{circuit_name}_{compound}"
                        self.bayesian_models[model_key] = model_data
                
                print(f"Loaded comprehensive data for {circuit_name}")
                
            except Exception as e:
                print(f"Error loading {model_file}: {e}")
    
    def calculate_empirical_tire_performance(self, compound, stint_lap, tire_age, base_laptime, circuit_name):
        """Calculate tire performance using empirical 5-point curves"""
        if circuit_name not in self.empirical_curves:
            return None
        
        if compound not in self.empirical_curves[circuit_name]:
            return None
        
        curve = self.empirical_curves[circuit_name][compound]
        effective_lap = stint_lap + tire_age
        
        # Start with base laptime
        laptime = base_laptime
        
        # Point 1: Use empirical baseline adjustment
        if 'baseline_laptime' in curve:
            baseline_adjustment = curve['baseline_laptime'] - base_laptime
            if abs(baseline_adjustment) < 5.0:
                laptime = curve['baseline_laptime']
        
        # Point 2: Warmup penalty
        if stint_lap <= curve.get('warmup_laps', 1):
            warmup_factor = 1 - (stint_lap - 1) / max(1, curve.get('warmup_laps', 1))
            laptime += curve.get('warmup_penalty', 0) * warmup_factor
        
        # Point 3: Linear deg
        linear_rate = curve.get('linear_deg_rate', 0)
        laptime += linear_rate * effective_lap
        
        # Point 4: Performance cliff
        cliff_start = curve.get('cliff_start_lap', 999)
        if effective_lap > cliff_start:
            cliff_laps = effective_lap - cliff_start
            cliff_steepness = curve.get('cliff_steepness', 0)
            laptime += cliff_steepness * (cliff_laps ** 1.5)
        
        # Point 5: Maximum stint penalty
        max_stint = curve.get('practical_max_stint', 30)
        if effective_lap > max_stint:
            danger_laps = effective_lap - max_stint
            laptime += 1.0 * danger_laps
        
        return laptime
    
    def get_curve_quality_info(self, circuit_name, compound):
        """Get quality information for empirical curves"""
        if circuit_name not in self.empirical_curves:
            return None
        
        if compound not in self.empirical_curves[circuit_name]:
            return None
        
        curve = self.empirical_curves[circuit_name][compound]
        
        quality_info = {
            'total_laps_analyzed': curve.get('total_laps_analyzed', 0),
            'driver_count': curve.get('driver_count', 0),
            'warmup_confidence': curve.get('warmup_confidence', 0),
            'linear_r_squared': curve.get('linear_r_squared', 0),
            'linear_p_value': curve.get('linear_p_value', 1),
            'cliff_confidence': curve.get('cliff_confidence', 0),
            'stint_count': curve.get('stint_count', 0),
            'has_bayesian_fallback': f"{circuit_name}_{compound}" in self.bayesian_models
        }
        
        # Calculate overall quality score
        quality_factors = []
        
        # Sample size quality
        if quality_info['total_laps_analyzed'] > 100:
            quality_factors.append(1.0)
        elif quality_info['total_laps_analyzed'] > 50:
            quality_factors.append(0.7)
        else:
            quality_factors.append(0.3)
        
        # Linear model quality
        if quality_info['linear_r_squared'] > 0.7 and quality_info['linear_p_value'] < 0.05:
            quality_factors.append(1.0)
        elif quality_info['linear_r_squared'] > 0.4:
            quality_factors.append(0.6)
        else:
            quality_factors.append(0.2)
        
        # Analysis completeness
        analysis_completeness = sum([
            1 if curve.get('warmup_confidence', 0) > 0.5 else 0,
            1 if curve.get('cliff_confidence', 0) > 0.5 else 0,
            1 if curve.get('stint_count', 0) > 5 else 0
        ]) / 3.0
        quality_factors.append(analysis_completeness)
        
        quality_info['overall_quality_score'] = np.mean(quality_factors)
        
        return quality_info
    
    def get_available_circuits_and_compounds(self):
        """Get list of available circuits and compounds with quality info"""
        availability = {}
        
        for circuit_name in self.empirical_curves.keys():
            circuit_info = {
                'empirical_compounds': [],
                'bayesian_compounds': [],
                'quality_scores': {}
            }
            
            # Check empirical curves
            for compound in ['SOFT', 'MEDIUM', 'HARD']:
                if compound in self.empirical_curves[circuit_name]:
                    circuit_info['empirical_compounds'].append(compound)
                    quality = self.get_curve_quality_info(circuit_name, compound)
                    if quality:
                        circuit_info['quality_scores'][compound] = quality['overall_quality_score']
                
                # Check Bayesian models
                model_key = f"{circuit_name}_{compound}"
                if model_key in self.bayesian_models:
                    circuit_info['bayesian_compounds'].append(compound)
            
            availability[circuit_name] = circuit_info
        
        return availability
    
    def export_curve_parameters(self, circuit_name, compound):
        """Export curve parameters for external analysis"""
        if circuit_name not in self.empirical_curves:
            return None
        
        if compound not in self.empirical_curves[circuit_name]:
            return None
        
        curve = self.empirical_curves[circuit_name][compound]
        quality = self.get_curve_quality_info(circuit_name, compound)
        
        export_data = {
            'circuit': circuit_name,
            'compound': compound,
            'curve_parameters': curve,
            'quality_metrics': quality,
            'export_timestamp': pd.Timestamp.now().isoformat()
        }
        
        return export_data