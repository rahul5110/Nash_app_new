import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import pandas as pd
from scipy.optimize import minimize, differential_evolution, dual_annealing
import io

# Page configuration
st.set_page_config(
    page_title="Nash Unit Hydrograph Analysis",
    page_icon="💧",
    layout="wide"
)

# Title and description
st.title("💧 Nash Unit Hydrograph Analysis")
st.markdown("""
Upload your rainfall-runoff data to calculate Nash IUH parameters using multiple optimization methods:
- **Method of Moments**
- **L-BFGS-B Optimization**
- **Differential Evolution**
- **Genetic Algorithm (Dual Annealing)**
""")

# ================== MOMENTS DISCRETE FUNCTION ==================
def moments_discrete(t, w):
    """Calculate the first moment (mean) and second central moment (variance)"""
    if np.sum(w) == 0:
        return np.nan, np.nan
    m1 = np.sum(t * w) / np.sum(w)
    var = np.sum((t - m1)**2 * w) / np.sum(w)
    return m1, var

# ================== NASH DRH CALCULATION FUNCTION ==================
def calculate_nash_drh(rainfall_excess_mm, area_ha, dt_hr, n, k, output_length):
    """Calculates the Nash Unit Hydrograph and convolves it with rainfall excess"""
    A_m2 = area_ha * 10000.0
    
    if n <= 0 or k <= 0:
        return np.zeros(output_length) * np.nan
    
    t_iuh = np.arange(output_length) * dt_hr
    
    if k > 1e-9:
        iuh = (area_ha / 360.0) * (t_iuh / k)**(n - 1) * np.exp(-t_iuh / k) / (k * gamma(n))
    else:
        return np.zeros(output_length) * np.nan
    
    iuh[~np.isfinite(iuh)] = 0
    Q_est = np.convolve(rainfall_excess_mm, iuh)[:output_length]
    
    if len(Q_est) < output_length:
        Q_est = np.pad(Q_est, (0, output_length - len(Q_est)), 'constant')
    elif len(Q_est) > output_length:
        Q_est = Q_est[:output_length]
    
    Q_est[~np.isfinite(Q_est)] = 0
    return Q_est

# ================== OPTIMIZATION OBJECTIVE FUNCTIONS ==================
def objective_rmse(params, Q_obs, rainfall_excess_mm, area_ha, dt_hr):
    """Objective function to minimize (RMSE)"""
    n, k = params
    
    if n <= 0.01 or k <= 0.01:
        return np.inf
    
    try:
        Q_est = calculate_nash_drh(rainfall_excess_mm, area_ha, dt_hr, n, k, len(Q_obs))
        min_len = min(len(Q_obs), len(Q_est))
        Q_obs_trimmed = Q_obs[:min_len]
        Q_est_trimmed = Q_est[:min_len]
        rmse = np.sqrt(np.mean((Q_obs_trimmed - Q_est_trimmed)**2))
        
        if not np.isfinite(rmse):
            return np.inf
        return rmse
    except Exception as e:
        return np.inf

# ================== MAIN APP ==================
# Sidebar for file upload and parameters
with st.sidebar:
    st.header("⚙️ Settings")
    
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help="File must contain: Event_ID, Time step, Q_obs_event, rain_mm_interval"
    )
    
    area_ha = st.number_input(
        "Catchment Area (hectares)",
        min_value=1.0,
        value=645.0,
        step=10.0
    )
    
    st.markdown("---")
    st.markdown("### Analysis Methods")
    run_moments = st.checkbox("Method of Moments", value=True)
    run_lbfgs = st.checkbox("L-BFGS-B Optimization", value=True)
    run_de = st.checkbox("Differential Evolution", value=True)
    run_ga = st.checkbox("Genetic Algorithm", value=True)

# Initialize session state
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'event_results' not in st.session_state:
    st.session_state.event_results = {}

# Main content
if uploaded_file is not None:
    try:
        # Read CSV
        all_events_df = pd.read_csv(uploaded_file)
        
        # Validate columns
        required_cols = ['Event_ID', 'Time step', 'Q_obs_event', 'rain_mm_interval']
        if not all(col in all_events_df.columns for col in required_cols):
            st.error(f"❌ CSV file must contain the following columns: {required_cols}")
        else:
            # Get unique event IDs
            unique_event_ids = all_events_df['Event_ID'].unique()
            
            st.success(f"✅ File uploaded successfully! Found {len(unique_event_ids)} events.")
            
            # Process button
            if st.button("🚀 Run Analysis", type="primary"):
                # Initialize results DataFrame
                results_columns = [
                    'Event ID', 'Total Rainfall Excess Depth (mm)',
                    'Volume from Rainfall Excess (m³)', 'Observed Runoff Volume (m³)',
                    'Observed Peak Discharge (m³/s)', 'Observed Time to Peak (hr)',
                    'Observed Time of Concentration (hr)', 'Observed Lag Time (hr)'
                ]
                
                methods = []
                if run_moments:
                    methods.append('Moments')
                if run_lbfgs:
                    methods.append('L-BFGS-B')
                if run_de:
                    methods.append('Differential Evolution')
                if run_ga:
                    methods.append('Genetic Algorithm')
                
                for method in methods:
                    results_columns.extend([
                        f'Estimated ({method}) n',
                        f'Estimated ({method}) k (hr)',
                        f'Estimated ({method}) RMSE',
                        f'Estimated ({method}) NSE',
                        f'Estimated ({method}) Peak Discharge (m³/s)',
                        f'Estimated ({method}) Time to Peak (hr)',
                        f'Estimated ({method}) Runoff Volume (m³)'
                    ])
                
                df = pd.DataFrame(columns=results_columns)
                event_results = {}
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process each event
                for idx, event_id in enumerate(unique_event_ids):
                    status_text.text(f"Processing Event {event_id}...")
                    
                    event_data = all_events_df[all_events_df['Event_ID'] == event_id].copy()
                    Q_obs_event = event_data['Q_obs_event'].values
                    rainfall_excess_mm = event_data['rain_mm_interval'].values
                    dt_hr = event_data['Time step'].iloc[1] - event_data['Time step'].iloc[0] if len(event_data['Time step']) > 1 else 0.25
                    
                    # Calculate moments
                    t_q = (np.arange(len(Q_obs_event)) + 0.5) * dt_hr
                    t_r = (np.arange(len(rainfall_excess_mm)) + 0.5) * dt_hr
                    w_q = Q_obs_event * (3600.0 * dt_hr)
                    w_r = rainfall_excess_mm.copy()
                    
                    result_row = {'Event ID': event_id}
                    
                    if len(t_q) > 0 and np.sum(w_q) > 0 and len(t_r) > 0 and np.sum(w_r) > 0:
                        m1_q, var_q = moments_discrete(t_q, w_q)
                        m1_r, var_r = moments_discrete(t_r, w_r)
                        m1_u = m1_q - m1_r
                        var_u = var_q - var_r
                    else:
                        m1_u, var_u = np.nan, np.nan
                    
                    # Calculate observed characteristics
                    depth_mm = np.sum(rainfall_excess_mm)
                    A_m2 = area_ha * 10_000.0
                    Vol_from_rain = A_m2 * (depth_mm / 1000.0)
                    Vol_obs = np.sum(Q_obs_event * 3600.0 * dt_hr)
                    peak_q_obs = np.max(Q_obs_event)
                    time_to_peak_obs = np.argmax(Q_obs_event) * dt_hr
                    peak_rain_idx = np.argmax(rainfall_excess_mm) if np.max(rainfall_excess_mm) > 1e-6 else 0
                    
                    if np.sum(rainfall_excess_mm) > 0:
                        centroid_rain_time = np.sum(t_r * rainfall_excess_mm) / np.sum(rainfall_excess_mm)
                        centroid_rain_idx_approx = np.argmin(np.abs(t_r - centroid_rain_time))
                    else:
                        centroid_rain_idx_approx = 0
                    
                    time_of_concentration_obs = t_q[np.argmax(Q_obs_event)] - t_r[peak_rain_idx] if len(t_q) > 0 and len(t_r) > 0 else np.nan
                    time_to_lag_obs = t_q[np.argmax(Q_obs_event)] - t_r[centroid_rain_idx_approx] if len(t_q) > 0 and len(t_r) > 0 else np.nan
                    
                    result_row.update({
                        'Total Rainfall Excess Depth (mm)': depth_mm,
                        'Volume from Rainfall Excess (m³)': Vol_from_rain,
                        'Observed Runoff Volume (m³)': Vol_obs,
                        'Observed Peak Discharge (m³/s)': peak_q_obs,
                        'Observed Time to Peak (hr)': time_to_peak_obs,
                        'Observed Time of Concentration (hr)': time_of_concentration_obs,
                        'Observed Lag Time (hr)': time_to_lag_obs
                    })
                    
                    # Store data for plotting
                    event_results[event_id] = {
                        'Q_obs': Q_obs_event,
                        'rainfall': rainfall_excess_mm,
                        'dt': dt_hr,
                        't_q': t_q,
                        't_r': t_r,
                        'estimates': {}
                    }
                    
                    # Check for valid moments
                    if np.isfinite(m1_u) and np.isfinite(var_u) and m1_u > 0 and var_u >= 0:
                        n_moments = (m1_u ** 2) / var_u if var_u > 1e-9 else np.nan
                        k_moments = m1_u / n_moments if n_moments > 1e-9 else np.nan
                        
                        # Process each method
                        optimization_results = {}
                        
                        if run_moments and np.isfinite(n_moments) and np.isfinite(k_moments):
                            Q_est_moments = calculate_nash_drh(rainfall_excess_mm, area_ha, dt_hr, n_moments, k_moments, len(Q_obs_event))
                            
                            # Calculate metrics
                            min_len = min(len(Q_obs_event), len(Q_est_moments))
                            Q_obs_trimmed = Q_obs_event[:min_len]
                            Q_est_trimmed = Q_est_moments[:min_len]
                            
                            rmse = np.sqrt(np.mean((Q_obs_trimmed - Q_est_trimmed)**2))
                            denominator = np.sum((Q_obs_trimmed - np.mean(Q_obs_trimmed))**2)
                            nse = 1 - (np.sum((Q_obs_trimmed - Q_est_trimmed)**2) / denominator) if denominator != 0 else np.nan
                            
                            Vol_est = np.sum(Q_est_moments * 3600.0 * dt_hr)
                            peak_q_est = np.max(Q_est_moments)
                            time_to_peak_est = np.argmax(Q_est_moments) * dt_hr
                            
                            result_row.update({
                                'Estimated (Moments) n': n_moments,
                                'Estimated (Moments) k (hr)': k_moments,
                                'Estimated (Moments) RMSE': rmse,
                                'Estimated (Moments) NSE': nse,
                                'Estimated (Moments) Peak Discharge (m³/s)': peak_q_est,
                                'Estimated (Moments) Time to Peak (hr)': time_to_peak_est,
                                'Estimated (Moments) Runoff Volume (m³)': Vol_est
                            })
                            
                            event_results[event_id]['estimates']['Moments'] = {
                                'Q_est': Q_est_moments,
                                'n': n_moments,
                                'k': k_moments
                            }
                        
                        # Other optimization methods
                        if any([run_lbfgs, run_de, run_ga]) and np.isfinite(n_moments) and np.isfinite(k_moments):
                            initial_guess = [n_moments, k_moments]
                            bounds = [(0.01, None), (0.01, None)]
                            bounds_de_ga = [(0.01, 20), (0.01, 50)]
                            
                            if run_lbfgs:
                                result_lbfgs = minimize(
                                    objective_rmse,
                                    initial_guess,
                                    args=(Q_obs_event, rainfall_excess_mm, area_ha, dt_hr),
                                    method='L-BFGS-B',
                                    bounds=bounds
                                )
                                optimization_results['L-BFGS-B'] = {'n': result_lbfgs.x[0], 'k': result_lbfgs.x[1]}
                            
                            if run_de:
                                result_de = differential_evolution(
                                    objective_rmse,
                                    bounds_de_ga,
                                    args=(Q_obs_event, rainfall_excess_mm, area_ha, dt_hr),
                                    seed=42,
                                    maxiter=300,
                                    atol=1e-4,
                                    tol=1e-4
                                )
                                optimization_results['Differential Evolution'] = {'n': result_de.x[0], 'k': result_de.x[1]}
                            
                            if run_ga:
                                result_ga = dual_annealing(
                                    objective_rmse,
                                    bounds_de_ga,
                                    args=(Q_obs_event, rainfall_excess_mm, area_ha, dt_hr),
                                    seed=42,
                                    maxiter=300
                                )
                                optimization_results['Genetic Algorithm'] = {'n': result_ga.x[0], 'k': result_ga.x[1]}
                            
                            # Process optimization results
                            for method in optimization_results:
                                n_opt = optimization_results[method]['n']
                                k_opt = optimization_results[method]['k']
                                
                                Q_est_opt = calculate_nash_drh(rainfall_excess_mm, area_ha, dt_hr, n_opt, k_opt, len(Q_obs_event))
                                
                                min_len = min(len(Q_obs_event), len(Q_est_opt))
                                Q_obs_trimmed = Q_obs_event[:min_len]
                                Q_est_trimmed = Q_est_opt[:min_len]
                                
                                rmse = np.sqrt(np.mean((Q_obs_trimmed - Q_est_trimmed)**2))
                                denominator = np.sum((Q_obs_trimmed - np.mean(Q_obs_trimmed))**2)
                                nse = 1 - (np.sum((Q_obs_trimmed - Q_est_trimmed)**2) / denominator) if denominator != 0 else np.nan
                                
                                Vol_est = np.sum(Q_est_opt * 3600.0 * dt_hr)
                                peak_q_est = np.max(Q_est_opt)
                                time_to_peak_est = np.argmax(Q_est_opt) * dt_hr
                                
                                result_row.update({
                                    f'Estimated ({method}) n': n_opt,
                                    f'Estimated ({method}) k (hr)': k_opt,
                                    f'Estimated ({method}) RMSE': rmse,
                                    f'Estimated ({method}) NSE': nse,
                                    f'Estimated ({method}) Peak Discharge (m³/s)': peak_q_est,
                                    f'Estimated ({method}) Time to Peak (hr)': time_to_peak_est,
                                    f'Estimated ({method}) Runoff Volume (m³)': Vol_est
                                })
                                
                                event_results[event_id]['estimates'][method] = {
                                    'Q_est': Q_est_opt,
                                    'n': n_opt,
                                    'k': k_opt
                                }
                    
                    df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)
                    progress_bar.progress((idx + 1) / len(unique_event_ids))
                
                status_text.text("✅ Analysis complete!")
                st.session_state.results_df = df
                st.session_state.event_results = event_results
            
            # Display results
            if st.session_state.results_df is not None:
                st.markdown("---")
                st.header("📊 Results")
                
                # Event selector
                event_id = st.selectbox("Select Event", st.session_state.results_df['Event ID'].values)
                
                # Get event data
                event_row = st.session_state.results_df[st.session_state.results_df['Event ID'] == event_id].iloc[0]
                event_data = st.session_state.event_results[event_id]
                
                # Display metrics in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Observed Peak Discharge", f"{event_row['Observed Peak Discharge (m³/s)']:.2f} m³/s")
                    st.metric("Observed Time to Peak", f"{event_row['Observed Time to Peak (hr)']:.2f} hr")
                
                with col2:
                    st.metric("Total Rainfall Excess", f"{event_row['Total Rainfall Excess Depth (mm)']:.2f} mm")
                    st.metric("Observed Runoff Volume", f"{event_row['Observed Runoff Volume (m³)']:,.0f} m³")
                
                with col3:
                    st.metric("Time of Concentration", f"{event_row['Observed Time of Concentration (hr)']:.2f} hr")
                    st.metric("Lag Time", f"{event_row['Observed Lag Time (hr)']:.2f} hr")
                
                # Method comparison table
                st.subheader("Method Comparison")
                comparison_data = []
                for method in ['Moments', 'L-BFGS-B', 'Differential Evolution', 'Genetic Algorithm']:
                    if f'Estimated ({method}) n' in event_row and pd.notna(event_row[f'Estimated ({method}) n']):
                        comparison_data.append({
                            'Method': method,
                            'n': f"{event_row[f'Estimated ({method}) n']:.3f}",
                            'k (hr)': f"{event_row[f'Estimated ({method}) k (hr)']:.3f}",
                            'RMSE': f"{event_row[f'Estimated ({method}) RMSE']:.3f}",
                            'NSE': f"{event_row[f'Estimated ({method}) NSE']:.3f}",
                            'Peak Discharge (m³/s)': f"{event_row[f'Estimated ({method}) Peak Discharge (m³/s)']:.2f}"
                        })
                
                if comparison_data:
                    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
                
                # Plot
                st.subheader("Hydrograph Comparison")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                               gridspec_kw={'height_ratios': [1, 3]})
                
                # Rainfall hyetograph
                ax1.bar(np.arange(len(event_data['rainfall'])) * event_data['dt'], 
                       event_data['rainfall'] / event_data['dt'],
                       width=0.8 * event_data['dt'], alpha=0.7, color='skyblue', 
                       edgecolor='navy', label='Rainfall Excess')
                ax1.set_ylabel('Rainfall (mm/hr)')
                ax1.set_title(f'Event {event_id}: Rainfall Excess')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Hydrograph
                time_axis = np.arange(len(event_data['Q_obs'])) * event_data['dt']
                ax2.plot(time_axis, event_data['Q_obs'], 'o-', linewidth=2, 
                        markersize=4, label='Observed DRH', color='black')
                
                colors = {'Moments': 'red', 'L-BFGS-B': 'blue', 
                         'Differential Evolution': 'green', 'Genetic Algorithm': 'orange'}
                linestyles = {'Moments': '--', 'L-BFGS-B': '-', 
                             'Differential Evolution': '-.', 'Genetic Algorithm': ':'}
                
                for method, estimates in event_data['estimates'].items():
                    ax2.plot(time_axis[:len(estimates['Q_est'])], estimates['Q_est'],
                            linestyles[method], linewidth=2, color=colors[method],
                            label=f"{method} (n={estimates['n']:.2f}, k={estimates['k']:.2f})")
                
                ax2.set_xlabel('Time (hr)')
                ax2.set_ylabel('Discharge (m³/s)')
                ax2.set_title(f'Event {event_id}: Observed vs Estimated DRH')
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='best')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Download results
                st.markdown("---")
                st.subheader("📥 Download Results")
                
                csv_buffer = io.StringIO()
                st.session_state.results_df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="nash_analysis_results.csv",
                    mime="text/csv"
                )
                
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)
else:
    st.info("👆 Please upload a CSV file to begin analysis")
    
    # Sample data format
    st.markdown("### 📋 Required CSV Format")
    st.markdown("""
    Your CSV file must contain the following columns:
    - `Event_ID`: Unique identifier for each storm event
    - `Time step`: Time in hours
    - `Q_obs_event`: Observed discharge (m³/s)
    - `rain_mm_interval`: Rainfall excess (mm) per interval
    """)