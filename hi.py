import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.signal import savgol_filter
from scipy.optimize import minimize, differential_evolution, dual_annealing
import io

# Page configuration
st.set_page_config(
    page_title="Nash Unit Hydrograph Analysis",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

def calculate_nash_unit_hydrograph(rainfall_excess_mm, area_ha, dt_hr, n, k, output_length):
    """Calculate Nash Unit Hydrograph and convolve with rainfall excess"""
    area_m2 = area_ha * 10000.0
    
    if n <= 0 or k <= 0:
        return np.zeros(output_length) * np.nan
    
    t_uh = np.arange(output_length) * dt_hr
    
    if k > 1e-9:
        unit_hydrograph = (area_ha / 360.0) * (t_uh / k)**(n - 1) * np.exp(-t_uh / k) / (k * gamma(n))
    else:
        return np.zeros(output_length) * np.nan
    
    unit_hydrograph[~np.isfinite(unit_hydrograph)] = 0
    Q_estimated = np.convolve(rainfall_excess_mm, unit_hydrograph)[:output_length]
    
    if len(Q_estimated) < output_length:
        Q_estimated = np.pad(Q_estimated, (0, output_length - len(Q_estimated)), 'constant')
    elif len(Q_estimated) > output_length:
        Q_estimated = Q_estimated[:output_length]
    
    Q_estimated[~np.isfinite(Q_estimated)] = 0
    return Q_estimated

def calculate_goodness_of_fit(observed, estimated):
    """Calculate RMSE and NSE"""
    min_length = min(len(observed), len(estimated))
    obs_trimmed = observed[:min_length]
    est_trimmed = estimated[:min_length]
    
    rmse = np.sqrt(np.mean((obs_trimmed - est_trimmed)**2))
    
    numerator = np.sum((obs_trimmed - est_trimmed)**2)
    denominator = np.sum((obs_trimmed - np.mean(obs_trimmed))**2)
    nse = 1 - (numerator / denominator) if denominator != 0 else np.nan
    
    return rmse, nse

def moments_discrete(t, w):
    """Calculate first moment (mean) and variance"""
    if np.sum(w) == 0:
        return np.nan, np.nan
    m1 = np.sum(t * w) / np.sum(w)
    var = np.sum((t - m1)**2 * w) / np.sum(w)
    return m1, var

def objective_rmse(params, Q_obs, rainfall_excess_mm, area_ha, dt_hr):
    """Objective function for optimization"""
    n, k = params
    
    if n <= 0.01 or k <= 0.01:
        return np.inf
    
    try:
        Q_est = calculate_nash_unit_hydrograph(rainfall_excess_mm, area_ha, dt_hr, n, k, len(Q_obs))
        min_len = min(len(Q_obs), len(Q_est))
        rmse = np.sqrt(np.mean((Q_obs[:min_len] - Q_est[:min_len])**2))
        
        if not np.isfinite(rmse):
            return np.inf
        return rmse
    except:
        return np.inf

def leave_one_out_geometric_mean(series):
    """Calculate leave-one-out geometric mean for each value in series"""
    gmeans = []
    for i in range(len(series)):
        subset = [x for j, x in enumerate(series) if j != i and x > 0]
        if len(subset) > 0:
            gm = np.exp(np.mean(np.log(subset)))
        else:
            gm = np.nan
        gmeans.append(gm)
    return gmeans

def load_and_validate_csv(uploaded_file):
    """Load and validate CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        
        # Column mapping
        colmap = {}
        for c in df.columns:
            cl = c.lower()
            if 'event' in cl and 'id' in cl:
                colmap[c] = 'Event_ID'
            elif 'time' in cl and 'step' in cl:
                colmap[c] = 'Time_step'
            elif 'q' in cl and ('obs' in cl or 'observ' in cl):
                colmap[c] = 'Q_obs_event'
            elif 'rain' in cl and ('mm' in cl or 'interval' in cl):
                colmap[c] = 'rain_mm_interval'
        
        df = df.rename(columns=colmap)
        
        required = ['Event_ID', 'Time_step', 'Q_obs_event', 'rain_mm_interval']
        for r in required:
            if r not in df.columns:
                st.error(f"Required column '{r}' not found in CSV")
                return None
        
        df['Time_step'] = df['Time_step'].astype(float)
        df['Q_obs_event'] = df['Q_obs_event'].astype(float)
        df['rain_mm_interval'] = df['rain_mm_interval'].astype(float)
        
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

# ==================== MAIN APP ====================

def main():
    st.markdown('<div class="main-header">💧 Nash Unit Hydrograph Analysis Tool</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    analysis_mode = st.sidebar.radio(
        "Select Analysis Mode:",
        ["📊 Observed vs Estimated (Fixed n, k)",
         "🔍 Parameter Estimation (Multiple Methods)",
         "📈 Representative Unit Hydrograph",
         "🔮 Runoff Prediction (Rainfall Only)"]
    )
    
    st.sidebar.markdown("---")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])
    
    if uploaded_file is None:
        st.info("👆 Please upload a CSV file to begin analysis")
        st.markdown("""
        ### Required CSV Format:
        - **Event_ID**: Event identifier
        - **Time step**: Time in hours
        - **Q_obs_event**: Observed discharge (m³/s)
        - **rain_mm_interval**: Rainfall excess (mm)
        """)
        return
    
    # Load data
    df = load_and_validate_csv(uploaded_file)
    if df is None:
        return
    
    # Common parameters
    st.sidebar.markdown("### Watershed Parameters")
    area_ha = st.sidebar.number_input("Catchment Area (ha)", value=645.0, min_value=1.0, step=10.0, 
                                       help="Enter the catchment area in hectares")
    dt_hours = st.sidebar.number_input("Time Step (hours)", value=0.25, min_value=0.01, step=0.05,
                                        help="Time interval between data points")
    
    unique_events = df['Event_ID'].unique()
    st.sidebar.markdown(f"**Events found:** {len(unique_events)}")
    
    # ==================== MODE 1: OBSERVED VS ESTIMATED ====================
    if analysis_mode == "📊 Observed vs Estimated (Fixed n, k)":
        st.markdown('<div class="sub-header">Observed vs Estimated Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            n_value = st.number_input("Nash parameter n", value=1.735, min_value=0.1, max_value=10.0, step=0.01)
        with col2:
            k_value = st.number_input("Nash parameter k (hours)", value=1.031, min_value=0.1, max_value=50.0, step=0.01)
        
        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Processing events..."):
                results_list = []
                
                for event_id in unique_events:
                    event_data = df[df['Event_ID'] == event_id].copy()
                    time_steps = event_data['Time_step'].values
                    observed_discharge = event_data['Q_obs_event'].values
                    rainfall_excess = event_data['rain_mm_interval'].values
                    
                    dt_hr = time_steps[1] - time_steps[0] if len(time_steps) > 1 else dt_hours
                    
                    estimated_discharge = calculate_nash_unit_hydrograph(
                        rainfall_excess, area_ha, dt_hr, n_value, k_value, len(observed_discharge)
                    )
                    
                    rmse, nse = calculate_goodness_of_fit(observed_discharge, estimated_discharge)
                    
                    results_list.append({
                        'Event_ID': event_id,
                        'Total_Rainfall_mm': np.sum(rainfall_excess),
                        'Observed_Peak_m3s': np.max(observed_discharge),
                        'Estimated_Peak_m3s': np.max(estimated_discharge),
                        'RMSE': rmse,
                        'NSE': nse,
                        'Time_to_Peak_Obs_hr': np.argmax(observed_discharge) * dt_hr,
                        'Time_to_Peak_Est_hr': np.argmax(estimated_discharge) * dt_hr
                    })
                
                results_df = pd.DataFrame(results_list)
                
                st.markdown("### 📋 Results Summary")
                st.dataframe(results_df, use_container_width=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average NSE", f"{results_df['NSE'].mean():.3f}")
                with col2:
                    st.metric("Average RMSE", f"{results_df['RMSE'].mean():.3f}")
                with col3:
                    st.metric("Best NSE", f"{results_df['NSE'].max():.3f}")
                with col4:
                    st.metric("Events with NSE>0.5", len(results_df[results_df['NSE'] > 0.5]))
                
                st.markdown("### 📊 Event Hydrographs")
                
                for event_id in unique_events:
                    event_data = df[df['Event_ID'] == event_id].copy()
                    time_steps = event_data['Time_step'].values
                    observed_discharge = event_data['Q_obs_event'].values
                    rainfall_excess = event_data['rain_mm_interval'].values
                    
                    dt_hr = time_steps[1] - time_steps[0] if len(time_steps) > 1 else dt_hours
                    time_axis = time_steps - time_steps[0]
                    
                    estimated_discharge = calculate_nash_unit_hydrograph(
                        rainfall_excess, area_ha, dt_hr, n_value, k_value, len(observed_discharge)
                    )
                    
                    rmse, nse = calculate_goodness_of_fit(observed_discharge, estimated_discharge)
                    
                    fig, ax1 = plt.subplots(figsize=(12, 6))
                    
                    ax1.plot(time_axis, observed_discharge, 'o-', linewidth=2, markersize=4,
                            color='black', label='Observed')
                    ax1.plot(time_axis, estimated_discharge, '--', linewidth=2,
                            color='red', label=f'Estimated (n={n_value}, k={k_value})')
                    ax1.set_xlabel('Time (hr)', fontsize=12)
                    ax1.set_ylabel('Discharge (m³/s)', fontsize=12)
                    ax1.set_title(f'Event {event_id} | NSE={nse:.3f}, RMSE={rmse:.3f}', fontsize=13)
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    ax2 = ax1.twinx()
                    ax2.bar(time_axis, rainfall_excess/dt_hr, width=0.8*dt_hr,
                           alpha=0.5, color='skyblue', label='Rainfall')
                    ax2.set_ylabel('Rainfall (mm/hr)', fontsize=12)
                    ax2.invert_yaxis()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
    
    # ==================== MODE 2: PARAMETER ESTIMATION ====================
    elif analysis_mode == "🔍 Parameter Estimation (Multiple Methods)":
        st.markdown('<div class="sub-header">Multi-Method Parameter Estimation</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Run All Models & Calculate Geometric Mean", type="primary"):
            with st.spinner("Calculating parameters for all events..."):
                results_list = []
                all_event_plots = [] 

                for event_id in unique_events:
                    event_data = df[df['Event_ID'] == event_id].copy()
                    Q_obs = event_data['Q_obs_event'].values
                    rainfall_excess = event_data['rain_mm_interval'].values
                    time_steps = event_data['Time_step'].values
                    dt_hr = time_steps[1] - time_steps[0] if len(time_steps) > 1 else dt_hours
                    time_axis = time_steps - time_steps[0]
                    
                    event_results = {'Event_ID': event_id}
                    model_outputs = {} 

                    # 1. Moments Method
                    t_q = (np.arange(len(Q_obs)) + 0.5) * dt_hr
                    t_r = (np.arange(len(rainfall_excess)) + 0.5) * dt_hr
                    w_q = Q_obs * (3600.0 * dt_hr)
                    w_r = rainfall_excess.copy()
                    
                    n_mom, k_mom = 2.0, 2.0
                    if np.sum(w_q) > 0 and np.sum(w_r) > 0:
                        m1_q, var_q = moments_discrete(t_q, w_q)
                        m1_r, var_r = moments_discrete(t_r, w_r)
                        m1_u, var_u = m1_q - m1_r, var_q - var_r
                        if var_u > 1e-9:
                            n_mom = (m1_u ** 2) / var_u
                            k_mom = m1_u / n_mom
                            Q_mom = calculate_nash_unit_hydrograph(rainfall_excess, area_ha, dt_hr, n_mom, k_mom, len(Q_obs))
                            rmse_mom, nse_mom = calculate_goodness_of_fit(Q_obs, Q_mom)
                            event_results.update({
                                'Moments_n': n_mom, 
                                'Moments_k': k_mom, 
                                'Moments_RMSE': rmse_mom,
                                'Moments_NSE': nse_mom
                            })
                            model_outputs['Moments'] = {'q': Q_mom, 'n': n_mom, 'k': k_mom}

                    bounds = [(0.01, 20), (0.01, 50)]
                    initial_guess = [n_mom, k_mom]

                    # 2. L-BFGS-B
                    res_lb = minimize(objective_rmse, initial_guess, args=(Q_obs, rainfall_excess, area_ha, dt_hr), method='L-BFGS-B', bounds=bounds)
                    n_lb, k_lb = res_lb.x
                    Q_lb = calculate_nash_unit_hydrograph(rainfall_excess, area_ha, dt_hr, n_lb, k_lb, len(Q_obs))
                    rmse_lb, nse_lb = calculate_goodness_of_fit(Q_obs, Q_lb)
                    event_results.update({
                        'LBFGSB_n': n_lb, 
                        'LBFGSB_k': k_lb, 
                        'LBFGSB_RMSE': rmse_lb,
                        'LBFGSB_NSE': nse_lb
                    })
                    model_outputs['L-BFGS-B'] = {'q': Q_lb, 'n': n_lb, 'k': k_lb}

                    # 3. Differential Evolution
                    res_de = differential_evolution(objective_rmse, bounds, args=(Q_obs, rainfall_excess, area_ha, dt_hr), seed=42)
                    n_de, k_de = res_de.x
                    Q_de = calculate_nash_unit_hydrograph(rainfall_excess, area_ha, dt_hr, n_de, k_de, len(Q_obs))
                    rmse_de, nse_de = calculate_goodness_of_fit(Q_obs, Q_de)
                    event_results.update({
                        'DE_n': n_de, 
                        'DE_k': k_de,
                        'DE_RMSE': rmse_de,
                        'DE_NSE': nse_de
                    })
                    model_outputs['Diff. Evolution'] = {'q': Q_de, 'n': n_de, 'k': k_de}

                    # 4. Genetic Algorithm
                    res_ga = dual_annealing(objective_rmse, bounds, args=(Q_obs, rainfall_excess, area_ha, dt_hr), seed=42)
                    n_ga, k_ga = res_ga.x
                    Q_ga = calculate_nash_unit_hydrograph(rainfall_excess, area_ha, dt_hr, n_ga, k_ga, len(Q_obs))
                    rmse_ga, nse_ga = calculate_goodness_of_fit(Q_obs, Q_ga)
                    event_results.update({
                        'GA_n': n_ga, 
                        'GA_k': k_ga,
                        'GA_RMSE': rmse_ga,
                        'GA_NSE': nse_ga
                    })
                    model_outputs['Gen. Algorithm'] = {'q': Q_ga, 'n': n_ga, 'k': k_ga}

                    results_list.append(event_results)
                    all_event_plots.append({
                        'id': event_id, 
                        'time': time_axis, 
                        'obs': Q_obs, 
                        'rain': rainfall_excess, 
                        'models': model_outputs, 
                        'dt': dt_hr
                    })

                results_df = pd.DataFrame(results_list)
                
                st.markdown("### 📋 Comparative Parameter Table")
                st.dataframe(results_df, use_container_width=True)

                # ==================== GEOMETRIC MEAN CALCULATION ====================
                st.markdown("---")
                st.markdown('<div class="sub-header">🎯 Leave-One-Out Geometric Mean Parameters</div>', unsafe_allow_html=True)
                st.info("These optimized parameters exclude each event when calculating its geometric mean, providing robust cross-validated values for prediction.")
                
                gm_results = pd.DataFrame({
                    'Event_ID': results_df['Event_ID'],
                    'GM_Moments_n': leave_one_out_geometric_mean(results_df['Moments_n']),
                    'GM_Moments_k': leave_one_out_geometric_mean(results_df['Moments_k']),
                    'GM_LBFGSB_n': leave_one_out_geometric_mean(results_df['LBFGSB_n']),
                    'GM_LBFGSB_k': leave_one_out_geometric_mean(results_df['LBFGSB_k']),
                    'GM_DE_n': leave_one_out_geometric_mean(results_df['DE_n']),
                    'GM_DE_k': leave_one_out_geometric_mean(results_df['DE_k']),
                    'GM_GA_n': leave_one_out_geometric_mean(results_df['GA_n']),
                    'GM_GA_k': leave_one_out_geometric_mean(results_df['GA_k'])
                })
                
                gm_results = gm_results.round(3)
                st.dataframe(gm_results, use_container_width=True)
                
                # Overall Geometric Mean (for all events combined)
                st.markdown("---")
                st.markdown("### 🌟 Overall Recommended Parameters (Use for Prediction)")
                
                # Calculate individual method GMs
                overall_n_mom = np.exp(np.mean(np.log(results_df['Moments_n'])))
                overall_k_mom = np.exp(np.mean(np.log(results_df['Moments_k'])))
                
                overall_n_lb = np.exp(np.mean(np.log(results_df['LBFGSB_n'])))
                overall_k_lb = np.exp(np.mean(np.log(results_df['LBFGSB_k'])))
                
                overall_n_de = np.exp(np.mean(np.log(results_df['DE_n'])))
                overall_k_de = np.exp(np.mean(np.log(results_df['DE_k'])))
                
                overall_n_ga = np.exp(np.mean(np.log(results_df['GA_n'])))
                overall_k_ga = np.exp(np.mean(np.log(results_df['GA_k'])))
                
                # Calculate Combined Geometric Mean across all methods
                all_n_values = np.concatenate([
                    results_df['Moments_n'].values,
                    results_df['LBFGSB_n'].values,
                    results_df['DE_n'].values,
                    results_df['GA_n'].values
                ])
                all_k_values = np.concatenate([
                    results_df['Moments_k'].values,
                    results_df['LBFGSB_k'].values,
                    results_df['DE_k'].values,
                    results_df['GA_k'].values
                ])
                
                combined_n = np.exp(np.mean(np.log(all_n_values)))
                combined_k = np.exp(np.mean(np.log(all_k_values)))
                
                # Store in session state for use in prediction mode
                st.session_state['gm_params'] = {
                    'Moments_n': overall_n_mom,
                    'Moments_k': overall_k_mom,
                    'LBFGSB_n': overall_n_lb,
                    'LBFGSB_k': overall_k_lb,
                    'DE_n': overall_n_de,
                    'DE_k': overall_k_de,
                    'GA_n': overall_n_ga,
                    'GA_k': overall_k_ga,
                    'Combined_n': combined_n,
                    'Combined_k': combined_k
                }
                
                # Create beautiful summary table
                summary_params_df = pd.DataFrame({
                    'Optimization Method': [
                        'Moments Method',
                        'L-BFGS-B',
                        'Differential Evolution',
                        'Genetic Algorithm',
                        '⭐ COMBINED (Recommended)'
                    ],
                    'Nash n': [
                        f"{overall_n_mom:.4f}",
                        f"{overall_n_lb:.4f}",
                        f"{overall_n_de:.4f}",
                        f"{overall_n_ga:.4f}",
                        f"{combined_n:.4f}"
                    ],
                    'Nash k (hours)': [
                        f"{overall_k_mom:.4f}",
                        f"{overall_k_lb:.4f}",
                        f"{overall_k_de:.4f}",
                        f"{overall_k_ga:.4f}",
                        f"{combined_k:.4f}"
                    ],
                    'Description': [
                        'Statistical moment matching',
                        'Gradient-based optimization',
                        'Evolutionary global search',
                        'Simulated annealing approach',
                        'Geometric mean of all methods'
                    ]
                })
                
                st.dataframe(
                    summary_params_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Optimization Method": st.column_config.TextColumn(
                            "Optimization Method",
                            width="medium",
                        ),
                        "Nash n": st.column_config.TextColumn(
                            "Nash n",
                            width="small",
                        ),
                        "Nash k (hours)": st.column_config.TextColumn(
                            "Nash k (hours)",
                            width="small",
                        ),
                        "Description": st.column_config.TextColumn(
                            "Description",
                            width="large",
                        )
                    }
                )
                
                st.info(f"💡 **Recommended for Prediction:** Use the ⭐ COMBINED parameters (n={combined_n:.4f}, k={combined_k:.4f}) for most robust results across all optimization methods.")
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    csv1 = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Event Parameters",
                        data=csv1,
                        file_name="event_parameters.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    csv2 = gm_results.to_csv(index=False)
                    st.download_button(
                        "📥 Download Geometric Mean Parameters",
                        data=csv2,
                        file_name="geometric_mean_parameters.csv",
                        mime="text/csv"
                    )

                st.markdown("---")
                st.markdown("### 📊 Overlap-Aware Comparison Hydrographs")

                for plot_data in all_event_plots:
                    with st.expander(f"Event {plot_data['id']} Comparison", expanded=False):
                        fig, ax1 = plt.subplots(figsize=(12, 5))
                        ax1.plot(plot_data['time'], plot_data['obs'], 'k-', linewidth=3, label='Observed', zorder=2)
                        
                        styles = {
                            'Moments': {'c': '#e41a1c', 'lw': 5.0, 'ls': '-', 'alpha': 0.4},
                            'L-BFGS-B': {'c': '#377eb8', 'lw': 3.5, 'ls': '--', 'alpha': 0.7},
                            'Diff. Evolution': {'c': '#4daf4a', 'lw': 2.0, 'ls': ':', 'alpha': 0.9},
                            'Gen. Algorithm': {'c': '#984ea3', 'lw': 1.0, 'ls': '-.', 'alpha': 1.0}
                        }

                        for m_name, data in plot_data['models'].items():
                            s = styles[m_name]
                            label_str = f"{m_name} (n={data['n']:.2f}, k={data['k']:.2f})"
                            ax1.plot(plot_data['time'], data['q'], 
                                    color=s['c'], linewidth=s['lw'], linestyle=s['ls'], 
                                    alpha=s['alpha'], label=label_str, zorder=3)
                        
                        ax1.set_xlabel('Time (hr)')
                        ax1.set_ylabel('Discharge (m³/s)')
                        ax1.legend(loc='upper right', fontsize='x-small', frameon=True, shadow=True)
                        ax1.grid(True, alpha=0.2)

                        ax2 = ax1.twinx()
                        ax2.bar(plot_data['time'], plot_data['rain'], width=plot_data['dt']*0.7, alpha=0.15, color='gray')
                        ax2.set_ylabel('Rainfall (mm)')
                        ax2.invert_yaxis()
                        
                        st.pyplot(fig)
                        plt.close()
    
    # ==================== MODE 3: REPRESENTATIVE UH ====================
    elif analysis_mode == "📈 Representative Unit Hydrograph":
        st.markdown('<div class="sub-header">Representative Unit Hydrograph</div>', unsafe_allow_html=True)
        
        smooth_option = st.checkbox("Apply smoothing to Representative UH", value=True)
        
        if st.button("🚀 Generate Representative UH", type="primary"):
            with st.spinner("Computing representative unit hydrograph..."):
                uh_list = []
                max_len = 0
                
                for event_id in unique_events:
                    event_data = df[df['Event_ID'] == event_id].copy()
                    time_steps = event_data['Time_step'].values
                    Q_obs = event_data['Q_obs_event'].values
                    rainfall_excess = event_data['rain_mm_interval'].values
                    
                    dt_hr = time_steps[1] - time_steps[0] if len(time_steps) > 1 else dt_hours
                    
                    A_m2 = area_ha * 1e4
                    dt_s = dt_hr * 3600.0
                    runoff_vol_per_step = Q_obs * dt_s
                    runoff_depth_mm = (runoff_vol_per_step / A_m2) * 1000.0
                    P_total = np.sum(rainfall_excess)
                    
                    if P_total > 0:
                        UH_event = runoff_depth_mm / P_total
                    else:
                        UH_event = np.zeros_like(runoff_depth_mm)
                    
                    uh_list.append(UH_event)
                    if len(UH_event) > max_len:
                        max_len = len(UH_event)
                
                uh_matrix = np.zeros((len(unique_events), max_len))
                for i, uh in enumerate(uh_list):
                    uh_matrix[i, :len(uh)] = uh
                
                rep_uh = np.nanmean(uh_matrix, axis=0)
                
                if rep_uh.sum() > 0:
                    rep_uh = rep_uh / rep_uh.sum()
                
                if smooth_option and max_len >= 5:
                    win = min(5, max_len if max_len % 2 == 1 else max_len - 1)
                    if win >= 3 and win % 2 == 1:
                        try:
                            rep_uh = savgol_filter(rep_uh, window_length=win, polyorder=2)
                        except:
                            pass
                
                t_rep = np.arange(len(rep_uh)) * dt_hours
                peak_idx = np.argmax(rep_uh)
                Tp = t_rep[peak_idx]
                Qp_per_cm = rep_uh[peak_idx] * 10
                
                threshold = rep_uh[peak_idx] * 0.01
                significant_indices = np.where(rep_uh > threshold)[0]
                Tb = t_rep[significant_indices[-1]] if len(significant_indices) > 0 else t_rep[-1]
                
                st.markdown("### 📊 Representative UH Characteristics")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Time to Peak (Tp)", f"{Tp:.2f} hr")
                with col2:
                    st.metric("Peak Discharge (Qp)", f"{Qp_per_cm:.4f} (m³/s)/cm")
                with col3:
                    st.metric("Base Time (Tb)", f"{Tb:.2f} hr")
                
                fig, ax = plt.subplots(figsize=(14, 7))
                
                ax.plot(t_rep, rep_uh, 'b-', linewidth=3, label='Representative UH',
                       marker='o', markersize=5)
                ax.plot(Tp, rep_uh[peak_idx], 'r*', markersize=20, 
                       label=f"Tp = {Tp:.2f} h", zorder=10)
                ax.axvline(x=Tp, color='red', linestyle=':', linewidth=2, alpha=0.6)
                ax.axvline(x=Tb, color='purple', linestyle='--', linewidth=2, alpha=0.6,
                          label=f"Tb = {Tb:.2f} h")
                
                ax.set_xlabel('Time (hours)', fontsize=13, fontweight='bold')
                ax.set_ylabel('UH Ordinate (mm/mm)', fontsize=13, fontweight='bold')
                ax.set_title(f'Representative Unit Hydrograph - Area: {area_ha} ha',
                            fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.markdown("### 🎯 Representative UH Validation")
                
                validation_results = []
                
                for event_id in unique_events:
                    event_data = df[df['Event_ID'] == event_id].copy()
                    time_steps = event_data['Time_step'].values
                    Q_obs = event_data['Q_obs_event'].values
                    rainfall_excess = event_data['rain_mm_interval'].values
                    
                    dt_hr = time_steps[1] - time_steps[0] if len(time_steps) > 1 else dt_hours
                    
                    rep_uh_trim = rep_uh[:len(rainfall_excess)]
                    conv_mm = np.convolve(rainfall_excess, rep_uh_trim, mode='full')[:len(Q_obs)]
                    A_m2 = area_ha * 1e4
                    dt_s = dt_hr * 3600.0
                    Q_sim = conv_mm * (A_m2 / 1000.0) / dt_s
                    
                    rmse, nse = calculate_goodness_of_fit(Q_obs, Q_sim)
                    
                    validation_results.append({
                        'Event_ID': event_id,
                        'NSE': nse,
                        'RMSE': rmse,
                        'Peak_Obs': np.max(Q_obs),
                        'Peak_Sim': np.max(Q_sim)
                    })
                
                val_df = pd.DataFrame(validation_results)
                st.dataframe(val_df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average NSE", f"{val_df['NSE'].mean():.3f}")
                with col2:
                    st.metric("Average RMSE", f"{val_df['RMSE'].mean():.3f}")
                
                rep_uh_df = pd.DataFrame({
                    'Time_h': t_rep,
                    'UH_Ordinate_mm_per_mm': rep_uh
                })
                
                csv = rep_uh_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Representative UH",
                    data=csv,
                    file_name="representative_uh.csv",
                    mime="text/csv"
                )
    
    # ==================== MODE 4: RUNOFF PREDICTION ====================
    else:
        st.markdown('<div class="sub-header">Predict Runoff from Rainfall Data</div>', unsafe_allow_html=True)
        st.info("This mode uses your calibrated parameters to forecast runoff. No observed discharge (Q_obs) is required.")

        # Check if geometric mean parameters are available
        if 'gm_params' in st.session_state:
            st.success("✅ Geometric Mean parameters loaded from Parameter Estimation!")
            gm = st.session_state['gm_params']
            
            st.markdown("### 📊 Available Optimized Parameters")
            param_df = pd.DataFrame({
                'Method': ['Moments', 'L-BFGS-B', 'Differential Evolution', 'Genetic Algorithm', '⭐ COMBINED (All Methods)'],
                'n': [gm['Moments_n'], gm['LBFGSB_n'], gm['DE_n'], gm['GA_n'], gm['Combined_n']],
                'k (hrs)': [gm['Moments_k'], gm['LBFGSB_k'], gm['DE_k'], gm['GA_k'], gm['Combined_k']]
            })
            st.dataframe(param_df, use_container_width=True)
        else:
            st.warning("⚠️ No geometric mean parameters found. Run 'Parameter Estimation' mode first, or enter parameters manually below.")
        
        st.markdown("---")
        st.markdown("### 🔧 Select Prediction Method")
        
        # Method selection
        if 'gm_params' in st.session_state:
            method_choice = st.radio(
                "Choose parameter set:",
                ["⭐ Combined (Recommended)", "Moments", "L-BFGS-B", "Differential Evolution", "Genetic Algorithm", "Manual Entry"],
                horizontal=True
            )
            
            if method_choice == "⭐ Combined (Recommended)":
                n_pred = gm['Combined_n']
                k_pred = gm['Combined_k']
                st.info(f"Using Combined GM: n={n_pred:.4f}, k={k_pred:.4f}")
            elif method_choice == "Moments":
                n_pred = gm['Moments_n']
                k_pred = gm['Moments_k']
            elif method_choice == "L-BFGS-B":
                n_pred = gm['LBFGSB_n']
                k_pred = gm['LBFGSB_k']
            elif method_choice == "Differential Evolution":
                n_pred = gm['DE_n']
                k_pred = gm['DE_k']
            elif method_choice == "Genetic Algorithm":
                n_pred = gm['GA_n']
                k_pred = gm['GA_k']
            else:  # Manual Entry
                col1, col2 = st.columns(2)
                with col1:
                    n_pred = st.number_input("Calibrated Nash n", value=1.735, min_value=0.1, step=0.01)
                with col2:
                    k_pred = st.number_input("Calibrated Nash k (hours)", value=1.031, min_value=0.1, step=0.01)
        else:
            col1, col2 = st.columns(2)
            with col1:
                n_pred = st.number_input("Calibrated Nash n", value=1.735, min_value=0.1, step=0.01)
            with col2:
                k_pred = st.number_input("Calibrated Nash k (hours)", value=1.031, min_value=0.1, step=0.01)

        if st.button("🔮 Generate Prediction", type="primary"):
            st.markdown(f"**Using Parameters: n = {n_pred:.4f}, k = {k_pred:.4f}**")
            
            prediction_results = []
            
            for event_id in unique_events:
                event_data = df[df['Event_ID'] == event_id].copy()
                rain = event_data['rain_mm_interval'].values
                time_steps = event_data['Time_step'].values
                dt_hr = time_steps[1] - time_steps[0] if len(time_steps) > 1 else dt_hours
                time_axis = time_steps - time_steps[0]

                output_len = len(rain) + int(10/dt_hr) 
                Q_pred = calculate_nash_unit_hydrograph(rain, area_ha, dt_hr, n_pred, k_pred, output_len)
                
                time_axis_pred = np.arange(output_len) * dt_hr

                res_df = pd.DataFrame({
                    'Time_hr': time_axis_pred,
                    'Rainfall_mm': np.pad(rain, (0, output_len - len(rain)), 'constant'),
                    'Predicted_Runoff_m3s': Q_pred
                })
                
                with st.expander(f"Prediction for Event {event_id}", expanded=True):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig, ax1 = plt.subplots(figsize=(12, 5))
                        ax1.plot(time_axis_pred, Q_pred, 'r-', linewidth=2.5, label=f'Predicted (n={n_pred:.3f}, k={k_pred:.3f})')
                        ax1.set_xlabel('Time (hr)', fontsize=12)
                        ax1.set_ylabel('Predicted Discharge (m³/s)', color='red', fontsize=12)
                        ax1.tick_params(axis='y', labelcolor='red')
                        ax1.grid(True, alpha=0.2)
                        ax1.legend(loc='upper left')

                        ax2 = ax1.twinx()
                        ax2.bar(time_axis_pred, res_df['Rainfall_mm'], width=dt_hr*0.7, alpha=0.2, color='blue', label='Rainfall Input')
                        ax2.set_ylabel('Rainfall (mm)', color='blue', fontsize=12)
                        ax2.invert_yaxis()
                        ax2.legend(loc='upper right')
                        
                        plt.title(f"Event {event_id} - Runoff Prediction", fontsize=14, fontweight='bold')
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        st.markdown("**Prediction Summary**")
                        st.metric("Peak Flow", f"{np.max(Q_pred):.3f} m³/s")
                        st.metric("Total Rainfall", f"{np.sum(rain):.2f} mm")
                        st.metric("Time to Peak", f"{time_axis_pred[np.argmax(Q_pred)]:.2f} hr")

                    csv = res_df.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download Event {event_id} Prediction",
                        data=csv,
                        file_name=f"prediction_event_{event_id}.csv",
                        mime="text/csv",
                        key=f"download_{event_id}"
                    )

if __name__ == "__main__":
    main()
