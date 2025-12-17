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
         "📈 Representative Unit Hydrograph"]
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
    area_ha = st.sidebar.number_input("Catchment Area (ha)", value=645.0, min_value=1.0)
    dt_hours = st.sidebar.number_input("Time Step (hours)", value=0.25, min_value=0.01)
    
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
                
                # Display results
                st.markdown("### 📋 Results Summary")
                st.dataframe(results_df, use_container_width=True)
                
                # Overall statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average NSE", f"{results_df['NSE'].mean():.3f}")
                with col2:
                    st.metric("Average RMSE", f"{results_df['RMSE'].mean():.3f}")
                with col3:
                    st.metric("Best NSE", f"{results_df['NSE'].max():.3f}")
                with col4:
                    st.metric("Events with NSE>0.5", len(results_df[results_df['NSE'] > 0.5]))
                
                # Plots for each event
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
        st.markdown('<div class="sub-header">Parameter Estimation Analysis</div>', unsafe_allow_html=True)
        
        methods = st.multiselect(
            "Select Optimization Methods:",
            ["Moments", "L-BFGS-B", "Differential Evolution", "Genetic Algorithm"],
            default=["Moments", "L-BFGS-B"]
        )
        
        if st.button("🚀 Run Estimation", type="primary"):
            with st.spinner("Estimating parameters..."):
                results_list = []
                
                for event_id in unique_events:
                    event_data = df[df['Event_ID'] == event_id].copy()
                    Q_obs = event_data['Q_obs_event'].values
                    rainfall_excess = event_data['rain_mm_interval'].values
                    time_steps = event_data['Time_step'].values
                    
                    dt_hr = time_steps[1] - time_steps[0] if len(time_steps) > 1 else dt_hours
                    
                    event_results = {'Event_ID': event_id}
                    
                    # Moments method
                    if "Moments" in methods:
                        t_q = (np.arange(len(Q_obs)) + 0.5) * dt_hr
                        t_r = (np.arange(len(rainfall_excess)) + 0.5) * dt_hr
                        w_q = Q_obs * (3600.0 * dt_hr)
                        w_r = rainfall_excess.copy()
                        
                        if np.sum(w_q) > 0 and np.sum(w_r) > 0:
                            m1_q, var_q = moments_discrete(t_q, w_q)
                            m1_r, var_r = moments_discrete(t_r, w_r)
                            m1_u = m1_q - m1_r
                            var_u = var_q - var_r
                            
                            if var_u > 1e-9:
                                n_moments = (m1_u ** 2) / var_u
                                k_moments = m1_u / n_moments
                                
                                Q_est = calculate_nash_unit_hydrograph(
                                    rainfall_excess, area_ha, dt_hr, n_moments, k_moments, len(Q_obs)
                                )
                                rmse, nse = calculate_goodness_of_fit(Q_obs, Q_est)
                                
                                event_results['Moments_n'] = n_moments
                                event_results['Moments_k'] = k_moments
                                event_results['Moments_RMSE'] = rmse
                                event_results['Moments_NSE'] = nse
                    
                    # Optimization methods
                    if any(m in methods for m in ["L-BFGS-B", "Differential Evolution", "Genetic Algorithm"]):
                        if 'n_moments' in locals() and np.isfinite(n_moments):
                            initial_guess = [n_moments, k_moments]
                        else:
                            initial_guess = [2.0, 2.0]
                        
                        bounds = [(0.01, 20), (0.01, 50)]
                        
                        if "L-BFGS-B" in methods:
                            result = minimize(
                                objective_rmse, initial_guess,
                                args=(Q_obs, rainfall_excess, area_ha, dt_hr),
                                method='L-BFGS-B', bounds=bounds
                            )
                            n_opt, k_opt = result.x
                            Q_est = calculate_nash_unit_hydrograph(
                                rainfall_excess, area_ha, dt_hr, n_opt, k_opt, len(Q_obs)
                            )
                            rmse, nse = calculate_goodness_of_fit(Q_obs, Q_est)
                            
                            event_results['LBFGSB_n'] = n_opt
                            event_results['LBFGSB_k'] = k_opt
                            event_results['LBFGSB_RMSE'] = rmse
                            event_results['LBFGSB_NSE'] = nse
                        
                        if "Differential Evolution" in methods:
                            result = differential_evolution(
                                objective_rmse, bounds,
                                args=(Q_obs, rainfall_excess, area_ha, dt_hr),
                                seed=42, maxiter=100
                            )
                            n_opt, k_opt = result.x
                            Q_est = calculate_nash_unit_hydrograph(
                                rainfall_excess, area_ha, dt_hr, n_opt, k_opt, len(Q_obs)
                            )
                            rmse, nse = calculate_goodness_of_fit(Q_obs, Q_est)
                            
                            event_results['DE_n'] = n_opt
                            event_results['DE_k'] = k_opt
                            event_results['DE_RMSE'] = rmse
                            event_results['DE_NSE'] = nse
                        
                        if "Genetic Algorithm" in methods:
                            result = dual_annealing(
                                objective_rmse, bounds,
                                args=(Q_obs, rainfall_excess, area_ha, dt_hr),
                                seed=42, maxiter=100
                            )
                            n_opt, k_opt = result.x
                            Q_est = calculate_nash_unit_hydrograph(
                                rainfall_excess, area_ha, dt_hr, n_opt, k_opt, len(Q_obs)
                            )
                            rmse, nse = calculate_goodness_of_fit(Q_obs, Q_est)
                            
                            event_results['GA_n'] = n_opt
                            event_results['GA_k'] = k_opt
                            event_results['GA_RMSE'] = rmse
                            event_results['GA_NSE'] = nse
                    
                    results_list.append(event_results)
                
                results_df = pd.DataFrame(results_list)
                
                st.markdown("### 📋 Estimated Parameters")
                st.dataframe(results_df, use_container_width=True)
                
                # Comparison plot
                st.markdown("### 📊 Method Comparison")
                
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                for method_prefix in ['Moments', 'LBFGSB', 'DE', 'GA']:
                    nse_col = f'{method_prefix}_NSE'
                    if nse_col in results_df.columns:
                        axes[0, 0].plot(results_df['Event_ID'], results_df[nse_col], 
                                       marker='o', label=method_prefix)
                
                axes[0, 0].set_xlabel('Event ID')
                axes[0, 0].set_ylabel('NSE')
                axes[0, 0].set_title('NSE Comparison')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
                
                for method_prefix in ['Moments', 'LBFGSB', 'DE', 'GA']:
                    rmse_col = f'{method_prefix}_RMSE'
                    if rmse_col in results_df.columns:
                        axes[0, 1].plot(results_df['Event_ID'], results_df[rmse_col],
                                       marker='o', label=method_prefix)
                
                axes[0, 1].set_xlabel('Event ID')
                axes[0, 1].set_ylabel('RMSE')
                axes[0, 1].set_title('RMSE Comparison')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                for method_prefix in ['Moments', 'LBFGSB', 'DE', 'GA']:
                    n_col = f'{method_prefix}_n'
                    if n_col in results_df.columns:
                        axes[1, 0].plot(results_df['Event_ID'], results_df[n_col],
                                       marker='o', label=method_prefix)
                
                axes[1, 0].set_xlabel('Event ID')
                axes[1, 0].set_ylabel('n parameter')
                axes[1, 0].set_title('n Parameter Comparison')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                for method_prefix in ['Moments', 'LBFGSB', 'DE', 'GA']:
                    k_col = f'{method_prefix}_k'
                    if k_col in results_df.columns:
                        axes[1, 1].plot(results_df['Event_ID'], results_df[k_col],
                                       marker='o', label=method_prefix)
                
                axes[1, 1].set_xlabel('Event ID')
                axes[1, 1].set_ylabel('k parameter (hr)')
                axes[1, 1].set_title('k Parameter Comparison')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    # ==================== MODE 3: REPRESENTATIVE UH ====================
    else:  # Representative Unit Hydrograph
        st.markdown('<div class="sub-header">Representative Unit Hydrograph</div>', unsafe_allow_html=True)
        
        smooth_option = st.checkbox("Apply smoothing to Representative UH", value=True)
        
        if st.button("🚀 Generate Representative UH", type="primary"):
            with st.spinner("Computing representative unit hydrograph..."):
                # Process events and derive UHs
                uh_list = []
                max_len = 0
                
                for event_id in unique_events:
                    event_data = df[df['Event_ID'] == event_id].copy()
                    time_steps = event_data['Time_step'].values
                    Q_obs = event_data['Q_obs_event'].values
                    rainfall_excess = event_data['rain_mm_interval'].values
                    
                    dt_hr = time_steps[1] - time_steps[0] if len(time_steps) > 1 else dt_hours
                    
                    # Derive UH
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
                
                # Pad UHs to same length
                uh_matrix = np.zeros((len(unique_events), max_len))
                for i, uh in enumerate(uh_list):
                    uh_matrix[i, :len(uh)] = uh
                
                # Compute mean representative UH
                rep_uh = np.nanmean(uh_matrix, axis=0)
                
                # Normalize
                if rep_uh.sum() > 0:
                    rep_uh = rep_uh / rep_uh.sum()
                
                # Apply smoothing
                if smooth_option and max_len >= 5:
                    win = min(5, max_len if max_len % 2 == 1 else max_len - 1)
                    if win >= 3 and win % 2 == 1:
                        try:
                            rep_uh = savgol_filter(rep_uh, window_length=win, polyorder=2)
                        except:
                            pass
                
                # Calculate characteristics
                t_rep = np.arange(len(rep_uh)) * dt_hours
                peak_idx = np.argmax(rep_uh)
                Tp = t_rep[peak_idx]
                Qp_per_cm = rep_uh[peak_idx] * 10
                
                threshold = rep_uh[peak_idx] * 0.01
                significant_indices = np.where(rep_uh > threshold)[0]
                Tb = t_rep[significant_indices[-1]] if len(significant_indices) > 0 else t_rep[-1]
                
                # Display characteristics
                st.markdown("### 📊 Representative UH Characteristics")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Time to Peak (Tp)", f"{Tp:.2f} hr")
                with col2:
                    st.metric("Peak Discharge (Qp)", f"{Qp_per_cm:.4f} (m³/s)/cm")
                with col3:
                    st.metric("Base Time (Tb)", f"{Tb:.2f} hr")
                
                # Plot Representative UH
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
                
                # Validation with events
                st.markdown("### 🎯 Representative UH Validation")
                
                validation_results = []
                
                for event_id in unique_events:
                    event_data = df[df['Event_ID'] == event_id].copy()
                    time_steps = event_data['Time_step'].values
                    Q_obs = event_data['Q_obs_event'].values
                    rainfall_excess = event_data['rain_mm_interval'].values
                    
                    dt_hr = time_steps[1] - time_steps[0] if len(time_steps) > 1 else dt_hours
                    
                    # Simulate using rep UH
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
                
                # Download button for Rep UH
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

if __name__ == "__main__":
    main()