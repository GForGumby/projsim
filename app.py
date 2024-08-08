import streamlit as st
import pandas as pd
import numpy as np
from projections_sim import default_projections, prepare_draft_results, run_parallel_simulations

# Streamlit app
st.title('Fantasy Football Projection Simulator')

# File upload for draft results
uploaded_draft_file = st.file_uploader("Upload your Draft Results CSV file", type=["csv"])

# File upload for custom projections
uploaded_proj_file = st.file_uploader("Upload your Custom Projections CSV file (optional)", type=["csv"])

if uploaded_draft_file is not None:
    draft_results_df = pd.read_csv(uploaded_draft_file)
    st.write("Draft Results Preview:")
    st.dataframe(draft_results_df.head())
    
    if uploaded_proj_file is not None:
        custom_projections_df = pd.read_csv(uploaded_proj_file)
        st.write("Custom Projections Preview:")
        st.dataframe(custom_projections_df.head())
        
        # Create a projection lookup dictionary
        projection_lookup = {
            name: (default_projections[name]['proj'], default_projections[name]['projsd'])
            for name in default_projections
        }
        for index, row in custom_projections_df.iterrows():
            player_name = row['player_name']
            proj = row['proj']
            projsd = row.get('projsd', default_projections.get(player_name, {}).get('projsd', 3))  # Default to 3 if not found
            projection_lookup[player_name] = (proj, projsd)
    else:
        projection_lookup = {
            name: (default_projections[name]['proj'], default_projections[name]['projsd'])
            for name in default_projections
        }
    
    # Parameters for the simulation
    num_simulations = st.number_input("Number of simulations", min_value=1, value=10)
    
    if st.button("Run Simulation"):
        final_results = run_parallel_simulations(num_simulations, draft_results_df, projection_lookup)

        # Display the results
        st.write("Simulation Results:")
        st.dataframe(final_results)

        # Download link for the results
        csv = final_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Simulation Results",
            data=csv,
            file_name='simulation_results.csv',
            mime='text/csv',
        )
