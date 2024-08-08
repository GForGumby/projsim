import streamlit as st
import pandas as pd
from projections_sim import projections, prepare_draft_results, run_parallel_simulations

st.title('Fantasy Football Projection Simulator')

# File upload for draft results
uploaded_file = st.file_uploader("Upload your draft results CSV file", type=["csv"])

if uploaded_file is not None:
    draft_results_df = pd.read_csv(uploaded_file)

    st.write("Draft Results Data Preview:")
    st.dataframe(draft_results_df.head())

    # Number of simulations for projection
    num_simulations = st.number_input("Number of simulations", min_value=1, value=1000)

    if st.button("Run Projection Simulation"):
        # Create a projection lookup dictionary for quick access
        projection_lookup = {
            name: (projections[name]['proj'], projections[name]['projsd'])
            for name in projections
        }

        # Run simulations
        final_results = run_parallel_simulations(num_simulations, draft_results_df, projection_lookup)

        # Display the results
        st.dataframe(final_results)

        # Download link for the results
        csv = final_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Projection Results",
            data=csv,
            file_name='projection_results.csv',
            mime='text/csv',
        )
