import streamlit as st
import pandas as pd
from projections_sim import prepare_draft_results, run_parallel_simulations

st.title('Fantasy Football Projection Simulator')

# File upload for draft results
uploaded_draft_file = st.file_uploader("Upload your draft results CSV file", type=["csv"])

# File upload for custom projections
uploaded_projections_file = st.file_uploader("Upload your custom projections CSV file (optional)", type=["csv"])

if uploaded_draft_file is not None:
    draft_results_df = pd.read_csv(uploaded_draft_file)

    st.write("Draft Results Data Preview:")
    st.dataframe(draft_results_df.head())

    # Define default projections
    default_projections = {
        "Christian McCaffrey": {'proj': 30, 'projsd': 9},
        "CeeDee Lamb": {'proj': 29, 'projsd': 9},
        # Additional default player projections here...
    }

    if uploaded_projections_file is not None:
        custom_projections_df = pd.read_csv(uploaded_projections_file)
        st.write("Custom Projections Data Preview:")
        st.dataframe(custom_projections_df.head())

        # Create a projection lookup dictionary from the custom projections
        projection_lookup = {}
        for _, row in custom_projections_df.iterrows():
            player_name = row['player_name']
            proj = row['proj']
            projsd = row.get('projsd', default_projections.get(player_name, {}).get('projsd', 6))  # Default projsd = 6 if not specified
            projection_lookup[player_name] = (proj, projsd)
    else:
        # Create a projection lookup dictionary from the default projections
        projection_lookup = {
            name: (default_projections[name]['proj'], default_projections[name]['projsd'])
            for name in default_projections
        }

    # Number of simulations for projection
    num_simulations = st.number_input("Number of simulations", min_value=1, value=1000)

    if st.button("Run Projection Simulation"):
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
