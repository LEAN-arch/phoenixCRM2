# tests/test_pipeline.py
"""
Integration tests for the RedShield AI Phoenix application.

These tests verify that the major components of the system (DataManager,
PredictiveAnalyticsEngine, AdvancedAnalyticsLayer) work together correctly.
They test the full data processing and analytics pipeline from end to end,
using a consistent test configuration to ensure reproducibility.
"""

import pytest
import pandas as pd
from core import DataManager, PredictiveAnalyticsEngine, EnvFactors

# All fixtures used here (test_config, data_manager, analytics_engine, env_factors)
# are automatically loaded by pytest from the tests/conftest.py file.

def test_data_manager_initialization(data_manager: DataManager):
    """
    Integration Test: Verifies that the DataManager, when initialized with
    a real configuration, correctly builds all its internal data structures.
    """
    assert data_manager is not None
    assert not data_manager.zones_gdf.empty
    assert "Center" in data_manager.zones_gdf.index
    assert data_manager.road_graph.number_of_nodes() == 3
    assert "A01" in data_manager.ambulances


def test_full_kpi_generation_pipeline(analytics_engine: PredictiveAnalyticsEngine, env_factors: EnvFactors):
    """
    Integration Test: Runs the entire KPI generation pipeline from start to finish.

    This is the most critical test. It ensures that:
    1. Synthetic data can be generated.
    2. The data flows correctly into the KPI generation function.
    3. All calculations, including those from the AdvancedAnalyticsLayer, execute
       without errors.
    4. The final output DataFrame has the correct structure and valid values.
    """
    # 1. ARRANGE: Set up the inputs for the pipeline run.
    # Use the synthetic data generator from the already-initialized DataManager.
    # The test config forces this by pointing the API to a non-existent file.
    current_incidents = analytics_engine.dm.get_current_incidents(env_factors)
    
    # We will use an empty historical dataset for this test run.
    historical_data = []

    # 2. ACT: Execute the main analytics function.
    kpi_df = analytics_engine.generate_kpis(
        historical_data=historical_data,
        env_factors=env_factors,
        current_incidents=current_incidents
    )

    # 3. ASSERT: Verify the output is valid.
    # We check for structure, types, and value ranges, not for exact numerical results.
    assert isinstance(kpi_df, pd.DataFrame), "Output should be a pandas DataFrame"
    assert not kpi_df.empty, "KPI DataFrame should not be empty"
    
    # Check for essential columns
    required_cols = ['Zone', 'Integrated_Risk_Score', 'Ensemble Risk Score', 'GNN_Structural_Risk']
    for col in required_cols:
        assert col in kpi_df.columns, f"Essential column '{col}' is missing"

    # Check data types
    assert kpi_df['Zone'].dtype == 'object'
    assert pd.api.types.is_numeric_dtype(kpi_df['Integrated_Risk_Score']), "Risk score should be numeric"

    # Check value ranges for normalized scores
    for col in ['Integrated_Risk_Score', 'Ensemble Risk Score', 'GNN_Structural_Risk']:
        if col in kpi_df.columns:
            assert kpi_df[col].between(0, 1).all(), f"Values in '{col}' should be between 0 and 1"

    # Check that the number of zones in the output matches the input config
    assert len(kpi_df) == len(analytics_engine.dm.zones)


def test_full_pipeline_with_allocation(analytics_engine: PredictiveAnalyticsEngine, env_factors: EnvFactors):
    """
    Integration Test: Verifies that the pipeline runs through to allocation recommendations.
    """
    # ARRANGE
    current_incidents = analytics_engine.dm.get_current_incidents(env_factors)
    historical_data = []

    # ACT
    kpi_df = analytics_engine.generate_kpis(historical_data, env_factors, current_incidents)
    allocations = analytics_engine.generate_allocation_recommendations(kpi_df)

    # ASSERT
    assert isinstance(allocations, dict)
    
    # Based on the test config, there are 2 available ambulances ('A01', 'A02')
    assert sum(allocations.values()) == 2, "Total allocated units should match available units"
    
    # All zones should be present in the allocation dictionary
    for zone in analytics_engine.dm.zones:
        assert zone in allocations
