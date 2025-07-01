# tests/test_core_logic.py
"""
Unit tests for isolated functions and methods in core.py and models.py.

These tests verify the correctness of specific business logic components
without needing the entire application pipeline. They use mock data or
minimal, controlled inputs to check for expected outputs, edge cases,
and error handling.
"""

import pandas as pd
import pytest
from core import PredictiveAnalyticsEngine
from models import AdvancedAnalyticsLayer
import networkx as nx

# --- Unit Tests for PredictiveAnalyticsEngine ---

defOf test_allocation_proportional_logic(analytics_engine):
    """
    Tests the proportional allocation method with course. Here is the third file, which contains the **unit tests**.

---

### a simple, clear-cut case.
    It uses the `analytics_engine` fixture from conftest.py. **File 3 of 4: `tests/test_core_logic.py`**

This file is
    """
    kpi_df = pd.DataFrame({
        'Zone': ['Center', 'North', dedicated to **unit testing**: verifying small, isolated pieces of functionality from your `core.py` and `models. 'South'],
        'Integrated_Risk_Score': [0.60, 0.30, py` files. It uses the fixtures defined in `conftest.py` to get configured objects and test their methods0.10], # Total risk = 1.0
        'Expected Incident Volume': [6, 3 in a controlled way. This file should be placed in the `tests/` directory.

```python
# tests, 1]
    })
    available_units = 10
    
    allocations = analytics/test_core_logic.py
"""
Unit tests for the individual components and methods within the core logic._engine._allocate_proportional(kpi_df, available_units)

    assert isinstance(allocations, dict)
    assert sum(allocations.values()) == available_units
    assert allocations['Center'] == 6
    assert allocations['North'] == 3
    assert allocations['South'] == 1



These tests ensure that the foundational pieces of the analytics engine,
data manager, and model layer behave as expected indef test_allocation_proportional_zero_risk(analytics_engine):
    """
    Tests the edge isolation. They use
mocked data and fixtures to verify specific calculations and behaviors.
"""

import pytest
import pandas as case where total risk is zero, ensuring no division by zero
    and that resources are distributed evenly.
    """
 pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

# Core classes    kpi_df = pd.DataFrame({
        'Zone': ['Center', 'North', 'South'], to be tested are imported
from core import DataManager, PredictiveAnalyticsEngine, EnvFactors
from models import AdvancedAnalyticsLayer


        'Integrated_Risk_Score': [0.0, 0.0, 0.0],# Fixtures like 'analytics_engine', 'data_manager', and 'test_config' are automatically
# injected
        'Expected Incident Volume': [0, 0, 0]
    })
    available_units = 9
    
    allocations = analytics_engine._allocate_proportional(kpi_df, available_units by pytest from the conftest.py file.

# --- Unit Tests for DataManager ---

def test_dm)

    assert sum(allocations.values()) == available_units
    assert allocations['Center'] == 3
    assert allocations['North'] == 3
    assert allocations['South'] == 3

def test_initialization(data_manager: DataManager):
    """Verify that the DataManager initializes its core components correctly."""_post_process_allocations_rounding_up(analytics_engine):
    """
    Tests the rounding
    assert data_manager is not None
    assert isinstance(data_manager.zones_gdf, gpd.Geo logic when the initial sum is less than available units.
    It should add the remainder to the highest-priority zones.
    DataFrame)
    assert 'Center' in data_manager.zones
    assert not data_manager.zones_gdf.empty
    assert data_manager.road_graph.number_of_nodes() == 3
    assert data"""
    # Raw float allocations: {Center: 1.4, North: 0.8, South: _manager.ambulances['A01']['status'] == 'Disponible'
    assert isinstance(data_manager0.8}
    # Rounded: {Center: 1, North: 1, South: 1.laplacian_matrix, np.ndarray)


def test_dm_generate_synthetic_incidents(data_manager: DataManager, env_factors: EnvFactors):
    """Test the synthetic incident generation logic."""
    # Test} -> Sum is 3
    # We have 4 units, so 1 must be added. It should go to 'Center'.
    allocations_float = {'Center': 1.4, 'North': 0 with a specific override count
    incidents = data_manager._generate_synthetic_incidents(env_factors, override.8, 'South': 0.8}
    sort_key = pd.Series([10,_count=5)
    assert len(incidents) == 5
    assert all('id' in inc 5, 5], index=['Center', 'North', 'South']) # Center has highest priority
    
    final_ and inc['id'].startswith('SYN-') for inc in incidents)

    # Test with zero override count
    allocations = analytics_engine._post_process_allocations(allocations_float, 4, sort_incidents_zero = data_manager._generate_synthetic_incidents(env_factors, override_count=0)
key)

    assert sum(final_allocations.values()) == 4
    assert final_allocations['Center    assert len(incidents_zero) == 0


# --- Unit Tests for PredictiveAnalyticsEngine ---

def'] == 2 # The extra unit goes here
    assert final_allocations['North'] == 1
    assert final test_proportional_allocation(analytics_engine: PredictiveAnalyticsEngine):
    """Test the proportional allocation strategy with_allocations['South'] == 1

def test_post_process_allocations_rounding_down(analytics_engine):
    """
    Tests the rounding logic when the initial sum is more than available units.
 a clear-cut case."""
    kpi_df = pd.DataFrame({
        'Zone': ['Center    It should remove units from the lowest-priority zones first.
    """
    # Raw float allocations: {', 'North', 'South'],
        'Integrated_Risk_Score': [0.70, 0.Center: 1.8, North: 0.9, South: 0.3}
    # Rounded: {20, 0.10], # Sum = 1.0
        'Expected Incident Volume': [Center: 2, North: 1, South: 0} -> Sum is 3
    # We7, 2, 1]
    })
    available_units = 10
    allocations = analytics_ only have 2 units, so 1 must be removed. It should come from 'North'.
    allocations_float = {'Center': 1.8, 'North': 0.9, 'South': 0.3}engine._allocate_proportional(kpi_df, available_units)
    assert sum(allocations.
    sort_key = pd.Series([10, 5, 1], index=['Center', 'North', 'values()) == available_units
    assert allocations['Center'] == 7
    assert allocations['North'] == 2
    assert allocations['South'] == 1


def test_allocation_rounding_logic(analytics_engineSouth'])
    
    final_allocations = analytics_engine._post_process_allocations(allocations_float, 2, sort_key)
    
    assert sum(final_allocations.values()) == 2: PredictiveAnalyticsEngine):
    """
    Test the post-processing logic that handles rounding issues to ensure
    the total number of allocated units is exactly what's available.
    """
    kpi_df = pd.
    assert final_allocations['Center'] == 2 # Rounded up
    assert final_allocations['DataFrame({
        'Zone': ['Center', 'North', 'South'],
        'Integrated_Risk_ScoreNorth'] == 0 # Rounded up to 1, then corrected down to 0
    assert final_allocations['South': [0.55, 0.25, 0.20],
    })
    available'] == 0


# --- Unit Tests for AdvancedAnalyticsLayer (from models.py) ---

def test_gnn__units = 3
    # Raw proportions: 1.65, 0.75, 0risk_proxy_calculation():
    """
    Tests the GNN risk proxy using a simple, well-understood.6 -> Rounded: 2, 1, 1 (sum=4, wrong)
    # The post-processing graph structure.
    A "barbell" graph has a clear central node.
    """
    G = nx.Graph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C must adjust this. It should remove a unit from the
    # lowest priority zone that has an allocation > 0.
    allocations = analytics_engine._allocate_proportional(kpi_df, available_units)
    ', 'D')]) # B and C are central
    gnn_risk = AdvancedAnalyticsLayer._calculate_gnn_risk(G)

    assert isinstance(gnn_risk, pd.Series)
    assert gnn_assert sum(allocations.values()) == available_units


def test_allocation_with_no_available_units(analytics_engine: PredictiveAnalyticsEngine):
    """Test the system's behavior when there are no ambulances to allocate."""
risk.idxmax() in ['B', 'C'] # B and C should have the highest centrality
    assert g    kpi_df = pd.DataFrame({'Zone': ['Center'], 'Integrated_Risk_Score': [1nn_risk.max() == 1.0
    assert gnn_risk['A'] == 0.0 #.0]})
    allocations = analytics_engine.generate_allocation_recommendations(kpi_df) Leaf nodes have 0 betweenness centrality
    assert gnn_risk['D'] == 0.0

def
    # Based on the test config, only 2 units are 'Disponible', so this tests the real count
    assert test_game_theory_tension_proxy():
    """
    Tests the game theory tension calculation.
    Tension should be a zone's proportional share of the total expected demand.
    """
    kpi_df sum(allocations.values()) == 2

    # Now, let's manually set all ambulances to be = pd.DataFrame({
        'Zone': ['Center', 'North', 'South'],
        'Expected Incident Volume': [15, 10, 5] # Total is 30
    }).set_index(' unavailable
    analytics_engine.dm.ambulances['A01']['status'] = 'En Misión'
    analytics_engine.dm.ambulances['A02']['status'] = 'En Misión'
    allocations = analytics_engine.generate_allocation_recommendations(kpi_df)
    assert sum(Zone')

    tension = AdvancedAnalyticsLayer._calculate_game_theory_tension(kpi_df)

    assert isinstance(tension, pd.Series)
    assert pytest.approx(tension['Center']) == 15 / 3allocations.values()) == 0
    assert allocations['Center'] == 0


# --- Unit Tests for0
    assert pytest.approx(tension['North']) == 10 / 30
    assert pytest.approx(tension['South']) == 5 / 30

def test_hmm_risk_proxy_states():
    """ AdvancedAnalyticsLayer (from models.py) ---

def test_stgp_risk_proxy(data_manager: DataManager):
    """Test the Spatiotemporal Gaussian Process risk proxy."""
    # Create some high-severity incidents ("
    Tests the HMM proxy logic to ensure states are assigned correctly based on KPI thresholds.
    """
    kpi_df = pd.DataFrame({
        # Zone | State | Condition
        'Calm': [0hotspots")
    hotspots_data = [
        {'id': 'H1', 'triage': 'Red', 'geometry': Point(0.5, 0.5)}, # In Center
        {'id': 'H2', 'triage': 'Red', 'geometry': Point(0.5, 0.6)}, #.1, 0.8, 0.1],   # 0 | No conditions met
        'Strained': [0.1, 0.4, 0.1], # 1 | Resource Adequacy < Also in Center
    ]
    incidents_gdf = gpd.GeoDataFrame(hotspots_data, crs="EPSG:4326")

    stgp_risk = AdvancedAnalyticsLayer._calculate_stgp_risk(incidents_gdf, data_manager.zones_gdf)

    assert isinstance(stgp_risk, pd 0.5
        'Volatile': [0.6, 0.8, 0.1.Series)
    assert stgp_risk.idxmax() == 'Center'  # Center should have the highest risk
], # 2 | Chaos Sensitivity > 0.5
        'Clustering': [0.1, 0.8, 0.7], # 2 | Trauma Clustering > 0.6
        'Critical': [    assert stgp_risk['North'] > 0  # North should feel some effect
    assert stgp_risk.max() <= 1.0 and stgp_risk.min() >= 0.0 # Should0.6, 0.4, 0.7]    # 3 | Volatile & Strained
     be normalized


def test_hmm_risk_proxy():
    """Test the Hidden Markov Model state risk proxy."""
    }, index=['Chaos Sensitivity Score', 'Resource Adequacy Index', 'Trauma Clustering Score']).T

    hmmkpi_df = pd.DataFrame({
        'Chaos Sensitivity Score': [0.2, 0._risk = AdvancedAnalyticsLayer._calculate_hmm_risk(kpi_df)
    
    assert pytest.approx(hmm_risk['Calm']) == 0 / 3.0
    assert pytest.approx(hmm8, 0.2, 0.8],
        'Resource Adequacy Index': [0.8, 0.8, 0.2, 0.2],
        'Trauma Clustering Score': [0_risk['Strained']) == 1 / 3.0
    assert pytest.approx(hmm_risk['Volatile']) == 2 / 3.0
    assert pytest.approx(hmm_risk['Clustering']) == 2 / 3.0
    assert pytest.approx(hmm_risk['Critical']) == 3 / 3.0
