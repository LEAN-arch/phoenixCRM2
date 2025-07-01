# tests/conftest.py
"""
This file defines shared fixtures for the entire test suite.
Fixtures are reusable setup functions that provide data, mock objects,
and configured class instances to tests, ensuring a consistent and
clean testing environment.
"""

import pytest
import pandas as pd
import networkx as nx
from core import DataManager, PredictiveAnalyticsEngine, EnvFactors

@pytest.fixture(scope="session")
def test_config() -> dict:
    """
    Provides a minimal, valid, and consistent configuration dictionary for all tests.
    This isolates tests from changes in the main application's default config
    and ensures a stable testing environment.
    """
    return {
        "mapbox_api_key": None,
        "forecast_horizons_hours": [3, 12, 72],
        "kpi_columns": [], # Let the engine define this
        "data": {
            "zones": {
                "Center": {"polygon": [[0,0], [0,1], [1,1], [1,0]], "population": 1000, "crime_rate_modifier": 1.2},
                "North": {"polygon": [[0,1], [0,2], [1,2], [1,1]], "population": 500, "crime_rate_modifier": 0.8},
                "South": {"polygon": [[0,-1], [0,0], [1,0], [1,-1]], "population": 750, "crime_rate_modifier": 1.0},
            },
            "ambulances": {
                "A01": {"status": "Disponible", "home_base": "Center", "location": [0.5, 0.5]},
                "A02": {"status": "Disponible", "home_base": "North", "location": [0.5, 1.5]},
                "A03": {"status": "En Misión", "home_base": "South", "location": [0.5, -0.5]},
            },
            "distributions": {
                "zone": {"Center": 0.6, "North": 0.25, "South": 0.15},
                "incident_type": {"Trauma-Violence": 0.4, "Trauma-Accident": 0.6}
            },
            "road_network": {"edges": [["Center", "North", 2], ["Center", "South", 2]]},
            "real_time_api": {"endpoint": "non_existent_file.json"} # Force fallback to synthetic data
        },
        "model_params": {
            "hawkes_process": {"kappa": 0.5, "beta": 1.0, "trauma_weight": 1.5, "violence_weight": 1.8},
            "sir_model": {"beta": 0.3, "gamma": 0.1},
            "laplacian_diffusion_factor": 0.1,
            "response_time_penalty": 3.0,
            "ensemble_weights": {"violence": 0.5, "accident": 0.5},
            "advanced_model_weights": {"base_ensemble": 1.0, "stgp": 0, "hmm": 0, "gnn": 0, "game_theory": 0},
            "allocation_strategy": "proportional",
            "fallback_forecast_decay_rates": {"3": 0.8, "12": 0.6, "72": 0.2},
        },
        "bayesian_network": { # Minimal valid BN config to prevent errors
            "structure": [["Holiday", "IncidentRate"]],
            "cpds": {
                "Holiday": {"card": 2, "values": [[0.9],[0.1]]},
                "IncidentRate": {"card": 3, "values": [[0.8, 0.3], [0.15, 0.5], [0.05, 0.2]], "evidence": ["Holiday"], "evidence_card": [2]}
            }
        },
        "tcnn_params": {"input_size": 1, "output_size": 1, "channels": [1], "kernel_size": 1, "dropout": 0.1}
    }


@pytest.fixture(scope="session")
def data_manager(test_config: dict) -> DataManager:
    """
    Provides a fully initialized DataManager instance based on the test_config.
    This fixture is session-scoped as the DataManager's internal state
    (like the GDF) doesn't change between tests.
    """
    return DataManager(test_config)


@pytest.fixture(scope="function")
def analytics_engine(data_manager: DataManager, test_config: dict) -> PredictiveAnalyticsEngine:
    """
    Provides a fully initialized PredictiveAnalyticsEngine instance for each test function.
    This is function-scoped to ensure that tests are isolated and one test cannot
    affect another by modifying the engine's state (e.g., forecast_df).
    """
    return PredictiveAnalyticsEngine(data_manager, test_config)


@pytest.fixture(scope="session")
def env_factors() -> EnvFactors:
    """Provides a standard, neutral EnvFactors object for use in tests."""
    return EnvFactors(
        is_holiday=False,
        weather="Clear",
        traffic_level=1.0,
        major_event=False,
        population_density=1000,
        air_quality_index=50.0,
        heatwave_alert=False,
        day_type='Weekday',
        time_of_day='Midday',
        public_event_type='None',
        hospital_divert_status=0.0,
        police_activity='Normal',
        school_in_session=True
    )
