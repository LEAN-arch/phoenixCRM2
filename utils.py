# utils.py
"""
Utility functions and classes for the RedShield AI Phoenix application.

This module provides helper functionalities including:
- PDF Reporting: A class to generate comprehensive PDF situational reports from
  the application's output data (KPIs, forecasts, allocations).
"""

import io
import json
import logging
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# Assuming core.py is in the same directory for type hinting
from core import EnvFactors

# --- System Setup ---
logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Handles the generation of PDF reports.
    Refactored for clarity and maintainability.
    """

    @staticmethod
    def _get_styles() -> Dict[str, Any]:
        """Centralizes style definitions for the report."""
        styles = getSampleStyleSheet()
        return {
            'title': styles['Title'],
            'h2': styles['Heading2'],
            'normal': styles['Normal'],
            'env_table': TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (1, -1), colors.lightblue),
            ]),
            'kpi_table': TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkslategray),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
            ]),
            'alloc_table': TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ])
        }

    @classmethod
    def generate_pdf_report(
        cls, kpi_df: pd.DataFrame, forecast_df: pd.DataFrame,
        allocations: Dict[str, int], env_factors: EnvFactors
    ) -> io.BytesIO:
        """
        Main method to generate the complete PDF report.
        Orchestrates the creation of different report sections.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, title="RedShield AI: Phoenix Situational Report")
        styles = cls._get_styles()
        elements = []

        try:
            cls._add_header(elements, styles)
            cls._add_env_factors_table(elements, styles, env_factors)
            cls._add_kpi_table(elements, styles, kpi_df)
            cls._add_forecast_table(elements, styles, forecast_df)
            cls._add_allocation_table(elements, styles, allocations)

            doc.build(elements)
            buffer.seek(0)
            logger.info("PDF report generated successfully.")
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}", exc_info=True)
            return io.BytesIO()

        return buffer

    @staticmethod
    def _add_header(elements: List, styles: Dict):
        """Adds the main title and timestamp to the report."""
        elements.append(Paragraph("RedShield AI: Phoenix v4.0 - Situational Report", styles['title']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['normal']))
        elements.append(Spacer(1, 12))

    @staticmethod
    def _add_env_factors_table(elements: List, styles: Dict, env_factors: EnvFactors):
        """Adds the environmental factors context table."""
        elements.append(Paragraph("Scenario Context: Environmental Factors", styles['h2']))
        env_data = [
            ["Factor", "Value"],
            ["Is Holiday", str(env_factors.is_holiday)],
            ["Weather", str(env_factors.weather)],
            ["Traffic Level", f"{env_factors.traffic_level:.2f}"],
            ["Public Event", str(env_factors.public_event_type)],
            ["AQI", f"{env_factors.air_quality_index:.1f}"],
            ["Heatwave", str(env_factors.heatwave_alert)],
            ["Hospital Strain", f"{env_factors.hospital_divert_status:.0%}"],
            ["Police Activity", str(env_factors.police_activity)]
        ]
        table = Table(env_data, colWidths=[200, 200])
        table.setStyle(styles['env_table'])
        elements.append(table)
        elements.append(Spacer(1, 12))

    @staticmethod
    def _add_kpi_table(elements: List, styles: Dict, kpi_df: pd.DataFrame):
        """Adds the main KPI summary table."""
        elements.append(Paragraph("Risk Analysis Summary", styles['h2']))
        if kpi_df.empty:
            elements.append(Paragraph("No KPI data available.", styles['normal']))
            return

        cols_to_report = [
            'Zone', 'Integrated_Risk_Score', 'Ensemble Risk Score', 'Expected Incident Volume',
            'STGP_Risk', 'HMM_State_Risk', 'GNN_Structural_Risk', 'Game_Theory_Tension'
        ]
        report_cols = [col for col in cols_to_report if col in kpi_df.columns]
        kpi_report_df = kpi_df[report_cols].round(3)

        header = [col.replace('_', ' ').title() for col in kpi_report_df.columns]
        body = kpi_report_df.values.tolist()
        data = [header] + body

        table = Table(data, hAlign='LEFT', repeatRows=1)
        table.setStyle(styles['kpi_table'])
        elements.append(table)
        elements.append(Spacer(1, 12))

    @staticmethod
    def _add_forecast_table(elements: List, styles: Dict, forecast_df: pd.DataFrame):
        """Adds the risk forecast summary table."""
        elements.append(Paragraph("Forecast Summary (Integrated Risk)", styles['h2']))
        if forecast_df.empty:
            elements.append(Paragraph("No forecast data available.", styles['normal']))
            return

        pivot_df = forecast_df.pivot_table(
            index='Zone', columns='Horizon (Hours)', values='Combined Risk'
        ).round(3)

        header = [['Zone'] + [f"{col} hrs" for col in pivot_df.columns]]
        body = [[idx] + row.tolist() for idx, row in pivot_df.iterrows()]
        data = header + body

        table = Table(data, hAlign='LEFT', repeatRows=1)
        table.setStyle(styles['kpi_table'])
        elements.append(table)
        elements.append(Spacer(1, 12))

    @staticmethod
    def _add_allocation_table(elements: List, styles: Dict, allocations: Dict[str, int]):
        """Adds the resource allocation recommendation table."""
        elements.append(Paragraph("Strategic Allocation Recommendation", styles['h2']))
        if not allocations:
            elements.append(Paragraph("No allocation recommendations available.", styles['normal']))
            return

        data = [['Zone', 'Recommended Units']] + list(allocations.items())
        table = Table(data, colWidths=[200, 200])
        table.setStyle(styles['alloc_table'])
        elements.append(table)
