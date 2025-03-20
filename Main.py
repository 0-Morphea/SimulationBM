
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict

# For a cleaner display, we use matplotlib in 'inline' mode integrated with Streamlit
# so that the figures can be displayed automatically.

# ---------------------------
# Enums and Data Classes
# ---------------------------

class UserSegment(Enum):
    ALGO_TRADER = "Algo Trader"
    MARKET_MAKER = "Market Maker"
    MUTUAL_BETS = "Mutual Bets"
    WHITE_LABEL = "White Label"
    INSTITUTIONAL = "Institutional"

@dataclass
class Scenario:
    name: str
    user_growth_rate: float
    volume_growth_rate: float
    fee_growth_rate: float
    retention_rate: float
    marketing_cost_growth: float
    opex_growth: float

# We keep other classes minimal for clarity. You can reintroduce MarketingCampaign, etc. if needed.

# ---------------------------
# Core Business Model
# ---------------------------

class BusinessModel:
    def __init__(self, name: str):
        self.name = name
        self.initial_users = 10000
        self.initial_volume_total = 75000000
        self.initial_fee_rate = 0.0002
        self.gross_margin = 0.90

        # Base segment distribution
        self.segment_base_params = {
            UserSegment.ALGO_TRADER: {
                'base_users': 4000,
                'volume_multiplier': 1.5,
                'fee_multiplier': 1.0
            },
            UserSegment.MARKET_MAKER: {
                'base_users': 2000,
                'volume_multiplier': 3.0,
                'fee_multiplier': 0.7
            },
            UserSegment.MUTUAL_BETS: {
                'base_users': 1500,
                'volume_multiplier': 0.4,
                'fee_multiplier': 1.0
            },
            UserSegment.WHITE_LABEL: {
                'base_users': 1500,
                'volume_multiplier': 2.0,
                'fee_multiplier': 0.8
            },
            UserSegment.INSTITUTIONAL: {
                'base_users': 1000,
                'volume_multiplier': 4.0,
                'fee_multiplier': 0.6
            }
        }

# ---------------------------
# Simulation Class
# ---------------------------

class Simulation:
    def __init__(self, business_model: BusinessModel, scenarios: List[Scenario], months: int):
        """
        :param business_model: The business model reference.
        :param scenarios: A list of Scenario instances.
        :param months: Number of months to simulate.
        """
        self.business_model = business_model
        self.scenarios = scenarios
        self.months = months
        self.results = {}  # Dictionary: scenario_name -> DataFrame

    def run(self):
        for scenario in self.scenarios:
            df = self._simulate_scenario(scenario)
            self.results[scenario.name] = df

    def _simulate_scenario(self, scenario: Scenario) -> pd.DataFrame:
        rows = []
        # We do a simple model: each month, we compute users, volume, etc.
        for month in range(1, self.months+1):
            # Basic user growth
            total_users = self.business_model.initial_users * (1 + scenario.user_growth_rate)**(month-1)

            # Simple approach to volume (no segmentation logic for user retention, etc.):
            total_volume = (self.business_model.initial_volume_total
                            * (1 + scenario.volume_growth_rate)**(month-1))

            fee_rate = (self.business_model.initial_fee_rate
                        * (1 + scenario.fee_growth_rate)**(month-1))

            # Let’s pretend we have net profit = ( total_revenue*gross_margin - cost )
            # For illustration
            total_revenue = total_volume * fee_rate
            cost = self._calculate_costs(scenario, month-1)
            net_profit = (total_revenue * self.business_model.gross_margin) - cost

            rows.append({
                "Month": month,
                "Total_Users": total_users,
                "Total_Volume": total_volume,
                "Fee_Rate": fee_rate,
                "Total_Revenue": total_revenue,
                "Cost": cost,
                "Net_Profit": net_profit
            })
        return pd.DataFrame(rows)

    def _calculate_costs(self, scenario: Scenario, month_idx: int) -> float:
        """
        Basic cost model: marketing grows by marketing_cost_growth,
        OPEX grows by opex_growth. We can do a basic formula here.
        """
        base_marketing = 25000
        base_opex = 300000
        # Ex: marketing = base_marketing*(1+growth)^month_idx
        marketing = base_marketing * (1 + scenario.marketing_cost_growth)**month_idx
        opex = base_opex * (1 + scenario.opex_growth)**month_idx
        return marketing + opex

    def plot_results(self):
        """
        Basic plot for each scenario: Net Profit over time, plus Volume over time.
        """
        sns.set_style("whitegrid")
        st.subheader("Simulation Results")

        for scenario_name, df in self.results.items():
            st.write(f"**Scenario: {scenario_name}**")
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))

            # Plot Net Profit
            ax[0].plot(df["Month"], df["Net_Profit"], marker='o', color='green')
            ax[0].set_title("Net Profit")
            ax[0].set_xlabel("Month")
            ax[0].set_ylabel("Profit (€)")

            # Plot Volume
            ax[1].plot(df["Month"], df["Total_Volume"], marker='s', color='blue')
            ax[1].set_title("Total Volume")
            ax[1].set_xlabel("Month")
            ax[1].set_ylabel("Volume (€)")

            st.pyplot(fig)

    def show_dataframes(self):
        """
        Optionally show dataframes in expanders or directly.
        """
        for scenario_name, df in self.results.items():
            with st.expander(f"Data - {scenario_name}", expanded=False):
                st.dataframe(df)

    def download_excel(self):
        """
        Export all results to a single Excel file (in-memory).
        """
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for scenario_name, df in self.results.items():
                df.to_excel(writer, sheet_name=scenario_name, index=False)
        return output.getvalue()

# ---------------------------
# Streamlit App
# ---------------------------

def main():
    st.title("RunBot Simulation Platform - Parametric Streamlit Version")
    st.write("""
    This Streamlit app allows you to parameterize the simulation
    and visualize the results for multiple scenarios.
    """)

    # 1) Global parameters for the simulation
    st.sidebar.header("Global Simulation Parameters")
    months = st.sidebar.number_input("Number of months", min_value=1, max_value=120, value=24)

    # 2) Let's define a few scenarios paramétriquement
    st.sidebar.header("Scenarios Setup")
    scenario_names = ["Bear", "Neutral", "Bull"]
    scenario_list = []

    for sc_name in scenario_names:
        st.sidebar.subheader(f"Scenario: {sc_name}")
        user_growth = st.sidebar.slider(f"{sc_name} - User Growth Rate (monthly)", 0.0, 0.2, 0.05, 0.01)
        volume_growth = st.sidebar.slider(f"{sc_name} - Volume Growth Rate (monthly)", 0.0, 0.02, 0.005, 0.001)
        fee_growth = st.sidebar.slider(f"{sc_name} - Fee Growth Rate (monthly)", -0.001, 0.001, 0.0, 0.0001)
        retention = st.sidebar.slider(f"{sc_name} - Retention Rate (dummy usage)", 0.5, 1.0, 0.95, 0.01)
        marketing_cost_grow = st.sidebar.slider(f"{sc_name} - Marketing Cost Growth", 0.0, 0.02, 0.01, 0.001)
        opex_grow = st.sidebar.slider(f"{sc_name} - OPEX Growth", 0.0, 0.02, 0.01, 0.001)

        sc = Scenario(name=sc_name,
                      user_growth_rate=user_growth,
                      volume_growth_rate=volume_growth,
                      fee_growth_rate=fee_growth,
                      retention_rate=retention,
                      marketing_cost_growth=marketing_cost_grow,
                      opex_growth=opex_grow)
        scenario_list.append(sc)

    # 3) Run button
    if st.button("Run Simulation"):
        bm = BusinessModel("RunBot.io")  # Could also param via UI if desired
        sim = Simulation(bm, scenario_list, months=months)
        sim.run()
        # Show plots
        sim.plot_results()
        # Optionally show data
        sim.show_dataframes()

        # Provide a download link for Excel
        excel_data = sim.download_excel()
        st.download_button(
            label="Download Excel Results",
            data=excel_data,
            file_name="simulation_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()
