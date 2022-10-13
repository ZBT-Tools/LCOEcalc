
from data_transfer import DC_FuelInput,DC_FinancialInput,DC_SystemInput
import pandas as pd


class System:
    """..."""

    def __init__(self, DC_SystemInput):
        self.lcoe_val = None
        self.fin = None
        self.fuel = None
        self.par = DC_SystemInput

    def load_financial_par(self, DC_FinancialInput):
        self.fin = DC_FinancialInput

    def load_fuel_par(self, DC_FuelInput):
        self.fuel = DC_FuelInput

    def lcoe(self):
        """..."""
        df = pd.DataFrame(index=range(0, self.fin.lifetime_yr + 1, 1), columns=["Investment", "OM", "Fuel", "Power"])
        df.loc[0, :] = 0

        # Investment Costs
        # ----------------------------
        df.loc[0, "Investment"] = self.par.size * self.par.capex_Eur_kW
        df.loc[1:, "Investment"] = 0

        # O&M Costs
        # ----------------------------
        df.loc[1:, "OM"] = self.par.size * self.fin.operatinghoursyearly * self.par.opex_Eur_kWh

        # Fuel Costs
        # ----------------------------
        df.loc[1:, "Fuel"] = self.par.size * self.fin.operatinghoursyearly * self.fuel.cost_Eur_per_kW * 100 / self.par.eta_perc

        # Electricity Geeration
        # ----------------------------
        df.loc[1:, "Power"] = self.par.size * self.fin.operatinghoursyearly

        # Financial Accumulation
        # ----------------------------
        df["Investment_fin"] = df.apply(lambda row: row.Investment / (1 + self.fin.discountrate_perc / 100) ** int(row.name),
                                        axis=1)
        df["OM_fin"] = df.apply(lambda row: row.OM / (1 + self.fin.discountrate_perc / 100) ** int(row.name), axis=1)
        df["Fuel_fin"] = df.apply(lambda row: row.Fuel / (1 + self.fin.discountrate_perc / 100) ** int(row.name), axis=1)
        df["Power_fin"] = df.apply(lambda row: row.Power / (1 + self.fin.discountrate_perc / 100) ** int(row.name), axis=1)

        self.lcoe_df = df
        self.lcoe_val = (df["Investment_fin"].sum() + df["OM_fin"].sum() + df["Fuel_fin"].sum()) / \
                        df["Power_fin"].sum()  # [€/kWh]


if __name__ == "__main__":
    sysinp = DC_SystemInput(name="hipowar", size = 100, capex_Eur_kW=1500, opex_Eur_kWh=1, eta_perc=50 )
    financialinp = DC_FinancialInput(discountrate_perc=3, lifetime_yr=10, operatinghoursyearly=6000)
    fuelinp = DC_FuelInput(name="NH3", cost_Eur_per_kW=0.05, costincrease_percent_per_year=0)

    hipowar = System(sysinp)
    hipowar.load_fuel_par(fuelinp)
    hipowar.load_financial_par(financialinp)

    hipowar.lcoe()
