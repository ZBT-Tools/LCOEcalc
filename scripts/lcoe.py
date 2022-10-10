from dataclasses import dataclass

import pandas as pd


@dataclass
class DC_SystemInput:
    """..."""
    name: str  # Component name, as "HiPowAr" or "ICE"
    size: float  # [kW]
    capex: float  # [€/kW]
    opex: float  # [€/kWh], without fuel
    eta: float  # [%]


@dataclass
class DC_FinancialInput:
    """..."""
    discountrate: float  # [%]
    lifetime: float  # [yr]
    yrly_operatinghours: float  # [hr/yr]


@dataclass
class DC_FuelInput:
    """..."""
    name: str  # fuel name, as "NH3","NG",...
    cost: float  # [€/kWh]
    yrly_costincrease: float  # [%]


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
        df = pd.DataFrame(index=range(0, self.fin.lifetime + 1, 1), columns=["Investment", "OM", "Fuel", "Power"])
        df.loc[0, :] = 0

        # Investment Costs
        # ----------------------------
        df.loc[0, "Investment"] = self.par.size * self.par.capex
        df.loc[1:, "Investment"] = 0

        # O&M Costs
        # ----------------------------
        df.loc[1:, "OM"] = self.par.size * self.fin.yrly_operatinghours * self.par.opex

        # Fuel Costs
        # ----------------------------
        df.loc[1:, "Fuel"] = self.par.size * self.fin.yrly_operatinghours * self.fuel.cost * 100 / self.par.eta

        # Electricity Geeration
        # ----------------------------
        df.loc[1:, "Power"] = self.par.size * self.fin.yrly_operatinghours

        # Financial Accumulation
        # ----------------------------
        df["Investment_fin"] = df.apply(lambda row: row.Investment / (1 + self.fin.discountrate / 100) ** int(row.name),
                                        axis=1)
        df["OM_fin"] = df.apply(lambda row: row.OM / (1 + self.fin.discountrate / 100) ** int(row.name), axis=1)
        df["Fuel_fin"] = df.apply(lambda row: row.Fuel / (1 + self.fin.discountrate / 100) ** int(row.name), axis=1)
        df["Power_fin"] = df.apply(lambda row: row.Power / (1 + self.fin.discountrate / 100) ** int(row.name), axis=1)

        self.lcoe_df = df
        self.lcoe_val = (df["Investment_fin"].sum() + df["OM_fin"].sum() + df["Fuel_fin"].sum()) / \
                        df["Power_fin"].sum()  # [€/kWh]


if __name__ == "__main__":
    sysinp = DC_SystemInput(name="hipowar", size=100, capex=1000, opex=0.01, eta=50)
    financialinp = DC_FinancialInput(discountrate=3, lifetime=10, yrly_operatinghours=6000)
    fuelinp = DC_FuelInput(name="NH3", cost=0.05, yrly_costincrease=0)

    hipowar = System(sysinp)
    hipowar.load_fuel_par(fuelinp)
    hipowar.load_financial_par(financialinp)

    hipowar.lcoe()
