#
# from data_transfer import DC_FuelInput,DC_FinancialInput,DC_SystemInput
import pandas as pd
from statistics import median
from itertools import product


class System:
    """..."""

    def __init__(self, DC_SystemInput):
        self.lcoe_table = None
        self.fin = None
        self.fuel = None
        self.p = DC_SystemInput

    def load_financial_par(self, DC_FinancialInput):
        self.fin = DC_FinancialInput

    def load_fuel_par(self, DC_FuelInput):
        self.fuel = DC_FuelInput

    def prep_lcoe_input(self, mode="minmax"):
        df = pd.DataFrame(columns=["p_size_kW",
                                   "p_capex_Eur_kW",
                                   "p_opex_Eur_kWh",
                                   "p_eta_perc",
                                   "fin_lifetime_yr",
                                   "fin_operatinghoursyearly",
                                   "fin_discountrate_perc",
                                   "fuel_cost_Eur_per_kWh"])

        mapping = {"p_size_kW": self.p.size_kW,
                   "p_capex_Eur_kW": self.p.capex_Eur_kW,
                   "p_opex_Eur_kWh": self.p.opex_Eur_kWh,
                   "p_eta_perc": self.p.eta_perc,
                   "fin_lifetime_yr": self.fin.lifetime_yr,
                   "fin_operatinghoursyearly": self.fin.operatinghoursyearly,
                   "fin_discountrate_perc": self.fin.discountrate_perc,
                   "fuel_cost_Eur_per_kWh": self.fuel.cost_Eur_per_kWh}

        # Simplest parameter: ALl min, all nominal, all max

        for key, val in mapping.items():
            df.loc["min", key] = min(val)
            df.loc["nominal", key] = median(val)
            df.loc["max", key] = max(val)

        if mode == "all":
            # All combinations
            df2 = pd.DataFrame(list(product(*[val for key, val in mapping.items()])),
                               columns=[key for key, val in mapping.items()])
            df = pd.concat([df, df2])

        elif mode == "all_minmax":
            # All combinations of each min and max
            # ToDo: Wir brauchen Tabelle, bei der alles systematisch variiert wird.
            # Also immer einen Min/Max, rest nominal

            # All possible combinations of all min & max values
            l1 = [val for key, val in mapping.items()]
            l2 = [list(set([min(le), max(le)])) for le in l1]

            df2 = pd.DataFrame(list(product(*l2)),
                               columns=[key for key, val in mapping.items()])
            df = pd.concat([df, df2])

            # All possible combinations of one variable min & max values, all others nominal
            df3 = pd.DataFrame(columns=[key for key, val in mapping.items()])
            for key, val in mapping.items():
                # Create dict
                dataset_min = df.loc["nominal"].to_dict()
                dataset_max = df.loc["nominal"].to_dict()
                dataset_min[key] = min(val)
                dataset_max[key] = max(val)
                df3 = pd.concat([df3, pd.DataFrame([dataset_max, dataset_min])])

            df = pd.concat([df, df3])





        self.lcoe_table = df

    def lcoe(self, inp: pd.core.series.Series):
        """..."""
        df = pd.DataFrame(index=range(0, int(inp.fin_lifetime_yr) + 1, 1),
                          columns=["Investment", "OM", "Fuel", "Power"])
        df.loc[0, :] = 0

        # Investment Costs
        # ----------------------------
        df.loc[0, "Investment"] = inp.p_size_kW * inp.p_capex_Eur_kW
        df.loc[1:, "Investment"] = 0

        # O&M Costs
        # ----------------------------
        df.loc[1:, "OM"] = inp.p_size_kW * inp.fin_operatinghoursyearly * inp.p_opex_Eur_kWh

        # Fuel Costs
        # ----------------------------
        df.loc[1:,
        "Fuel"] = inp.p_size_kW * inp.fin_operatinghoursyearly * inp.fuel_cost_Eur_per_kWh * 100 / inp.p_eta_perc

        # Electricity Generation
        # ----------------------------
        df.loc[1:, "Power"] = inp.p_size_kW * inp.fin_operatinghoursyearly

        # Financial Accumulation
        # ----------------------------
        df["Investment_fin"] = df.apply(
            lambda row: row.Investment / (1 + inp.fin_discountrate_perc / 100) ** int(row.name),
            axis=1)
        df["OM_fin"] = df.apply(lambda row: row.OM / (1 + inp.fin_discountrate_perc / 100) ** int(row.name), axis=1)
        df["Fuel_fin"] = df.apply(lambda row: row.Fuel / (1 + inp.fin_discountrate_perc / 100) ** int(row.name), axis=1)
        df["Power_fin"] = df.apply(lambda row: row.Power / (1 + inp.fin_discountrate_perc / 100) ** int(row.name),
                                   axis=1)

        lcoe_val = (df["Investment_fin"].sum() + df["OM_fin"].sum() + df["Fuel_fin"].sum()) / \
                   df["Power_fin"].sum()  # [€/kWh]

        return lcoe_val


if __name__ == "__main__":
    sysinp = DC_SystemInput(name="hipowar", size=100, capex_Eur_kW=1500, opex_Eur_kWh=1, eta_perc=50)
    financialinp = DC_FinancialInput(discountrate_perc=3, lifetime_yr=10, operatinghoursyearly=6000)
    fuelinp = DC_FuelInput(name="NH3", cost_Eur_per_kW=0.05, costincrease_percent_per_year=0)

    hipowar = System(sysinp)
    hipowar.load_fuel_par(fuelinp)
    hipowar.load_financial_par(financialinp)

    hipowar.lcoe()
