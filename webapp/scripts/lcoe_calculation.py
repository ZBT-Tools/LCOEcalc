import pandas as pd
from dataclasses import dataclass
from webapp.scripts.data_handler import DataHandlerLCOE


@dataclass
class DataclassLCOEsimpleInput:
    """..."""
    name: str  # Component name, as "HiPowAr" or "ICE"
    size_kW: float  # [kW]
    capex_Eur_kW: float  # [€/kW]
    opex_Eur_kWh: float  # [€/kWh], without fuel
    eta_perc: float  # [%]
    discountrate_perc: float  # [%]
    cost_CO2_per_tonne: float  # [€/T_CO2]
    CO2_costIncrease_percent_per_year: float  # [%]
    lifetime_yr: float  # [yr]
    operatinghoursyearly: float  # [hr/yr]
    fuel_name: str  # fuel name, as "NH3","NG",...
    fuel_cost_Eur_per_kWh: float  # [€/kWh]
    fuel_costIncrease_percent_per_year: float  # [%]
    fuel_CO2emission_tonnes_per_MWh: float  # [T_CO/MWh]


def lcoe(inp: DataclassLCOEsimpleInput):
    """
    Input as defined in dataclass
    Output: LCOE [€/kWh]
    """

    df = pd.DataFrame(index=range(0, int(inp.lifetime_yr) + 1, 1),
                      columns=["Investment", "OM", "Fuel", "Power", "CO2_Emission_Tonnes",
                               "CO2_Emission_Cost"])
    df.loc[0, :] = 0

    # Investment Costs (only in first year)
    # ----------------------------
    df.loc[0, "Investment"] = inp.size_kW * inp.capex_Eur_kW
    df.loc[1:, "Investment"] = 0

    # OPEX Costs
    # ----------------------------
    df.loc[1:, "OM"] = inp.size_kW * inp.operatinghoursyearly * inp.opex_Eur_kWh

    # Fuel Costs
    # ----------------------------
    df.loc[:,"Fuel_cost_Eur_per_kWh"] = inp.fuel_cost_Eur_per_kWh * (1 + inp.fuel_costIncrease_percent_per_year / 100)**df.index
    df.loc[1:,
    "Fuel"] = inp.size_kW * inp.operatinghoursyearly *  df.loc[1:,"Fuel_cost_Eur_per_kWh"] * 100 / inp.eta_perc

    # CO2 Emission [Tonnes]
    # ----------------------------
    df.loc[1:, "CO2_Emission_Tonnes"] = inp.fuel_CO2emission_tonnes_per_MWh * inp.size_kW / 1000 * \
                                        inp.operatinghoursyearly * 100 / inp.eta_perc

    # CO2 Emission Cost
    # ----------------------------
    df.loc[:,"CO2_Cost_per_tonne"] = inp.cost_CO2_per_tonne * (1 + inp.CO2_costIncrease_percent_per_year / 100)**df.index
    df.loc[:, "CO2_Emission_Cost"] = df.loc[:, "CO2_Emission_Tonnes"] * df.loc[:,"CO2_Cost_per_tonne"]

    # Combined Cost
    # ----------------------------
    df.loc[:, "Cost_combined"] = df.loc[:, "Investment"] + df.loc[:, "OM"] + df.loc[:, "Fuel"] + df.loc[:, "CO2_Emission_Cost"]

    # Combined Cumulative Cost
    # ----------------------------
    df.loc[:, "Cost_combined_cum"] = df.loc[:, "Cost_combined"].cumsum()

    # Electricity Generation
    # ----------------------------
    df.loc[1:, "Power"] = inp.size_kW * inp.operatinghoursyearly

    # Financial Discounting of costs
    # ----------------------------
    df["Investment_fin"] = df.apply(
        lambda row: row.Investment / (1 + inp.discountrate_perc / 100) ** int(row.name),
        axis=1)
    df["OM_fin"] = df.apply(lambda row: row.OM / (1 + inp.discountrate_perc / 100) ** int(row.name),
                            axis=1)
    df["Fuel_fin"] = df.apply(
        lambda row: row.Fuel / (1 + inp.discountrate_perc / 100) ** int(row.name), axis=1)
    df["CO2_Emission_Cost_fin"] = df.apply(
        lambda row: row.CO2_Emission_Cost / (1 + inp.discountrate_perc / 100) ** int(row.name),
        axis=1)
    df["Power_fin"] = df.apply(
        lambda row: row.Power / (1 + inp.discountrate_perc / 100) ** int(row.name),
        axis=1)

    lcoe_val = (df["Investment_fin"].sum() + df["OM_fin"].sum() + df["Fuel_fin"].sum() +
                df["CO2_Emission_Cost_fin"].sum()) / df["Power_fin"].sum()  # [€/kWh]

    return lcoe_val,df


def multisystem_calculation(df: pd.DataFrame, system_names: list, fuel_names: list, fuel_prop: dict,
                            mode: str):
    """
    Initialize InputHandlerLOCE-object for all systems and fuel combinations
    """
    dict_systems = {}

    for system in system_names:
        for fuel in fuel_names:
            if (system != "HiPowAR") or (fuel != "Fuel_NG"):
                # Reduce to one system and one fuel
                dfred = df.loc[(df.component == system) |
                               (df.component == "Financials") |
                               (df.component == fuel), :]

                # Init Input Handler
                inputhandler = DataHandlerLCOE(df=dfred, dc=DataclassLCOEsimpleInput,
                                               dict_additionalNames={"name": system, "fuel_name": fuel,
                                                                     **fuel_prop[fuel]})
                inputhandler.create_input_sets(mode=mode)
                inputhandler.submit_job(func=lcoe, resultcolumn="LCOE", mode=mode)
                dict_systems.update({f"{system}_{fuel[5:]}": inputhandler})

    return dict_systems


if __name__ == '__main__':
    dc = DataclassLCOEsimpleInput(eta_perc=99,
                                  name='Dream',
                                  fuel_name='Air',
                                  lifetime_yr=100,
                                  discountrate_perc=1,
                                  capex_Eur_kW=100,
                                  opex_Eur_kWh=0.1,
                                  operatinghoursyearly=6000,
                                  size_kW=1000,
                                  fuel_cost_Eur_per_kWh=0.1,
                                  fuel_costIncrease_percent_per_year=0,
                                  )

    # Print parameters of DataclassLCOEsimpleInput
    for key, val in DataclassLCOEsimpleInput.__dataclass_fields__.items():
        if val.type.__name__ != 'str':
            print(key)
