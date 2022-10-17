from dataclasses import dataclass

@dataclass
class DC_SystemInput:
    """..."""
    name: str  # Component name, as "HiPowAr" or "ICE"
    size_kW: float  # [kW]
    capex_Eur_kW: float  # [€/kW]
    opex_Eur_kWh: float  # [€/kWh], without fuel
    eta_perc: float  # [%]


@dataclass
class DC_FinancialInput:
    """..."""
    discountrate_perc: float  # [%]
    lifetime_yr: float  # [yr]
    operatinghoursyearly: float  # [hr/yr]

@dataclass
class DC_FuelInput:
    """..."""
    name: str  # fuel name, as "NH3","NG",...
    cost_Eur_per_kWh: float  # [€/kWh]
    costIncrease_percent_per_year: float  # [%]


def readCellInput():
    ...

if __name__ == "__main__":
    fuel = DC_FuelInput(name="nh3",cost_Eur_per_kW=2,costincrease_percent_per_year=1)
