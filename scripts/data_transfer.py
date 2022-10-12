from dataclasses import dataclass

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


def readCellInput():
    ...