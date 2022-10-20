from dataclasses import dataclass, fields


def sortlists(self):
    for field in fields(self):
        attr = getattr(self, field.name)
        if type(attr) == list:
            attr.sort()
            setattr(self, field.name, attr)

        print(field.name, attr)


@dataclass
class DC_SystemInput:
    """..."""
    name: str  # Component name, as "HiPowAr" or "ICE"
    size_kW: list[float]  # [kW]
    capex_Eur_kW: list[float]  # [€/kW]
    opex_Eur_kWh: list[float]  # [€/kWh], without fuel
    eta_perc: list[float]  # [%]

    def __post_init__(self):
        sortlists(self)


@dataclass
class DC_FinancialInput:
    """..."""
    discountrate_perc: list[float]  # [%]
    lifetime_yr: list[float]  # [yr]
    operatinghoursyearly: list[float]  # [hr/yr]

    def __post_init__(self):
        sortlists(self)


@dataclass
class DC_FuelInput:
    """..."""
    name: str  # fuel name, as "NH3","NG",...
    cost_Eur_per_kWh: list[float]  # [€/kWh]
    costIncrease_percent_per_year: list[float]  # [%]

    def __post_init__(self):
        sortlists(self)


def readCellInput():
    ...


if __name__ == "__main__":
    fuel = DC_FuelInput(name="nh3", cost_Eur_per_kW=2, costincrease_percent_per_year=1)
