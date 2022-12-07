import pandas as pd
import numpy as np
from dataclasses import dataclass
from statistics import median
from itertools import product
from dacite import from_dict


class InputHandler:
    """
    InputHandler is initialized with GUI input field data and sensitivity study parameter
    It provides functions for systematic generation of inputset and generation of Dataclass conform input
    It provides functions for running the code and storing the results.
    """


class InputHandlerLCOE(InputHandler):
    """
    HiPowAR LCOE Tool - Input Handler
    """

    def __init__(self, df: pd.DataFrame, dc: dataclass, dict_additionalNames: dict):
        """
        Input:  df: DataFrame with with columns: 'par','parInfo','value', as created by
                    scripts.gui_functions.read_input_fields() Note:  each component gets its own InputHandler dc: Input
                    dataclass for programm
                dict_additionalNames: In case, not all required inputs for dataclass are inside df (example:
                    string "Fuel Name" or string "Name"), they can be given here.
                    Current implementation: Non-numeric values will be given in this dict. #ToDo generalize

        Creates:
            - self.list_numeric_dc_pars, list of all numeric attributes of Dataclass, as only numeric ones will be varied
            - self.dict_var_par,        dict with structure {par1: [val1,val2,...], par2: [val1,val2,val3], sorted values,
                                        used for generation of variations
        """

        self.df_input_sets = None
        self.dc = dc
        self.dict_additionalNames = dict_additionalNames
        # Get all parameter of input DataClass which are numeric (= not string)
        self.list_numeric_dc_pars = [key for key, val in dc.__dataclass_fields__.items() if val.type.__name__ != 'str']
        self.list_nonnumeric_dc_pars = [key for key, val in dc.__dataclass_fields__.items() if
                                        val.type.__name__ == 'str']

        # Reduced input dfs of numeric & non-numeric parameter from Dataclass
        df_var_par = df.loc[df.par.isin(self.list_numeric_dc_pars), :].copy()
        df_nonvar_par = df.loc[df.par.isin(self.list_nonnumeric_dc_pars), :].copy()

        # Create dict with structure {par1: [val1,val2,...], par2: [val1,val2,val3], where values are sorted
        dict_var_par = {}
        for par in self.list_numeric_dc_pars:
            val_list = list(df_var_par.loc[df_var_par.par == par, 'value'].values)
            val_list = [x for x in val_list if x is not None] # Remove None
            val_list.sort()
            dict_var_par[par] = val_list
        self.dict_var_par = dict_var_par

    def create_input_sets(self, mode: str):
        """
        Input:  mode: str
                    ["nominal","minmax1","fullfactorial"], type of input set generation.

        Descr.: Based on selected mode, DataFrame with multiple input sets will be created.
                    1) combinations of numeric parameter will be generated.
                    2) add non-numeric parameter as additional columns

        Output: self.df_input_sets: pd.DataFrame
                    Dataframe, columns: input parameter, rows: individual input sets
        """

        # Initialize DataFrame with all dataclass variables
        df = pd.DataFrame(columns=[key for key in self.dc.__dataclass_fields__])

        # mapping = {"p_size_kW": self.p.size_kW,
        #            "p_capex_Eur_kW": self.p.capex_Eur_kW,
        #            "p_opex_Eur_kWh": self.p.opex_Eur_kWh,
        #            "p_eta_perc": self.p.eta_perc,
        #            "fin_lifetime_yr": self.fin.lifetime_yr,
        #            "fin_operatinghoursyearly": self.fin.operatinghoursyearly,
        #            "fin_discountrate_perc": self.fin.discountrate_perc,
        #            "fuel_cost_Eur_per_kWh": self.fuel.cost_Eur_per_kWh}

        # 1) combinations of numeric parameter will be generated.

        # Simple input sets will be added always:
        for key, val in self.dict_var_par.items():
            df.loc["min", key] = min(val)
            df.loc["nominal", key] = median(val)
            df.loc["max", key] = max(val)

        if mode == "all":
            # Create all possible combinations.
            df2 = pd.DataFrame(list(product(*[val for key, val in self.dict_var_par.items()])),
                               columns=[key for key, val in self.dict_var_par.items()])
            df = pd.concat([df, df2])

        elif mode == "nominal":
            # ... no additional input sets will be generated, as "nominal" is always included
            pass

        elif mode == "all_minmax":
            # Only vary one input variable to min and max, keep all others at nominal value.

            # All possible combinations of all min & max values
            l1 = [val for key, val in self.dict_var_par.items()]
            l2 = [list(set([min(le), max(le)])) for le in l1]

            df2 = pd.DataFrame(list(product(*l2)),
                               columns=[key for key, val in self.dict_var_par.items()])
            df = pd.concat([df, df2])

            # All possible combinations of one variable min & max values, all others nominal
            df3 = pd.DataFrame(columns=[key for key, val in self.dict_var_par.items()])
            for key, val in self.dict_var_par.items():
                # Create dict
                dataset_min = df.loc["nominal"].to_dict()
                dataset_max = df.loc["nominal"].to_dict()
                dataset_min[key] = min(val)
                dataset_max[key] = max(val)
                df3 = pd.concat([df3, pd.DataFrame([dataset_max, dataset_min])])

            df = pd.concat([df, df3])

        # 2) add non-numeric parameter as additional columns
        for key, val in self.dict_additionalNames.items():
            df.loc[:, key] = val

        # 3) Add additional column with dataclass inside
        df.loc[:, "dataclass"] = df.apply(lambda row: from_dict(data_class=self.dc, data=row.to_dict()), axis=1)

        self.df_input_sets = df

    def submit_job(self, prog):
        """
        Input: progr    Program or function to execute.
        Descr.: Uses self.df_input_sets and

        Output:
            - Valid Input for calculation (DataClass), Validity check will be done by programm
        """


if __name__ == '__main__':
    ...
