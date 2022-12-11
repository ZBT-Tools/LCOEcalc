from scripts.data_handler import DataHandler
import pandas as pd
from dacite import from_dict
from dataclasses import dataclass
from scripts.multiplication import multiplication

class DataHandlerSimpleApp(DataHandler):

    def create_input_sets(self) -> pd.DataFrame:
        """
        Single nominal input set
        :return:
        """
        # Initialize DataFrame with all dataclass variables
        df = pd.DataFrame(columns=[key for key in self.dc.__dataclass_fields__])

        # Add nominal input set
        for key, val in self.dict_var_par.items():
            df.loc["nominal", key] = val[0]

        # Add additional column with dataclass inside
        df.loc[:, "dataclass"] = df.apply(lambda row: from_dict(data_class=self.dc, data=row.to_dict()), axis=1)

        self.df_input_sets = df

    def submit_job(self):
        self.df_results = super().submit_job(multiplication, self.df_input_sets)


