# App styling and input functions for recurring use
# ----------------------------------------------------------------------------------------------------------------------
import random

import dash_bootstrap_components as dbc
import pandas as pd


def styling_input_row_generic(component: str, par: str, title: str, n_inputfields: int = 3, fieldtype: list = None,
                              parInfo: list = None, widths: list = None, xl_widths: list = None,
                              disabled: list = None) -> dbc.Row:
    """
    Creates dbc row with title and input fields.
    Example: Row title and 3 input fields -->    || Capex [%]   [...]  [...] [...] ||

    Structure: dbc.Row([dbc.Col(),dbc.Col(),...])

    Input field identifier 'id' is dict of type:
        id={'type': ...,'component', ..., 'par': ..., },
        'type' defines type of field, e.g. 'input',...
        'component' defines system, field is related to, e.g. 'SOFC'
        'par' defines system parameter, e.g. 'OPEX_€_per_kW'
        'parInfo' defines additional information, e.g. 'nominal', ',min', 'max'

        Example: id = {'type': 'input', 'component': 'SOFC', 'par': 'CAPEX_€_per_kW', 'parInfo':'min}

    :param fieldtype: str = list of field type for each input field, default handling below
    :param component:       used as field identifier
    :param par:             used as field identifier
    :param title:           Row title
    :param n_inputfields:   number of input fields
    :param parInfo:         list of field identifiers, default handling below
    :param widths:          list of width for each input field, default handling below
    :param disabled:        option to disable input field , default handling below
    :return:
    """
    # Default input field type
    if fieldtype is None:
        fieldtype = ['input'] * n_inputfields

    # Default column/field widths based on n_inputfields
    if widths is None:
        if n_inputfields == 1:
            widths = [12, 12]
            xl_widths = [6, 2]
        elif n_inputfields == 2:
            widths =[12, 6, 6]
            xl_widths = [6, 2, 2]
        elif n_inputfields == 3:
            widths = [12, 4, 4, 4]
            xl_widths = [6, 2, 2, 2]
        else:
            equal_wd = int(12 / n_inputfields)
            widths = [equal_wd] * n_inputfields

    # Default postfixes (...as 'min', 'max',...)
    if parInfo is None:
        if n_inputfields == 1:
            parInfo = ["nominal"]  # Nominal value only, no postfix required
        elif n_inputfields == 3:
            parInfo = ["nominal", "min", "max"]
        else:
            parInfo = [f"_{n}" for n in range(n_inputfields)]

    # Default non-disabled input fields
    if disabled is None:
        disabled = [False] * n_inputfields

    # First column: Label
    row_columns = [dbc.Col(dbc.Label(title), width=widths[0], xl=xl_widths[0])]

    # Add input-Fields
    for t, w, xlw, d, p in zip(fieldtype, widths[1:], xl_widths[1:], disabled, parInfo):
        col = dbc.Col(dbc.Input(id={'type': t,
                                    'component': component,
                                    'par': par,
                                    'parInfo': p}, type="number", size="sm",
                                disabled=d), width=w, xl=xlw),
        if type(col) == tuple:
            col = col[0]
        row_columns.append(col)

    return dbc.Row(row_columns)


# General Card definition with input rows
def styling_input_card_generic(component: str, header: str, rowinputs: list) -> dbc.Card:
    """
    Creates dbc.Card with header and input rows: "input_row_generic"s
    :param rowinputs:   dict with input_row_generic input information,
                            structure:  [inputrow_dict, inputrow_dict, inputrow_dict,...]
                                        [{'par': ..., 'title': 'inputrow', 'n_inputfields': ...}, ...]
                            example:    [{'par': 'size_kW', "title": "El. Output [kW]", "n_inputfields": 1}, ...],

    :param component:   ... name for all rows of card
    :param header:      card title
    :return:
    """

    # LCOE Tool specific column definition: 4 Columns
    rows = [dbc.Row([dbc.Col(width=12, xl=6),
                     dbc.Col(dbc.Label("Nominal"), width=4, xl=2),
                     dbc.Col(dbc.Label("Min"), width=4, xl=2),
                     dbc.Col(dbc.Label("Max"), width=4, xl=2)
                     ])]
    # Create rows
    rws = [styling_input_row_generic(component=component, par=rw["par"], title=rw["title"], n_inputfields=3) for rw in
           rowinputs]
    rows.extend(rws)

    # Create Card
    card = dbc.Card([
        dbc.CardHeader(header),
        dbc.CardBody(
            rows
        )])
    return card


def styling_input_card_component(component: str, header: str, add_rows: list = None) -> dbc.Card:
    """
    Creates dbc.Card with header and HiPowAR LCOE Workpackage specific Component input rows:

    :param add_rows:    optional list of dicts for additional input_row_generic rows
    :param component:   ... name for all rows of card
    :param header:      card title
    :return:
    """

    # Standard LCOE Component input
    LCOE_rowInput = [{'par': "size_kW", "title": "El. Output [kW]", "n_inputfields": 1},
                     {'par': "capex_Eur_kW", "title": "Capex [€/kW]", "n_inputfields": 3},
                     {'par': "opex_Eur_kWh", "title": "Opex (no Fuel) [€/kWh]", "n_inputfields": 3},
                     {'par': "eta_perc", "title": "Efficiency [%]", "n_inputfields": 3}]

    if add_rows is not None:
        LCOE_rowInput.extend(add_rows)
    card = styling_input_card_generic(component, header, LCOE_rowInput)
    return card


def styling_generic_dropdown(id_name: str, label: str, elements: list) -> dbc.DropdownMenu:
    """

    :param id_name: dash component name
    :param label: label
    :param elements: list of dropdown menu items, ID is generated like {id_name}_{list counter}
    :return:
    """
    dropdown = dbc.DropdownMenu(
        id=id_name,
        label=label,
        children=[dbc.DropdownMenuItem(el, id=f"{id_name}_{ct}", n_clicks=0) for ct, el in enumerate(elements)],
        className="d-grid gap-2"
    )
    return dropdown


def build_randomfill_input_fields(output: list) -> list:
    """
    Description:
    output: (Portion of) ctx.output_lists of apropriate callback

    Function for filling input fields with random data. For each element inside "output"
    (list of lists or single list)...
    """
    # For multiple outputs in callback, 'output' is list of lists [[output1],[output2],...]
    # If only one output is handed over, it will be wrapped in additional list
    if type(output[0]) is not list:
        output = [output]

    return_lists = []
    for li in output:
        return_list = []
        for _ in li:
            try:
                return_list.append(random.randrange(0, 100))
            except AttributeError:
                return_list.append(None)
            except ValueError:
                return_list.append(None)
        return_lists.append(return_list)
    return return_lists


def build_initial_collect(state_list: list):
    """
    Build function to create initial excel input table
    Input: One list from ctx.states_list, e.g. ctx.states_list[0]
    """
    df = pd.DataFrame(columns=["component", "par", "parInfo"])

    for dct in state_list:
        data = {'component': dct["id"]["component"], 'par': dct["id"]["par"], 'parInfo': dct["id"]["parInfo"]}
        try:
            data.update({0: dct["value"]})
        except KeyError:
            data.update({0: None})
        new_row = pd.Series(data)
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    return df


def fill_input_fields(input_str: str, df: pd.DataFrame, output: list) -> list:
    """
    Input:
        input_str: Preset name, selected by dropdown menu
        df: Input data table (Excel definition file)
        output: (Portion of) ctx.output_lists of apropriate callback

    Definition:
        Function for filling input fields based on preset dropdown menu selection. For each element inside "output"
        (list of lists or single list), appropriate data (component, par, parInfo) from df will be returned.
    """
    # For multiple outputs in callback, 'output' is list of lists [[output1],[output2],...]
    # If only one output is handed over, it will be wrapped in additional list
    if type(output[0]) is not list:
        output = [output]

    return_lists = []
    for li in output:
        return_list = []
        for el in li:
            comp = el["id"]["component"]
            par = el["id"]["par"]
            parInfo = el["id"]["parInfo"]
            try:
                return_list.append(
                    df.loc[(df.component == comp) & (df.par == par) & (df.parInfo == parInfo), input_str].item())
            except AttributeError:
                return_list.append(None)
            except ValueError:
                return_list.append(None)
        return_lists.append(return_list)
    return return_lists


def read_input_fields(state_selection: list) -> pd.DataFrame:
    """
    state_selection: Callback State ctx.states_list[:], can be list or list of lists

    Function reads state_selection and writes data into DataFrame
    """
    # For multiple states in callback, 'state_selection' is list of lists [[state1],[state2],...]
    # If only one state is passed, wrap it into list:
    if type(state_selection[0]) is not list:
        state_selection = [state_selection]

    #  Collect data of input fields in dataframe
    df = pd.DataFrame()
    for state_list in state_selection:
        for el in state_list:
            el_dict = {'component': el['id']['component'],
                       'par': el['id']['par'],
                       'parInfo': el['id']['parInfo']}
            try:
                el_dict.update({'value': el['value']})
            except KeyError:
                el_dict.update({'value': None})

            new_row = pd.Series(el_dict)
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

    return df
