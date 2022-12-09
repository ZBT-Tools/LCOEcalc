# App styling and input functions for recurring use
# ----------------------------------------------------------------------------------------------------------------------
import random

import dash_bootstrap_components as dbc
import pandas as pd


def styling_input_row_generic(label: str,
                              row_id_dict: dict,
                              n_inputfields: int,
                              field_id: dict,
                              widths: list,
                              fieldtype: list = None,
                              disabled: list = None) -> dbc.Row:
    """
    Creates dbc row with title and input fields.
    Example: Row title and 3 input fields -->    || Capex [%]   [...]  [...] [...] ||

    Structure: dbc.Row([dbc.Col(),dbc.Col(),...])


    :param label:           Row title
    :param row_id_dict:     Part of field id which is identical for all fields in row.
                            Example: {'type': 'input', 'component': 'SOFC', 'par': 'CAPEX_€_per_kW'}
    :param field_id:        Part of field id, which distinguish fields from each other.
                            Structure: {par:[str1,str2,str3]} for 3 input fields.
                            Example: {"valuetype":["nominal","min","max"]}
    :param n_inputfields:   number of data fields
    :param widths:          dbc width definition for title (1) and input fields (2)
                            Example [{"width":12,"xl:6}{"width":4,"xl:2}]
    :param fieldtype:            Option to change input type, default is "number"
    :param disabled:        option to disable input field , default handling below
    :return:
    """

    # Default field type 'number'
    if fieldtype is None:
        fieldtype = ['number'] * n_inputfields
    # Default non-disabled input fields
    if disabled is None:
        disabled = [False] * n_inputfields

    # First column: Label
    row_columns = [dbc.Col(dbc.Label(label), **widths[0])]

    # # Add input-Fields
    for n in range(n_inputfields):
        dct_id = row_id_dict
        fieldspec_id = {k: v[n] for (k, v) in field_id.items()}
        dct_id.update(fieldspec_id)  # Merge row and field id

        col = dbc.Col(dbc.Input(id=dct_id,
                                type=fieldtype[n],
                                disabled=[disabled[n]],
                                size="sm"),
                      **widths[1])

        # if type(col) == tuple: #
        #     col = col[0]
        row_columns.append(col)

    return dbc.Row(row_columns)


def test_styling_input_row_generic():
    row = styling_input_row_generic(label="testrow",
                                    row_id_dict={"type": "input", "rownumber": 1},
                                    n_inputfields=3,
                                    field_id={"fieldnumber": [1, 2, 3], "fielddescr": ["min", "max", "other"]},
                                    widths=[{"width": 12, "xl": 6}, {"width": 4, "xl": 2}], )

    return row


def styling_input_rows_generic(identic_row_input_dict: dict,
                               specific_row_input: list) -> list:
    """
    Description:
        Creates list of styling_input_row_generic(...) Input rows.
        identic_row_input_dict parameter will be used for all input fields.
        specific_row_input parameter will be used for individual rows.

    Input:
        identic_row_id_dict
        specificRowInput:  list of dicts with input_row_generic input information not given in identicalRowInput,
                            structure:  [inputrow_dict, inputrow_dict, inputrow_dict,...]
                                        [{'par': ..., 'title': 'inputrow', 'n_inputfields': ...}, ...]
                            example:    [{'par': 'size_kW', "title": "El. Output [kW]", "n_inputfields": 1}, ...]
    """
    rws = []
    for specific_row in specific_row_input:
        row_def = identic_row_input_dict
        row_def.update(specific_row)

        rws.append(styling_input_row_generic(**row_def))

    return rws


def styling_input_card_generic(header: str,
                               firstRow: list,
                               rws: list) -> dbc.Card:
    """
    Description:
        Creates dbc.card with header and multiple input rows generate by styling_input_rows_generic.
    Input:
        header:     card title
        firstRow:   definition of first row
        rws:        list of input rows
    """
    rows = firstRow.extend(rws)

    # Create Card
    card = dbc.Card([
        dbc.CardHeader(header),
        dbc.CardBody(
            rows
        )])
    return card


def styling_input_row_LCOE_generic(label: str,
                                   n_inputfields: int,
                                   component: str,
                                   par: str
                                   ) -> dbc.Row:
    """

    :param label:
    :param n_inputfields:
    :param component:
    :param par:
    :return:
    """
    # Specific LCOE Tool arguments for styling_input_row_generic()
    # -----------------------------------------------------------------------
    row_id_dict = {"type": "input", "component": component, "par": par}

    # parInfo & Width definition
    if n_inputfields == 1:
        field_id = {"parInfo": ["nominal"]}
        widths = [{"width": 12, "xl": 6}, {"width": 12, "xl": 2}]
    elif n_inputfields == 3:
        field_id = {"parInfo": ["nominal", "min", "max"]}
        widths = [{"width": 12, "xl": 6}, {"width": 4, "xl": 2}]
    else:
        print(f"Handling for n_inputfields ={n_inputfields} not defined in styling_input_row_LCOE_generic()")
        field_id = None
        widths = None

    row = styling_input_row_generic(label=label,
                                    row_id_dict=row_id_dict,
                                    n_inputfields=n_inputfields,
                                    field_id=field_id,
                                    widths=widths)

    return row


def styling_input_rows_LCOE_generic(identic_row_input_dict: dict,
                                    specific_row_input: list) -> list:
    """
    No specific LCOE input inside function, but uses styling_input_row_LCOE_generic() to create
    input rows

    :param identic_row_input_dict:
    :param specific_row_input:
    :return:
    """
    rws = []
    for specific_row in specific_row_input:
        row_def = identic_row_input_dict
        row_def.update(specific_row)

        rws.append(styling_input_row_LCOE_generic(**row_def))

    return rws


def styling_input_card_LCOE_generic(header: str,
                                    rws: list) -> dbc.Card:
    """
    Description:
        Creates dbc.card with header and multiple input rows generate by styling_input_rows_generic.
    Input:
        header:     card title
        firstRow:   definition of first row
        rws:        list of input rows
    """
    # LCOE Tool specific first row definition
    fist_row = [dbc.Row([dbc.Col(width=12, xl=6),
                         dbc.Col(dbc.Label("Nominal"), width=4, xl=2),
                         dbc.Col(dbc.Label("Min"), width=4, xl=2),
                         dbc.Col(dbc.Label("Max"), width=4, xl=2)
                         ])]

    rows = fist_row.extend(rws)

    # Create Card
    card = dbc.Card([
        dbc.CardHeader(header),
        dbc.CardBody(
            rows
        )])
    return card


def styling_input_card_LCOE_component(header: str,
                                      component: str,
                                      add_rows: list = None) -> dbc.Card:
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
    card = styling_input_card_LCOE_generic(component, header, LCOE_rowInput)
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


if __name__ == "__main__":
    print("hi")
    a = test_styling_input_row_generic()
