from scripts.gui_functions import style_inpRow_generic
import dash_bootstrap_components as dbc


def style_inpRow_simpleapp(label: str, ident: str) -> dbc.Row:
    row = style_inpRow_generic(label=label,
                               row_id_dict={"type": "input"},
                               n_inputfields=1,
                               field_id={"id": [ident]},
                               widths=[{"width": 12, "xl": 6}, {"width": 12, "xl": 6}])
    return row


def style_inpRows_simpleapp(specific_row_input: list) -> list:
    """
    Modified version of styling_input_card_generic().
    No specific LCOE input inside function, but uses style_inpRow_generic() to create
    input rows
    :param specific_row_input:      list with id dicts for each row
    :return:
    """
    rws = []
    for specific_row in specific_row_input:
        rws.append(style_inpRow_simpleapp(**specific_row))
    return rws


def style_inpCard_simpleapp(header: str,
                            specific_row_input: list) -> dbc.Card:
    """
    Description:
        Modified version of styling_input_card_generic().
        Creates dbc.card with header and multiple input rows generate by style_inpRows_simpleapp()
    Input:
        header:     card title
        rws:        list of input rows
    """
    # Specific SimpleApp arguments
    # -----------------------------------------------------------------------
    # Define first row
    row = [dbc.Row([dbc.Col(width=12, xl=6),
                    dbc.Col(dbc.Label("Nominal Input"), width=12, xl=6),
                    ])]

    rws = style_inpRows_simpleapp(specific_row_input=specific_row_input)
    row.extend(rws)

    # Create Card
    card = dbc.Card([
        dbc.CardHeader(header),
        dbc.CardBody(
            row
        )])
    return card
