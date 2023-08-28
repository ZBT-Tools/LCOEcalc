import pickle
import jsonpickle


def store_data(data):
    """
    https://github.com/jsonpickle/jsonpickle, as json.dumps can only handle
    simple variables, no objects, DataFrames..
    Info: Eigentlich sollte jsonpickle reichen, um dict mit Klassenobjekten,
    in denen DataFrames sind, zu speichern, es gibt jedoch Fehlermeldungen.
    Daher wird Datenstruktur vorher in pickle (Binärformat)
    gespeichert und dieser anschließend in json konvertiert.
    (Konvertierung in json ist notwendig für lokalen dcc storage)
    """
    data = pickle.dumps(data)
    data = jsonpickle.dumps(data)

    return data


def read_data(data):
    # Read NH3 data from storage
    data = jsonpickle.loads(data)
    data = pickle.loads(data)

    return data