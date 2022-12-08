from dataclasses import dataclass


@dataclass
class DataclassMultiplicationInput:
    """..."""
    a: float
    b: float


def multiplication(inp: DataclassMultiplicationInput):
    """
    Input:
        DataclassMultiplicationInput with attributes a and b

    Description:
        multiplies a * b

    Output:
        c: Product of a and b

    """
    c = inp.a * inp.b

    return c
