"""
Commonly used utils
"""

import inspect


def check_if_func_accepts_arg(func, arg):
    for param in inspect.signature(func).parameters:
        if param == arg:
            return True

    return False


def seconds_to_hours(seconds):
    return seconds / 3600


def float_to_latex(f):
    float_str = '{0:.2e}'.format(f)
    base, exponent = float_str.split('e')

    if 2 >= int(exponent) >= -2:
        return '{0:.3g}'.format(f)
    return r'{0} \times 10^{{{1}}}'.format(base, int(exponent))
