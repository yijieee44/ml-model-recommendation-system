from pycaret import regression as reg
from pycaret import classification as cla


def reg_compare(df, target):
    reg.setup(data = df, target = target, session_id=123, silent=True, verbose=False)
    best_model = reg.compare_models(verbose=False)
    result_df = reg.pull()
    return best_model, result_df


def cla_compare(df, target):
    cla.setup(data = df, target = target, session_id=123, silent=True, verbose=False)
    best_model = cla.compare_models(verbose=False)
    result_df = cla.pull()
    return best_model, result_df
