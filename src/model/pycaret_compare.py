from pycaret import regression as reg
from pycaret import classification as cla
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

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


def reg_create_graph(df):
    columns = df.columns.values
    metrics = columns[1:len(columns)-1]

    fig = plt.figure(figsize=(40,20))
    fig.tight_layout()

    models = df['Model'].tolist()
    
    i = 0
    while i != len(metrics):
        axis = fig.add_subplot(4, 2, i+1)
        
        values = df[metrics[i]].tolist()
        values_sort, models_sort = zip(*sorted(zip(values, models)))

        values_sort = list(values_sort)
        models_sort = list(models_sort)
        models_sort.reverse()
        values_sort.reverse()

        axis.barh(models_sort, values_sort)
        axis.set_title(metrics[i])

        i+=1

    fig.savefig('./static/images/graph.png', bbox_inches='tight')


def cla_create_graph(df):
    columns = df.columns.values
    metrics = columns[1:len(columns)-1]

    fig = plt.figure(figsize=(40,20))
    fig.tight_layout()

    models = df['Model'].tolist()
    
    i = 0
    while i != len(metrics):
        axis = fig.add_subplot(4, 2, i+1)
        
        values = df[metrics[i]].tolist()
        values_sort, models_sort = zip(*sorted(zip(values, models)))

        values_sort = list(values_sort)
        models_sort = list(models_sort)

        axis.barh(models_sort, values_sort)
        axis.set_title(metrics[i])

        i+=1

    fig.savefig('./static/images/graph.png', bbox_inches='tight')
