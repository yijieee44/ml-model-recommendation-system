from pycaret import regression as reg
from pycaret import classification as cla
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def reg_compare(df, target):
    reg.setup(data = df, target = target, session_id=123, silent=True, verbose=False)
    best_model = reg.compare_models(verbose=False)
    result_df = reg.pull()

    #get the best r2 to know whether it is suitable or not
    best_r2 = max(result_df['R2'])
    suitable = data_veri('reg', best_r2)
    return best_model, result_df, suitable


def cla_compare(df, target):
    cla.setup(data = df, target = target, session_id=123, silent=True, verbose=False)
    best_model = cla.compare_models(verbose=False)
    result_df = cla.pull()
    #get the best f1 to know whether it is suitable or not
    best_f1 = max(result_df['F1'])
    suitable = data_veri('cla', best_f1)
    return best_model, result_df, suitable

def data_veri(ml_type, val):
    '''
    Determine whether it is a good dataset to predict the target 
    'reg' or 'cla' - Regression or Classification problem
    '''

    T1_reg = 0.4 #R2
    T2_cla = 0.4 #F1_score

    if ml_type == 'reg':
        if val < T1_reg:
            return False
        else:
            return True
    else:
        if val < T2_cla:
            return False
        else:
            return True



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



