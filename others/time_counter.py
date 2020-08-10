import datetime
from IPython import embed

def calc_time(func):
    def wrapper(*args, **kw):  
        start_time = datetime.datetime.now() 
        out = func(*args, **kw) 
        end_time = datetime.datetime.now()
        ss = (end_time - start_time).total_seconds()
        if 1: #ss>0.5:
            print('<{}> takes {}s.'.format(func.__name__, ss))
        return out
    return wrapper
