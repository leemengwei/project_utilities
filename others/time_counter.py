import datetime

def calc_time(func):
    def wrapper(*args, **kw):  
        start_time = datetime.datetime.now() 
        func(*args, **kw) 
        end_time = datetime.datetime.now()
        ss = (end_time - start_time).total_seconds()
        print('<{}> takes {}s.'.format(func.__name__, ss))
    return wrapper

