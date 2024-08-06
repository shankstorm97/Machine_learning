import time

def measure_time(function, *args):
    start_time = time.time()
    result = function(*args)
    elapsed_time = time.time() - start_time
    return result, elapsed_time
