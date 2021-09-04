import time

def timeit(tag, start_time):
    end_time = time.time()
    print("{}: {}s".format(tag, end_time - start_time))
    return end_time