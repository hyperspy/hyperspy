def get_multi_processing_pool(parallel,self=None):
    
    from IPython.parallel import Client, error
    ipython_timeout = 1.
    try:
        c = Client(profile='hyperspy', timeout=ipython_timeout)
        pool = c[:]
        pool_type = 'iypthon'
    except (error.TimeoutError, IOError):
        print "Problem with multiprocessing"
        from multiprocessing import Pool
        pool_type = 'mp'
        pool = Pool(processes=parallel)

    return pool, pool_type
