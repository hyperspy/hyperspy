
def pool_hs(parallel, pool_type=None, ipython_timeout=1.):
    """
    Create a pool for multiprocessing

    Parameters
    ----------
    pool_type: 'ipython' or'mp'
        the type of pool
    """
    if pool_type is None:
        from IPython.parallel import Client, error
        try:
            c = Client(profile='hyperspy', timeout=ipython_timeout)
            _pool = c[:parallel]
            pool_type = 'ipython'
        except (error.TimeoutError, IOError):
            from multiprocessing import Pool
            pool_type = 'mp'
            _pool = Pool(processes=parallel)
    elif pool_type == 'iypthon':
        from IPython.parallel import Client
        c = Client(profile='hyperspy', timeout=ipython_timeout)
        _pool = c[:parallel]
        pool_type = 'ipython'
    else:
        from multiprocessing import Pool
        pool_type = 'mp'
        _pool = Pool(processes=parallel)
    return _pool, pool_type


def multifit(args):
    from hyperspy.model import Model
    model_dict, kwargs = args
    m = Model(model_dict)
    m.multifit(**kwargs)
    d = m.as_dictionary()
    del d['spectrum']
    # delete everything else that doesn't matter. Only maps of
    # parameters and chisq matter
    return d
