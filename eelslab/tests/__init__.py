import os
import nose

def test(args = [], no_path_adjustment = False):
    """Run tests.
        
       args : list of strings
           a list of options that will be passed to nosetests
       no_path_adjustment : bool
           If True it the --no-path-adjustment option wil be passed to nosetests
    """
    mod_loc = os.path.dirname(__file__)
    totest = os.path.join(mod_loc,'io', 'test_dm3.py')

    if no_path_adjustment is not None:
        args.append('--no-path-adjustment')
    args.insert(0,totest)
    return nose.run(argv = args)

