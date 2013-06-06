import os

def dump_dictionary(file, dic, string='root', node_separator='.',
                    value_separator=' = '):
    for key in dic.keys():
        if isinstance(dic[key], dict):
            dump_dictionary(file, dic[key], string + node_separator + key)
        else:
            file.write(string + node_separator + key + value_separator +
            str(dic[key]) + '\n')
            
def append2pathname(filename, to_append):
    """Append a string to a path name
    
    Parameters
    ----------
    filename : str
    to_append : str
    
    """
    pathname, extension = os.path.splitext(filename)
    return pathname + to_append + extension
    
def incremental_filename(filename, i=1):
    """If a file with the same file name exists, returns a new filename that
    does not exists.
    
    The new file name is created by appending `-n` (where `n` is an integer)
    to path name
    
    Parameters
    ----------
    filename : str
    i : int
       The number to be appended.
    """
    
    if os.path.isfile(filename):
        new_filename = append2pathname(filename, '-%s' % i) 
        if os.path.isfile(new_filename):
            return incremental_filename(filename, i + 1)
        else:
            return new_filename
    else:
        return filename
        
def ensure_directory(path):
    """Check if the path exists and if it does not create the directory"""
    directory = os.path.split(path)[0]
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
