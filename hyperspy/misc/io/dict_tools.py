def return_item(dictionary, path,  sep='.'):
    """return the node/item from the (nested/tree-like) dictionary
    at given path.
    -------------
    args:
    dictionary -- dictionary to parse
    path -- string of path with separator
            e.g. 'Signal.Detector.Type'
    sep -- separator (default: '.')
    """
    current_level = dictionary
    path_list = path.split(sep)
    for i in path_list:
        if type(current_level) == dict and i in current_level:         
            current_level = current_level[i]
        else:
            return
    return current_level


def _create_dict_path(path_list, dictionary, force_overwrite=False):
    """(create if missing and) return the branch of hierachical tree-like
    dictionary
    ----------
    args:
    path_list -- the list with nodes (e.g. ['Signal','Type','Whatever']).
    dictionary -- dictionary where branch will be created.
    force_overwrite -- should the already preocupied branch with value
      be overwriten with dictionary branch (bool; default: False).
    """
    current_level = dictionary
    for i in path_list:
        if i not in current_level:
            current_level[i] = {}
        elif type(current_level[i]) != dict:
            if force_overwrite:
                current_level[i] = {}
            else:
                raise TypeError("""
Can't intiate the dict node '{0}' as the key {1} is already preocupied
with {2} type value""".format('.'.join(path_list[:path_list.index(i)+1]),
                              i, type(current_level[i])))
        current_level = current_level[i]
    return current_level


def set_item(dictionary, path, value, sep='.', force_overwrite=False):
    current_level = dictionary
    if (type(path) == str) and (sep is not None): 
        sub_paths = path.split(sep)
    else:
        sub_paths = path
    node = _create_dict_path(sub_paths[:-1], current_level, force_overwrite)
    if (sub_paths[-1] not in node) or (type(node[sub_paths[-1]]) != dict):
        node[sub_paths[-1]] = value
    else:
        raise RuntimeError("""
Can't set the value at node '{0}' as the key '{1}' is already preocupied
with dictionary""".format(path, sub_paths[-1]))


def map_dict_to_dict(dict_to, dict_from, mapping_dict):
    for path_from, (path_to, function) in mapping_dict.items():
        value = return_node(dict_from, path_from)
        if value:
            if function is not None:
                value = function(value)
            if value is not None:
                set_item(dict_to, path_to, value) 
