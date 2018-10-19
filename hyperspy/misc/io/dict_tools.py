from collections import defaultdict
from ast import literal_eval
import re

# MSXML bug-workarounds:
# Define re with two capturing groups with comma in between.
# A first group looks for numeric value after <tag> (the '>' char) with or
# without minus sign, second group looks for numeric value with following
# closing <\tag> (the '<' char); '([Ee]-?\d*)' part (optionally a third group)
# checks for scientific notation (e.g. 8,843E-7 -> 'E-7');
# compiled pattern is binary, as raw xml string is binary.:
msxml_faulty_dec_patterns = re.compile(b'(>-?\\d+),(\\d*([Ee]-?\\d*)?<)')


def fix_msxml(xml_string):
    """return fixed xml string from xml produced with xml-standard
    by-design incomplient MSXML implementation."""
    # comma -> dot in decimals:
    fixed_xml = msxml_faulty_dec_patterns.sub(b'\\1.\\2', xml_string)
    
    return fixed_xml


def interpret(string):
    """interpret any string and return casted to appropriate
    dtype python object
    """
    try:
        return literal_eval(string)
    except (ValueError, SyntaxError):
        # SyntaxError due to:
        # literal_eval have problems with strings like this '8842_80'
        return string


def dictionarize(et):
    """translate and return the python dictionary/list tree
    from xml.etree.ElementTree.ElementTree instance"""
    d = {et.tag: {} if et.attrib else None}
    children = et.getchildren()
    if children:
        dd = defaultdict(list)
        for dc in map(dictionarize, children):
            for key, val in dc.items():
                dd[key].append(val)
        d = {et.tag: {key: interpret(val[0]) if len(
            val) == 1 else val for key, val in dd.items()}}
    if et.attrib:
        d[et.tag].update(('@_' + key, interpret(val))
                         for key, val in et.attrib.items())
    if et.text:
        if 0 < len(et.text) <=32:
            d.update({et.tag: {'#val': interpret(et.text)} 
                      if (et.attrib or et.getchildren())
                      else interpret(et.text)})
        else:
            d.update({et.tag: {'#ET_val': et}})
    return d


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
    for i in path_list[:-1]:
        if isinstance(current_level, dict) and i in current_level:         
            current_level = current_level[i]
        else:
            return
    if path_list[-1] in current_level:
        current_level = current_level[path_list[-1]]
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


def map_dict_to_dict(dict_to, dict_from, mapping_dict, sep='.',
                     force_overwrite=False):
    for path_from, (path_to, function) in mapping_dict.items():
        value = return_item(dict_from, path_from, sep=sep)
        if value:
            if function is not None:
                value = function(value)
            if value is not None:
                set_item(dict_to, path_to, value, sep=sep,
                         force_overwrite=force_overwrite) 
