# This is code is a slightly modified version of 
# http://code.activestate.com/recipes/116539/
# which author is Christoph Dietze
# In its original form it was distributed under the PSF license
# and the same license applies to this file.

from xml.dom.minidom import parseString


class NotTextNodeError:
    pass


def getTextFromNode(node):
    """
    scans through all children of node and gathers the
    text. if node has non-text child-nodes, then
    NotTextNodeError is raised.
    """
    t = ""
    for n in node.childNodes:
        if n.nodeType == n.TEXT_NODE:
            t += n.nodeValue
        else:
            raise NotTextNodeError
    try:
        t = int(t)
    except:
        try:
            t = float(t)
        except:
            pass         
    return t


def nodeToDic(node):
    """
    nodeToDic() scans through the children of node and makes a
    dictionary from the content.
    three cases are differentiated:
    - if the node contains no other nodes, it is a text-node
    and {nodeName:text} is merged into the dictionary.
    - if the node has the attribute "method" set to "true",
    then it's children will be appended to a list and this
    list is merged to the dictionary in the form: {nodeName:list}.
    - else, nodeToDic() will call itself recursively on
    the nodes children (merging {nodeName:nodeToDic()} to
    the dictionary).
    """
    dic = {} 
    for n in node.childNodes:
        if n.nodeType != n.ELEMENT_NODE:
            continue
        if n.getAttribute("multiple") == "true":
            # node with multiple children:
            # put them in a list
            l = []
            for c in n.childNodes:
                if c.nodeType != n.ELEMENT_NODE:
                    continue
                l.append(nodeToDic(c))
                dic.update({n.nodeName:l})
            continue
        
        try:
            text = getTextFromNode(n)
        except NotTextNodeError:
            # 'normal' node
            dic.update({n.nodeName:nodeToDic(n)})
            continue

        # text node
        dic.update({n.nodeName:text})
        continue
    return dic


def readConfig(filename):
    dom = parseString(filename)
    return nodeToDic(dom)

