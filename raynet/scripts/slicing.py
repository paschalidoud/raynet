def int_or_none(x):
    try:
        return int(x)
    except ValueError:
        return None


def frame_idxs_type(arg):
    """Create frame idxs type from a string passed as argument.
    
    # Returns
        slice or list of idxs
    """
    if ":" in arg:
        return slice(*list(map(int_or_none, arg.split(":"))))
    if "," in arg:
        return list(map(int, arg.split(",")))
    return [int(arg)]
