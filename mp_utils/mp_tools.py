import ntpath


def get_counter_code(counter, level=3):
    zeros = "0" * (level - len(str(counter)))
    code = f"{zeros}{counter}"
    return code


def path_leaf(path: str):
    """
    The function returns folder name or file name without extension
    :param path: str // full path
    :return: str // name
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
