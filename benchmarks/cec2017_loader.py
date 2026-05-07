import opfunu

def get_cec2017_function(func_id, dimension):
    """
    Returns a specific CEC2017 function instance.
    func_id: int (1 to 30)
    dimension: int (10, 30, 50)
    """
    # Filter classes to find the one matching the ID
    cec2017_classes = opfunu.get_functions_based_classname("2017")
    
    # CEC2017 F2 is usually omitted in the suite
    # Find the class that ends with 'F{func_id}2017'
    for func_class in cec2017_classes:
        if func_class.__name__ == f"F{func_id}2017":
            return func_class(ndim=dimension)
    
    return None