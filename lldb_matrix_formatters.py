import lldb


def lina_mat_summary(valobj, internal_dict):
    """Custom formatter for lina::mat<T, R, C>"""

    float_precision = 6

    # Get the type name to extract template parameters
    type_name = str(valobj.GetType())

    # Parse template parameters from type name like "lina::mat<float, 2, 3>"
    try:
        # Extract the part between < and >
        template_part = type_name.split('<')[1].split('>')[0]
        params = [p.strip() for p in template_part.split(',')]

        element_type = params[0]
        rows = int(params[1])
        cols = int(params[2])
    except:
        return "Could not parse matrix template parameters"

    # Get the array member
    array = valobj.GetChildMemberWithName('a')
    if not array.IsValid():
        return "Invalid matrix - no 'a' member found"

    # Build the formatted output
    result = "{ "

    for r in range(rows):
        result += "{ "
        for c in range(cols):
            if c > 0:
                result += ", "

            # Get array element at index [r * cols + c]
            idx = r * cols + c
            element = array.GetChildAtIndex(idx)

            if element.IsValid():
                value = str(round(float(element.GetValue()), float_precision))
                if value:
                    result += str(value)
                else:
                    result += "?"
            else:
                result += "?"

        result += " }"
        if r < rows - 1:
            result += ","

    result += " }"
    return result


def __lldb_init_module(debugger, internal_dict):
    """Called when the module is imported into LLDB"""

    # Register the full formatter
    debugger.HandleCommand(
        'type summary add -x "^lina::mat<.*>$" '
        '-F lldb_matrix_formatters.lina_mat_summary'
    )

    print("lina::mat formatters loaded!")
