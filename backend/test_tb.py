import traceback

try:
    import pkg_resources_does_not_exist
except Exception as e:
    tb = traceback.format_exc()
    error_str = f"{e}\n{tb[-3500:]}"
    print("REPR OF ERROR STR:")
    print(repr(error_str))
