def add_dict_defaults(main_dict, defaults_dict):
    for key, default_val in defaults_dict.items():
        if key not in main_dict: main_dict[key] = default_val
    return True
