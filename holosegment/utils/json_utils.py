import json

def ordered(obj):
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj

def remove_spaces_from_keys(obj):
    if isinstance(obj, dict):
        return {
            key.replace(" ", ""): remove_spaces_from_keys(value)
            for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [remove_spaces_from_keys(item) for item in obj]
    else:
        return obj