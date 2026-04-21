import json
import os

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
    
def flatten_schema(schema):
    flat = {}

    def walk(node, path=""):
        for k, v in node.items():
            new_path = os.path.join(path, k) if path else k
            if isinstance(v, dict):
                walk(v, new_path)
            else:
                flat[v] = new_path

    walk(schema)
    return flat