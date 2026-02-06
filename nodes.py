import json
from .video_save import TD_VideoCombine

class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False

class Any2ListNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (AlwaysEqualProxy("*"),),
            },
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    OUTPUT_IS_LIST = (True, )
    FUNCTION = "to"
    CATEGORY = "TDNodes"

    def to(self, any):
        return (list(any),)

def get_nested_value(data, dotted_key, default=None):
    keys = dotted_key.split('.')
    for key in keys:
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except:
                pass
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data

class ParseJsonNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "key": ("STRING", {"default": ""}),
            },
        }
    
    RETURN_TYPES = (AlwaysEqualProxy("*"), "STRING", "INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("any", "string", "int", "float", "boolean")
    FUNCTION = "process"
    CATEGORY = "TDNodes"
    
    def process(self, input, key=None, default=None):
        if isinstance(input, str):
            input = [input]
        elif isinstance(input, dict):
            input = [input]
            
        result = {
            "any": [None] * len(input),
            "string": [None] * len(input),
            "int": [None] * len(input),
            "float": [None] * len(input),
            "boolean": [None] * len(input)
        }
        for i, item in enumerate(input):
            val = default
            data = item
            
            if isinstance(item, str):
                # Clean up potential markdown code blocks
                clean_json = item.strip()
                if clean_json.startswith("```json"):
                    clean_json = clean_json.removeprefix("```json").removesuffix("```")
                elif clean_json.startswith("```"):
                    clean_json = clean_json.removeprefix("```").removesuffix("```")
                data = clean_json
            
            if key is not None and key != "":
                val = get_nested_value(data, key, default)
            else:
                # If key is empty, try to parse the whole JSON if it's a string
                if isinstance(data, str):
                    try:
                        val = json.loads(data)
                    except:
                        val = data
                else:
                    val = data

            result["any"][i] = val
            try:
                result["string"][i] = str(val)
            except Exception:
                result["string"][i] = str(val)
            
            try:
                result["int"][i] = int(val)
            except Exception:
                # If conversion fails, keep the original value if it makes sense or default?
                # Reference sets it to val, which might not be an int.
                # But RETURN_TYPES says INT. ComfyUI might complain if it's not an int.
                # I'll try to convert to float then int (for "1.0"), or 0.
                try:
                    result["int"][i] = int(float(val))
                except:
                    result["int"][i] = 0
            
            try:
                result["float"][i] = float(val)
            except Exception:
                result["float"][i] = 0.0
            
            try:
                if isinstance(val, bool):
                    result["boolean"][i] = val
                elif isinstance(val, str):
                    result["boolean"][i] = val.lower() == "true"
                else:
                    result["boolean"][i] = bool(val)
            except Exception:
                result["boolean"][i] = False
                
        if len(result["any"]) == 1:
            return (result["any"][0], result["string"][0], result["int"][0], result["float"][0], result["boolean"][0])
        
        return (result["any"], result["string"], result["int"], result["float"], result["boolean"])
