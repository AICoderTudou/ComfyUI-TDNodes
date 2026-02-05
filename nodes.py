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
