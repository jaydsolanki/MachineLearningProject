from io import BytesIO
import pickle
import base64
from .models import *


def object_to_pickle_bytes(obj):
    sio = BytesIO()
    pickle._dump(obj, sio)
    return sio.getvalue()


def pickle_bytes_to_base64_string(picked_obj):
    return base64.b64encode(picked_obj.decode('ascii'))


def save_pickle_obj_string_to_db(id, pickled_obj_base64_string):
    ml_model = MlModels(id=id, model=pickled_obj_base64_string)
    ml_model.save()


def dump_to_db(id, pow):
    byte_val = object_to_pickle_bytes(pow)
    b64_obj = pickle_bytes_to_base64_string(byte_val)
    save_pickle_obj_string_to_db(id, b64_obj)


def retrieve_from_db(model_id):
    model = MlModels.objects.filter(id=model_id).first()
    if model:
        model = base64.b64decode(model.model)
        sio = BytesIO(model)
        model = pickle.load(sio)
    return model


class PickleObjectWrapper:
    def __init__(self):
        self.param1 = None
        self.param2 = None
        self.param3 = None
        self.param4 = None
        self.param5 = None
        self.param6 = None

