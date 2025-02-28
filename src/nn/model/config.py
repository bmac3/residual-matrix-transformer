from abc import ABC

import msgspec


class AbstractConfig(ABC):

    def encode(self):
        return msgspec.json.encode(self)

    @classmethod
    def decode(cls, data):
        return msgspec.json.decode(data, type=cls)
