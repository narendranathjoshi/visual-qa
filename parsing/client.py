import json
from jsonrpc import ServerProxy, JsonRpc20, TransportTcpIp
from pprint import pprint


class StanfordNLP:
    def __init__(self):
        self.server = ServerProxy(JsonRpc20(),
                                  TransportTcpIp(addr=("127.0.0.1", 8080)))

    def parse(self, text):
        return json.loads(self.server.parse(text))


nlp = StanfordNLP()
result = nlp.parse("How many engines are there?")
pprint(result)