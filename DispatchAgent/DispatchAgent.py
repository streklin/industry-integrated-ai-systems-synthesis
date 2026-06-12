import numpy as np
import matplotlib.pyplot as plt

import torch

from DecisionTransformer import DecisionTransformer


class Tokenizer:
    """
    A class that tokenizes the state space of the UAV Dispatcher Command System
    """
    def __init__(self):
        # Tokens 0-99 are reserved for the xpos and ypos of the UAVs
        # UAVs will report their position as projected on the CA, assuming a 
        # 100x100 map.
        # Token 999 is served to mark the end of a command / radio sequence.
        self.token_map = {
            "END": 999,
            "PRIORITIZE": 1000,
            "DEPRIORITIZE": 1001,
            "SEARCH/RESCUE": 1002,
            "DEPLOY": 1003,
            "RECALL": 1004,
            "STATUS": 1005,
            "HUMAN": 1006,
            "NOHUMAN": 1007,
            "FIRE": 1008,
            "NOFIRE": 1009
        }     

    def detokenize(self, tokens: list[str]) -> list[str]:
        pass

    def tokenize(self, sequence: list[str]) -> list[str]:
        pass

class DispatchAgent:
    """
    A class that dispatches UAVs to respond to incidents.

    The Dispatcher sits between the command center the UAVs.
    It is responsible for:
        - processing radio messages from the field,
        - and producing radio messages for the UAVs,
        - managing the available UAVs
    
    DispatchAgent will use a DecisionTransformer to learn how to generate radio commands
    to the UAVs that achieve the goals provided by the Command Center System.
    """
    def __init__(self, model_file=None):
        pass