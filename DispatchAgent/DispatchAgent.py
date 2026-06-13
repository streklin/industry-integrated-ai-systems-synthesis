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
            "END": "999",
            "PRIORITIZE": "1000",
            "DEPRIORITIZE": "1001",
            "SEARCH/RESCUE": "1002",
            "DEPLOY": "1003",
            "RECALL": "1004",
            "STATUS": "1005",
            "HUMAN": "1006",
            "NOHUMAN": "1007",
            "FIRE": "1008",
            "NOFIRE": "1009",
            "TRANSMIT": "1010",
            "UNK": "1011"
        } 

        self.reverse_token_map = {v: k for k, v in self.token_map.items()}

    def detokenize(self, tokens: list[int]) -> list[str]:
        """
        Detokenizes a sequence of tokens.

        Args:
            tokens: A sequence of tokens to detokenize.

        Returns:
            A sequence of strings.
        """
        return [self.reverse_token_map[token] for token in tokens]
        
    def tokenize(self, sequence: list[str]) -> list[int]:
        """
        Tokenizes a sequence of strings.

        Args:
            sequence: A sequence of strings to tokenize.

        Returns:
            A sequence of tokens.
        """
        for token in sequence:
            if token in self.token_map:
                yield int(self.token_map[token])
            else:
                yield int(self.token_map["UNK"])


class DispatchActionMessageValidator:
    def __init__(self):
        self.valid_commands = {
            "DEPLOY", 
            "RECALL", 
            "STATUS"
        }


    def validate_deploy_message(self, message: list[str]) -> bool:
        """
        Validates of a deploy command is well formed.
        """
        # message must have 5 tokens 
        if len(message) != 5:
            return False
        
        # 2nd token is the uav id
        try:
            uav_id = int(message[1])
        except ValueError:
            return False
        
        # 3rd token is the x coordinate
        try:
            x = int(message[2])
        except ValueError:
            return False
        
        # 4th token is the y coordinate
        try:
            y = int(message[3])
        except ValueError:
            return False
        
        return True

    def validate_recall_message(self, message: list[str]) -> bool:
        """
        Validates of a recall command is well formed.
        """
        # must have three tokens
        if len(message) != 3:
            return False
        
        # second token is the uav id
        try:
            uav_id = int(message[1])
        except ValueError:
            return False
        
        return True

    def validate_status_message(self, message: list[str]) -> bool:
        """
        Validates of a status command is well formed.
        """
        # must have two tokens
        if len(message) != 2:
            return False
        
        # second token is the uav id
        try:
            uav_id = int(message[1])
        except ValueError:
            return False
        
        return True

    def is_valid(self, message: list[str]) -> bool:
        """
        Validates if a message is well-formed.
        """
        # must start with a command
        if message[0] not in self.valid_commands:
            return False

        if message[0] == "DEPLOY":
            return self.validate_deploy_message(message)
        elif message[0] == "RECALL":
            return self.validate_recall_message(message)
        elif message[0] == "STATUS":
            return self.validate_status_message(message)
        
        return False



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
        self.tokenizer = Tokenizer()
        self.message_buffer = [] # holds all unprocessed messages from UAVs and command center
        self.model = DecisionTransformer()
        self.max_response_length = 50

    def add_message_to_buffer(self, message: list[str]):
        """
        Adds a message to the buffer.
        """
        self.message_buffer.append(message)

    def update(self) -> list[str]:
        """
        Appends all messages together to create a single "context", then 
        generates the next radio signal to send out. Message is "transmitted"
        when then "TRANSMIT" token is generated
        """
        # combine the lists into a single context window
        context = []
        for message in self.message_buffer:
            context.extend(message)
        # tokenize 
        context_tokens = list(self.tokenizer.tokenize(context))

        # pad with UNK tokens if needed
        if len(context_tokens) < self.max_response_length:
            context_tokens.extend([self.tokenizer.token_map["UNK"]] * (self.max_response_length - len(context_tokens)))
        
        
        
        return None

    def load_model(self, model_file: str):
        """
        Loads the model from the given file.

        Args:
            model_file: The file to load the model from.
        """
        pass