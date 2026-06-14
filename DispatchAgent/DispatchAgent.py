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
            "NOFIRE": 1009,
            "TRANSMIT": 1010,
            "UNK": 1011
        } 

        self.reverse_token_map = {v: k for k, v in self.token_map.items()}

    def to_dense_id(self, external_id: int) -> int:
        """
        Maps an external token ID to a dense token ID in range 0-112.
        """
        if 0 <= external_id <= 99:
            return external_id
        elif 999 <= external_id <= 1011:
            return external_id - 899
        else:
            return 112  # dense ID for UNK (1011 - 899 = 112)

    def to_external_id(self, dense_id: int) -> int:
        """
        Maps a dense token ID in range 0-112 back to its external token ID.
        """
        if 0 <= dense_id <= 99:
            return dense_id
        elif 100 <= dense_id <= 112:
            return dense_id + 899
        else:
            return 1011  # external ID for UNK

    def detokenize(self, tokens: list[int]) -> list[str]:
        """
        Detokenizes a sequence of dense token IDs back to strings.

        Args:
            tokens: A sequence of dense tokens to detokenize.

        Returns:
            A sequence of strings.
        """
        result = []
        for dense_token in tokens:
            ext_id = self.to_external_id(dense_token)
            if ext_id in self.reverse_token_map:
                result.append(self.reverse_token_map[ext_id])
            elif 0 <= ext_id <= 99:
                result.append(str(ext_id))
            else:
                result.append("UNK")
        return result
        
    def tokenize(self, sequence: list[str]) -> list[int]:
        """
        Tokenizes a sequence of strings into dense token IDs.

        Args:
            sequence: A sequence of strings to tokenize.

        Returns:
            A generator of dense token IDs.
        """
        for token in sequence:
            if token in self.token_map:
                yield self.to_dense_id(self.token_map[token])
            else:
                try:
                    val = int(token)
                    if 0 <= val <= 99:
                        yield self.to_dense_id(val)
                    else:
                        yield self.to_dense_id(1011)  # UNK
                except ValueError:
                    yield self.to_dense_id(1011)  # UNK


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
        self.message_buffer = []  # holds all unprocessed messages from UAVs and command center
        
        # Configuration matching DecisionTransformer architecture
        self.config = {
            "embed_dim": 128,
            "head_size": 32,
            "heads_num": 4,
            "use_bias": True,
            "dropout_rate": 0.1,
            "layers_num": 4,
            "vocab_size": 113,  # Dense vocabulary size (0-112)
            "context_size": 128,
        }
        self.model = DecisionTransformer(self.config)
        self.max_response_length = 50

        if model_file is not None:
            self.load_model(model_file)

    def add_message_to_buffer(self, message: list[str]):
        """
        Adds a message to the buffer.
        """
        self.message_buffer.append(message)

    def update(self) -> list[str]:
        """
        Appends all messages together to create a single "context", then 
        generates the next radio signal to send out. Message is "transmitted"
        when then "TRANSMIT" token is generated.
        """
        # combine the lists into a single context window
        context = []
        for message in self.message_buffer:
            context.extend(message)
            
        # tokenize to dense IDs
        context_tokens = list(self.tokenizer.tokenize(context))

        if len(context_tokens) == 0:
            # If context is empty, initialize with dense ID for UNK
            context_tokens = [self.tokenizer.to_dense_id(self.tokenizer.token_map["UNK"])]

        generated_tokens = []
        
        # Dense ID equivalents of TRANSMIT (111) and END (100)
        transmit_dense_id = self.tokenizer.to_dense_id(self.tokenizer.token_map["TRANSMIT"])
        end_dense_id = self.tokenizer.to_dense_id(self.tokenizer.token_map["END"])
        
        current_tokens = list(context_tokens)
        max_context_size = self.config["context_size"]
        
        self.model.eval()
        for _ in range(self.max_response_length):
            # Truncate context to model's context_size
            input_tokens = current_tokens[-max_context_size:]
            
            # Convert to PyTorch tensor (batch size 1)
            input_tensor = torch.tensor([input_tokens], dtype=torch.long)
            
            # Predict logits
            with torch.no_grad():
                logits = self.model(input_tensor)  # (1, seq_length, vocab_size)
            
            # Get logits for the last token in the sequence
            next_token_logits = logits[0, -1, :]
            
            # Argmax to get next token ID
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            
            # Append to sequence
            current_tokens.append(next_token_id)
            generated_tokens.append(next_token_id)
            
            # Stop if TRANSMIT or END is generated
            if next_token_id in (transmit_dense_id, end_dense_id):
                break
                
        # Detokenize response
        return self.tokenizer.detokenize(generated_tokens)

    def load_model(self, model_file: str):
        """
        Loads the model from the given file.

        Args:
            model_file: The file to load the model from.
        """
        state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.eval()