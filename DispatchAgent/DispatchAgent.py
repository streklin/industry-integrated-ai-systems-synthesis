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
            "max_ep_length": 100,
            "max_state_len": 20,
            "max_action_len": 10,
        }
        self.model = DecisionTransformer(self.config)

        # History lists to keep track of trajectory for DecisionTransformer context
        self.history_states = []       # list of list of dense token IDs (each of length L_s)
        self.history_actions = []      # list of list of dense token IDs (each of length L_a)
        self.history_returns = []      # list of list of float (each of length 1)
        self.history_timesteps = []    # list of ints
        
        # Track current return-to-go
        self.target_return = 1.0
        self.current_return = self.target_return

        if model_file is not None:
            self.load_model(model_file)

    def add_message_to_buffer(self, message: list[str]):
        """
        Adds a message to the buffer.
        """
        self.message_buffer.append(message)

    def _pad_or_truncate(self, tokens: list[int], max_len: int) -> list[int]:
        if len(tokens) < max_len:
            return tokens + [112] * (max_len - len(tokens))  # 112 is dense ID for UNK
        else:
            return tokens[:max_len]

    def update(self) -> list[str]:
        """
        Appends all messages together to create a single "state", then 
        generates the next action sequence (radio command) using the DecisionTransformer.
        """
        # 1. Prepare current state s_t
        context = []
        for message in self.message_buffer:
            context.extend(message)
        # Clear buffer
        self.message_buffer = []
        
        # Tokenize current state to dense IDs and pad/truncate
        state_tokens = list(self.tokenizer.tokenize(context))
        padded_state = self._pad_or_truncate(state_tokens, self.config["max_state_len"])
        
        # 2. Determine current timestep t
        t = len(self.history_states)
        
        # 3. Append current state, return, and timestep to history
        self.history_states.append(padded_state)
        self.history_returns.append([self.current_return])
        self.history_timesteps.append(t)
        
        # 4. Prepare placeholder action for the current step (padded with UNK/112)
        current_action = [112] * self.config["max_action_len"]
        
        # We append current_action to history_actions and update it in-place during autoregressive generation.
        self.history_actions.append(current_action)
        
        # 5. Slice last K steps from history (context window size)
        K = 20
        states_seq = self.history_states[-K:]
        actions_seq = self.history_actions[-K:]
        returns_seq = self.history_returns[-K:]
        timesteps_seq = self.history_timesteps[-K:]
        
        # Convert to tensors with batch_size = 1
        states_tensor = torch.tensor([states_seq], dtype=torch.long)
        actions_tensor = torch.tensor([actions_seq], dtype=torch.long)
        returns_tensor = torch.tensor([returns_seq], dtype=torch.float)
        timesteps_tensor = torch.tensor([timesteps_seq], dtype=torch.long)
        
        # Get dense IDs for control tokens
        transmit_dense_id = self.tokenizer.to_dense_id(self.tokenizer.token_map["TRANSMIT"])
        end_dense_id = self.tokenizer.to_dense_id(self.tokenizer.token_map["END"])
        
        self.model.eval()
        generated_action_tokens = []
        
        for j in range(self.config["max_action_len"]):
            with torch.no_grad():
                # Forward pass: shape of action_preds is (1, seq_len, L_a, vocab_size)
                action_preds, reward_preds = self.model(
                    states_tensor, 
                    actions_tensor, 
                    returns_tensor, 
                    timesteps_tensor
                )
            
            # Extract logits for the j-th action token of the last timestep in sequence
            logits = action_preds[0, -1, j, :]
            
            # Predict the next token ID
            next_token_id = torch.argmax(logits, dim=-1).item()
            
            # Update the action tensors
            current_action[j] = next_token_id
            actions_tensor[0, -1, j] = next_token_id
            
            generated_action_tokens.append(next_token_id)
            
            # Stop early if TRANSMIT or END is generated
            if next_token_id in (transmit_dense_id, end_dense_id):
                break
        
        # Update current action remaining spots with UNK (112) in history
        for idx in range(len(generated_action_tokens), self.config["max_action_len"]):
            current_action[idx] = 112
            
        # 6. Update return-to-go for the next step using reward function
        step_reward = self.get_reward(padded_state, current_action)
        self.current_return -= step_reward
        
        # 7. Detokenize response
        return self.tokenizer.detokenize(generated_action_tokens)

    def get_reward(self, state: list[int], action: list[int]) -> float:
        """
        Placeholder reward function. Can be overridden with actual reward logic.
        """
        return 0.0

    def load_model(self, model_file: str):
        """
        Loads the model from the given file.

        Args:
            model_file: The file to load the model from.
        """
        state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.eval()