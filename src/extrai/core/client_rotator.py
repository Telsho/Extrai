from .base_llm_client import BaseLLMClient


class ClientRotator:
    """
    Manages rotation through a list of LLM clients.
    """

    def __init__(self, clients: BaseLLMClient | list[BaseLLMClient]):
        self.clients = clients if isinstance(clients, list) else [clients]
        if not self.clients:
            raise ValueError("At least one client must be provided")
        self._current_index = 0

    def get_next_client(self) -> BaseLLMClient:
        """Returns the next client in the rotation."""
        client = self.clients[self._current_index]
        self._current_index = (self._current_index + 1) % len(self.clients)
        return client

    @property
    def current_client(self) -> BaseLLMClient:
        """Returns the current client without advancing rotation."""
        return self.clients[self._current_index]

    def reset(self):
        """Resets the rotation to the start."""
        self._current_index = 0
