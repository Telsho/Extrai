.. _llm_providers:

LLM Providers
=============

This library is designed to be LLM-agnostic, allowing you to connect to a variety of LLM providers through a standardized client interface. All clients inherit from a common `BaseLLMClient`, ensuring consistent behavior across different backends.

Supported Providers
-------------------

Out of the box, we provide support for the following LLM providers:

- **OpenAI**: For models like GPT-4, GPT-3.5, etc.
- **DeepSeek**: For DeepSeek-specific models.
- **Gemini**: For Google's Gemini family of models.
- **Hugging Face**: For accessing thousands of models from the Hugging Face Hub via the Inference API.
- **Ollama**: For running local models with Ollama.

Each client is configured with sensible defaults but can be customized with your own API keys, model names, and base URLs.

Contributing a New Provider
---------------------------

We welcome contributions! If you'd like to add support for a new LLM provider, the process is straightforward:

1.  **Create a New Client Class**: Create a new Python file in `src/extrai/llm_providers/` for your client (e.g., `my_provider_client.py`).
2.  **Inherit from BaseLLMClient**: Your new class should inherit from `BaseLLMClient`.
3.  **Implement** `_execute_llm_call`: The main requirement is to implement the `async def _execute_llm_call(self, system_prompt: str, user_prompt: str) -> str` method. This is where you'll make the actual API call to your provider.
4.  **Add to** `__init__.py`: Add your new client to the `__all__` list in `src/extrai/llm_providers/__init__.py` to make it easily importable.
5.  **Submit a Pull Request**: Please see our :doc:`contributing` guide for details on coding standards and submitting your changes.

By following this pattern, your new client will automatically integrate with the rest of the library's workflow, including the consensus mechanism and analytics.

.. seealso::

   :ref:`how_to_using_multiple_llm_providers`
      A practical walkthrough of using multiple LLM providers.

   For a complete, runnable script, see the example file: `examples/rotating_llm_providers.py`.
