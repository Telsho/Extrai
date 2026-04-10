Contributing
============

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with a clear message.
4. Push your changes to your fork.
5. Create a pull request to the main repository.

Please make sure to update the documentation and add tests for any new features.

Coding Standards
----------------

This project uses `ruff` for linting and code formatting. Please make sure to run `ruff` before committing your changes:

.. code-block:: bash

   ruff check . --fix
   ruff format .

We use `pytest` for testing. Please make sure all tests pass before submitting a pull request:

.. code-block:: bash

   pytest

A SonorQube analysis and a CVE analysis (Meterian) are always welcome.

Documentation Contributions
---------------------------

Improving the documentation is just as important as improving the code. We welcome any contributions to make the documentation clearer, more comprehensive, and more helpful for new users.

If you'd like to contribute to the docs, you can:

-   Fix typos or grammatical errors.
-   Clarify confusing explanations.
-   Add new examples or tutorials.
-   Suggest a better structure or organization.

The documentation is built using Sphinx. To build the docs locally, first make sure you have installed the development dependencies:

.. code-block:: bash

   pip install -e .[dev]

Then, from the root of the project, run:

.. code-block:: bash

   sphinx-build -b html docs docs/_build

You can then view the generated documentation by opening `docs/_build/html/index.html` in your browser.
