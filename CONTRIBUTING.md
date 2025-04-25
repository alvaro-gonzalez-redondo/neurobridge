# Contributing to NeuroBridge

Thank you for your interest in contributing to NeuroBridge! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

We expect all contributors to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating.

## Getting Started

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally
   ```bash
   git clone https://github.com/alvaro-gonzalez-redondo/neurobridge
   cd neurobridge
   ```
3. Create a virtual environment and install development dependencies
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

### Development Workflow

1. Create a new branch for your feature or bugfix
   ```bash
   git checkout -b feature/your-feature-name
   ```
   
2. Make your changes, following the coding style guidelines

3. Write tests for your changes

4. Run the tests and ensure they pass
   ```bash
   pytest
   ```

5. Format your code using Black
   ```bash
   black neurobridge/
   ```

6. Submit a pull request

## Pull Request Process

1. Ensure your code follows the project's style guidelines
2. Update the documentation, including docstrings and README if necessary
3. Add or update tests for your changes
4. Make sure all tests pass before submitting the PR
5. Update the CHANGELOG.md with a brief description of your changes
6. The PR should target the `develop` branch, not `main`
7. Request a review from one of the core maintainers

## Coding Style Guidelines

NeuroBridge follows these coding conventions:

- Use [PEP 8](https://peps.python.org/pep-0008/) for Python code style
- Use [NumPy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html) for documentation
- Format code with [Black](https://black.readthedocs.io/en/stable/) with default settings
- Use type hints where appropriate

## Documentation

All new features should include:

- Docstrings for all public classes, methods, and functions
- Code examples where appropriate
- Updates to the main documentation if necessary

## Testing

We use pytest for testing. All new features should include tests:

- Unit tests for individual components
- Integration tests for component interactions
- Test with both single-GPU and multi-GPU configurations if applicable

## Reporting Bugs

When reporting a bug, please include:

- A clear and descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Environment information (OS, Python version, GPU/CUDA details)
- Any relevant logs or error messages

## Feature Requests

Feature requests are welcome. Please provide:

- A clear and descriptive title
- A detailed description of the proposed feature
- An explanation of why this feature would be useful to NeuroBridge users
- Example code or pseudocode if applicable

## License

By contributing to NeuroBridge, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

## Contact

If you have questions about contributing, please open an issue or contact the maintainers.

Thank you for contributing to NeuroBridge!