 # Contributing to Brain4J
 
Thank you for your interest in contributing to Brain4J! We welcome contributions of all kinds—from fixing bugs and improving documentation to adding new features and writing tests. By participating, you agree to follow our guidelines to maintain a positive and collaborative environment.
 
## Table of Contents
 
- [How to Contribute](#how-to-contribute)
- [Code of Conduct](#code-of-conduct)
- [Branching Model](#branching-model)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Submitting Pull Requests](#submitting-pull-requests)
- [Reporting Issues](#reporting-issues)
- [License](#license)
- [Acknowledgements](#acknowledgements)
 
## How to Contribute
 
1. **Fork the repository:** Create your own copy of the project.
2. **Create a branch:** Start a new branch from the main branch for your work.
3. **Make changes:** Implement your feature or fix. Please ensure your code includes tests and necessary documentation.
4. **Follow commit guidelines:** Use the commit message format described below.
5. **Submit a Pull Request:** Clearly explain your changes and reference any related issues.
 
## Code of Conduct
 
We are committed to providing a friendly, safe, and welcoming environment for all. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) for details on our expectations for participant behavior.
 
## Branching Model
 
- **Main Branch:** Contains the stable, production-ready code.
- **Feature Branches:** Create these from the main branch and name them appropriately (e.g., `feature/awesome-feature` or `bugfix/fix-issue-123`).
 
## Commit Message Guidelines
 
We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification with an additional requirement: **each commit message must include a scope** that reflects the part of the codebase being changed.
 
A commit message should be structured as follows:
 
> `<type>`(`<scope>`): `<subject>`
>  
> `<body>`

**Type**: Indicates the nature of the commit. Examples include:
 
```
feat (new feature)
 
fix (bug fix)
 
docs (documentation changes)
 
style (formatting, missing semicolons, etc.)
 
refactor (code changes that neither fix a bug nor add a feature)
 
perf (performance improvements)
 
test (adding or correcting tests)
 
chore (maintenance, tooling changes)
```
 
**Scope**: A concise description (lower-case and without spaces) of the area affected (e.g., cnn, convolution, layer).
 
**Subject**: A brief description of the change.
 
**Body (optional)**: More detailed explanatory text, if necessary.
 
### Examples
 
feat(transformers): implement #propagate method in decoder
 
fix(vector): resolve error when providing null data to Tensor#of
 
docs(examples): update with latest XOR NN tutorial
 
## Submitting Pull Requests
 
Before submitting a pull request:
* Ensure your changes pass all tests and build without errors.
* Write tests and update documentation as needed.
* Follow the project's coding style guidelines.
 
When ready:
 
1. Submit your pull request against the main branch.
2. Provide a detailed description of your changes, including references to any issues.
3. Respond to feedback promptly and be open to revisions.

## Reporting Issues
 
If you encounter any bugs or have suggestions for improvements:

* Open an issue detailing the problem or idea.
* Include steps to reproduce the issue, relevant logs, and screenshots if applicable.
 
## License
 
By contributing, you agree that your contributions will be licensed under the same terms as the project.
 
## Acknowledgements
 
Thank you for taking the time to contribute to Brain4J. Your efforts help us build a better project for everyone.
