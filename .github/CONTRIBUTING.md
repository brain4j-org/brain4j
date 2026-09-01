# Contributing to Brain4J

Brain4J is an open source project. If you are interested in making it better,
there are many ways you can contribute. For example, you can:

- Submit a bug report
- Suggest a new feature
- Provide feedback by commenting on feature requests/proposals
- Propose a patch by submitting a pull request
- Suggest or submit documentation improvements
- Review outstanding pull requests
- Answer questions from other users
- Share the software with other users who are interested
- Teach others to use the software
- Package and distribute the software in a downstream community

## Table of Contents

- [Bugs and Feature Requests](#bugs-and-feature-requests)
- [Getting Started](#getting-started)
- [Branching Model](#branching-model)
- [Patch Submission Tips](#patch-submission-tips)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Review Process](#review-process)
- [Timeline and Managing Expectations](#timeline-and-managing-expectations)
- [Code of Conduct](#code-of-conduct)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Bugs and Feature Requests

If you believe that you have found a bug or wish to propose a new feature,
please first search the existing [issues](https://github.com/brain4j-org/brain4j/issues) 
to see if it has already been reported. If you are unable to find an existing issue, 
consider using one of the provided templates to create a new issue and provide as many 
details as you can to assist in reproducing the bug or explaining your proposed feature.

When reporting bugs, please include:
- Steps to reproduce the issue
- Expected vs actual behavior
- Relevant logs, stack traces, and screenshots if applicable
- Your environment details (Java version, OS, Brain4J version)

> Please check the [SECURITY](SECURITY.md) guidelines if you think the issue you identified represents a vulnerability.

## Getting Started

1. **Fork the repository:** Create your own copy of the project.
2. **Set up your environment:** Ensure you have the required Java version and dependencies.
3. **Create a branch:** Start a new branch from main for your work.
4. **Make changes:** Implement your feature or fix with tests and documentation.
5. **Submit a Pull Request:** Follow the guidelines below.

## Branching Model

- **Main Branch:** Contains the stable, production-ready code.
- **Feature Branches:** Create these from the main branch with appropriate names:
  - `feature/awesome-feature` for new features
  - `bugfix/fix-issue-123` for bug fixes
  - `docs/update-readme` for documentation changes

## Patch Submission Tips

Patches should be submitted in the form of Pull Requests to the Brain4J repository 
on GitHub. Consider the following tips to ensure a smooth process:

- **Ensure your code compiles and passes all tests** before submitting. Even trivial 
  changes can cause unexpected problems.

- **If using AI (Artificial Intelligence) to assist in development**, please apply extra scrutiny to its 
  suggestions, in terms of both correctness and code quality.

- **Be understanding, patient, and friendly.** Maintainers may need time to review 
  your submissions before they can take action or respond. This does not mean your 
  contribution is not valued.

- **Limit your patches to the smallest reasonable change** to achieve your intended 
  goal. Do not make unnecessary formatting or indentation changes, but don't make 
  the patch so minimal that it isn't easy to read either. Consider the reviewer's 
  perspective.

- **Avoid large-scale refactoring** unless previously discussed with the maintainers. 
  These changes are difficult to review and may conflict with ongoing internal work.

- **Avoid "find and replace" style changes.** While it may be tempting to globally 
  update deprecated methods or change code style to fit personal preference, these 
  changes may have been intentionally left as-is.

- **Focus on real-world improvements.** Prioritize bug fixes discovered through 
  actual usage and features that clearly enhance Brain4J's functionality. Consider 
  opening a discussion first to ensure your efforts align with the project's goals.

- **Before submission, squash your commits** into logical units with clear messages 
  following our [commit guidelines](#commit-message-guidelines).

- **Isolate multiple patches.** If you wish to make several independent changes, 
  submit them as separate, smaller pull requests that can be reviewed more easily.

- **Be prepared to answer questions** from reviewers. They may request changes or 
  clarifications. Please accept this feedback constructively.

## Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification 
with an additional requirement: **each commit message must include a scope** that reflects 
the part of the codebase being changed.

### Format

```
<type>(<scope>): <subject>

<body>
```

### Types

| Type       | Description                                          |
|------------|------------------------------------------------------|
| `feat`     | New feature                                          |
| `fix`      | Bug fix                                              |
| `docs`     | Documentation changes                                |
| `style`    | Formatting, missing semicolons, etc.                 |
| `refactor` | Code changes that neither fix a bug nor add a feature|
| `perf`     | Performance improvements                             |
| `test`     | Adding or correcting tests                           |
| `chore`    | Maintenance, tooling changes                         |

### Scope

A concise description (lowercase, no spaces) of the area affected 
(e.g., `cnn`, `convolution`, `layer`, `transformers`, `vector`).

### Examples

```
feat(transformers): implement #propagate method in decoder

fix(vector): resolve error when providing null data to Tensor#of

docs(examples): update with latest XOR NN tutorial

perf(convolution): optimize forward pass computation

test(layer): add unit tests for activation functions
```

## Review Process

- We welcome code reviews from anyone. A maintainer is required to formally accept 
  and merge the changes.

- Reviewers will be looking for things like:
  - Threading issues and concurrency concerns
  - Performance implications
  - API design consistency
  - Duplication of existing functionality
  - Readability and code style
  - Avoidance of scope-creep
  - Test coverage

- Reviewers will likely ask questions to better understand your change.

- Reviewers will make comments about changes to your patch:
  - **MUST** - The change is required before merging
  - **SHOULD** - The change is suggested; further discussion may be needed
  - **COULD** - The change is optional, nice-to-have

## Timeline and Managing Expectations

As we continue to grow and learn best practices for running a successful open source 
project, our processes will likely evolve. Here's what to expect:

- **We prioritize small, focused contributions.** Bug fixes, documentation improvements, 
  and well-scoped features are easier to review and more likely to be merged quickly.

- **We are committed to code quality and stability.** Contributions will be carefully 
  reviewed to ensure they don't introduce bugs or regressions. This may take time.

- **Response times may vary.** Maintainers contribute in their spare time. A delayed 
  response does not mean your contribution is being ignored.

- **Not all contributions need to be merged.** Sometimes we may suggest maintaining 
  your enhancement as a separate extension or fork. This doesn't diminish its value 
  to the community.

- **Communication is key.** For larger changes, consider opening an issue or discussion 
  first to align with the project's direction before investing significant effort.

## Code of Conduct

We are committed to providing a friendly, safe, and welcoming environment for all. 
Please read our [Code of Conduct](CODE_OF_CONDUCT.md) for details on our expectations 
for participant behavior.

## License

By contributing, you agree that your contributions will be licensed under the same 
terms as the project (see [LICENSE](../LICENSE)).

## Acknowledgements

Thank you for taking the time to contribute to Brain4J. Your efforts help us build 
a better project for everyone. We appreciate every contribution, whether it's code, 
documentation, bug reports, or community support.
