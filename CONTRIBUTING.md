# Contributing to Isvik.cpp

Thank you for helping improve Isvik.cpp. This guide explains how to prepare, build, test, and submit a change.

## Before you start

- Search open issues and pull requests for related work.
- Open an issue for a large change so maintainers can discuss its scope.
- Keep each pull request focused on one user-visible change or one maintenance task.

## Set up the project

Install CMake 3.21 or later, Git, and a C++20 compiler. Follow the [build guide](docs/BUILDING.md) to configure a supported preset.

## Make a change

1. Create a branch for your change.
2. Make the smallest complete change that solves the issue.
3. Add or update tests for behavior changes.
4. Update user-facing documentation when commands, options, APIs, or model support change.
5. Run the relevant build and tests.
6. Review the diff for unrelated files, local paths, generated output, and secrets.

## Follow the project style

- Follow the Google C++ Style Guide and `.clang-format` for C++.
- Follow the [documentation style guide](docs/STYLE_GUIDE.md) for Markdown.
- Use clear names and explain non-obvious design decisions in comments.
- Document the license and build requirements for each new runtime dependency.

## Validate your change

Run the test preset for the configuration you changed:

~~~powershell
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release --parallel
ctest --preset windows-msvc-release
~~~

~~~sh
cmake --preset linux-gcc-release
cmake --build --preset linux-gcc-release --parallel
ctest --preset linux-gcc-release
~~~

If a backend requires hardware or an SDK that you do not have, state that limitation in the pull request. Do not report an unrun check as passing.

## Submit a pull request

Use a descriptive title and explain what changed, why the change is needed, how you validated it, and any known limitations.

Link related issues when available. Include screenshots for visible UI changes and sanitized logs for runtime changes. Do not include model files, credentials, personal paths, or generated build output.

By submitting a contribution, you agree that your contribution is licensed under the Apache License, Version 2.0, unless you and the project maintainers agree otherwise in writing.
