# Documentation style guide

Write public documentation for Isvik.cpp using the Google Developer Documentation Style Guide. Follow project-specific terminology when it conflicts with a general recommendation.

## Write for developers

- Put the reader's task first.
- Use active voice, present tense, and direct instructions.
- Use short sentences and define unfamiliar terms.
- Prefer concrete examples to abstract descriptions.
- State limitations where they affect the reader's next step.

## Organize pages

- Use sentence case for titles and headings.
- Use one H1 heading per page.
- Use H2 headings for major topics and H3 headings for related tasks.
- Put prerequisites before procedures.
- Use numbered lists for ordered steps and bullets for unordered items.
- Link with descriptive text.

## Format technical content

- Use fenced code blocks with a language identifier for commands and source code.
- Use inline code for file names, paths, commands, flags, API fields, and code symbols.
- Show complete commands that readers can copy.
- Use placeholders for values that depend on the reader's machine.
- Keep examples consistent with supported behavior.

## Keep documentation accurate

- Update relevant documentation with behavior changes.
- Describe tested behavior separately from planned behavior.
- Do not claim that a model, backend, or API feature works unless it has been verified.
- Keep machine-specific logs and paths out of public guides.

## C++ style

C++ source follows the Google C++ Style Guide and the repository's `.clang-format` configuration. Format changed C++ files with clang-format before submitting a pull request.

## References

- [Google Developer Documentation Style Guide](https://developers.google.com/style)
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
