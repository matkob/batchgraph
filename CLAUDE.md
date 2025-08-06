# Python Code Style Guide
- Prefer namespace imports over direct symbol imports.
  - Use `import typing as t` instead of `from typing import ...`.
- Prefer `type1 | None` over `t.Optional[type1]`.
- Use Google-style docstrings for all public functions, methods, and classes.
  - Keep descriptions short, clear, and informative.
  - Always document:
    - Arguments
    - Return values
    - Raised exceptions (if any)
- Avoid inline comments that explain what the code does.
  — The code should be self-explanatory.
  - Use inline comments only if:
    - The logic is non-obvious or complex.
    - There's a rationale that cannot be captured in code alone.
- Avoid abbreviations in variable names unless they are commonly understood.
- Prefer f-strings over % formatting or .format().
  - Exception: logging - use .format() over f-strings there.
- Use type hints for all function arguments and return values.
- Use t.Literal when working with fixed sets of values.
- Always use `logging` over `print`.

# Code Maintainability
- When testing new functionality, write unit tests over scripts.
- Keep the testing code and dependencies away from the main ones.
- When making an edit, check the surrounding code for consistency.
- Optimize the code for simplicity and clear abstractions.
- Check README files after major code changes and update their content accordingly.