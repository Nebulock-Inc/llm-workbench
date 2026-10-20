# Security Policy

## Supported versions

`llm-workbench` is pre-1.0 and under active development. Security fixes are
applied to the latest release on the `main` branch; older versions are not
maintained. Upgrade to the newest version before reporting an issue.

## Reporting a vulnerability

Please report security vulnerabilities privately — **do not open a public
issue.**

Use GitHub's private vulnerability reporting: go to the repository's
**Security** tab and click **Report a vulnerability**. This opens a private
advisory visible only to you and the maintainers.

Please include:

- a description of the issue and its impact,
- steps to reproduce (or a proof of concept), and
- any affected versions you know of.

We will acknowledge your report, keep you updated on our assessment, and
coordinate a fix and disclosure timeline with you. Please give us a reasonable
window to address the issue before any public disclosure.

## Scope

`llm-workbench` reads provider credentials (API keys, AWS profiles) from the
environment and a local `.env` file; it does not transmit them anywhere except
to the model provider you configure. Keep your `.env` out of version control
(it is gitignored by default) and treat run outputs as potentially containing
sensitive prompt/response data.
