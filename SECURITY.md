# Security

Report security vulnerabilities privately through [GitHub private vulnerability reporting](https://github.com/hyperspy/hyperspy/security/advisories/new).

Don't open a public issue for a security vulnerability.

## Scope and threat model

HyperSpy treats the content of files, stored models and learning-results files
as untrusted input, but it does not sandbox them.

- **Addressed (HyperSpy 2.6.0 and later):** code execution through file-derived
expressions and through stored-model or learning-results metadata, and
unpickling of legacy content unless the caller passes `trusted=True`.
- **Not addressed:** resource exhaustion. A crafted or merely very large file
can exhaust memory or CPU while it is read or while an expression is
evaluated. File-declared sizes, compression and array shapes are not bounded
as a security measure, `lazy=True` is not a mitigation, and the expression
parser's limits reject oversized literals and obvious constant-magnitude
blowups only: they are best effort, not a resource bound.
- **Elsewhere:** file-format readers live in RosettaSciIO; report reader
vulnerabilities to RosettaSciIO through its own private reporting. Vulnerable
versions of third-party dependencies belong to those projects.

Open files and stored models only from sources you trust. When that is not
possible, read them in a separate process or environment that you can
terminate and that holds no credentials.
