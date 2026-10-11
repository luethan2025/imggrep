## Pre-commit 
This hook is called before obtaining the proposed commit message. Exiting with anything other than zero will abort the commit. It is used to check the commit itself (rather than the message).

## Invoked by 
```bash
git commit
```

## Setup
Configure Git to use this hooks directory:
```bash
git config core.hooksPath .githooks
```
