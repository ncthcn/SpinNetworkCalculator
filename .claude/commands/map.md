---
description: Automatically updates the Project Map in CLAUDE.md using the tree command
---

Please perform these steps:
1. Run the bash command: `tree -I "__pycache__|.git|venv" -L 2`
2. Open `CLAUDE.md`.
3. Locate the `## Project Map` section.
4. Replace the contents of that section with the output from the `tree` command.
5. Confirm once the update is complete.
