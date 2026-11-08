#!/usr/bin/env bash
# Install the secrets check as a git pre-commit hook (hooks are not versioned, so run this once per clone).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
printf '#!/usr/bin/env bash\nexec bash tools/check_secrets.sh\n' > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
echo "pre-commit hook installed"
