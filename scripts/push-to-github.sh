#!/usr/bin/env bash
# Push skeletal structure and component branches to a new GitHub repo.
# Usage:
#   1. Create a new repository on GitHub (e.g. vinyl-management-system). Do NOT add a README or .gitignore.
#   2. Set REMOTE_URL below or pass as first argument, then run:
#      ./scripts/push-to-github.sh
#      # or
#      ./scripts/push-to-github.sh git@github.com:YOUR_USERNAME/vinyl-management-system.git

set -e
REMOTE_URL="${1:-}"
if [ -z "$REMOTE_URL" ]; then
  echo "Usage: $0 <remote-url>"
  echo "Example: $0 git@github.com:yourusername/vinyl-management-system.git"
  exit 1
fi

if ! git remote get-url origin 2>/dev/null; then
  git remote add origin "$REMOTE_URL"
else
  git remote set-url origin "$REMOTE_URL"
fi

echo "Pushing main and component branches..."
git push -u origin main
git push origin recommender nlp-classifier price-estimator web

echo "Done. Branches on GitHub:"
echo "  main (default)"
echo "  recommender"
echo "  nlp-classifier"
echo "  price-estimator"
echo "  web"
