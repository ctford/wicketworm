#!/bin/bash

echo "Installing git hooks..."

# Copy pre-commit hook
cp .githooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

echo "✓ Git hooks installed"
echo ""
echo "The pre-commit hook will now:"
echo "  1. Check Python compilation (syntax errors)"
echo "  2. Build shared-types (TypeScript compilation)"
echo "  3. Build UI (TypeScript + Vite bundling)"
