#!/bin/bash

# Install git pre-commit hook to build and check compilation

HOOK_FILE=".git/hooks/pre-commit"

cat > "$HOOK_FILE" << 'EOF'
#!/bin/bash

echo "Running pre-commit build check..."

# Build shared-types
echo "Building shared-types..."
cd packages/shared-types
pnpm build
if [ $? -ne 0 ]; then
  echo "❌ shared-types build failed"
  exit 1
fi
cd ../..

# Build UI
echo "Building UI..."
cd packages/ui
pnpm build
if [ $? -ne 0 ]; then
  echo "❌ UI build failed"
  exit 1
fi
cd ../..

echo "✓ All builds passed"
EOF

chmod +x "$HOOK_FILE"

echo "✓ Pre-commit hook installed at $HOOK_FILE"
echo "The hook will run build checks before each commit."
