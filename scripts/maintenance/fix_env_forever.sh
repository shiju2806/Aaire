#!/bin/bash
# Permanent fix for .env being overwritten repeatedly

echo "🔒 **PERMANENT .ENV PROTECTION SCRIPT**"
echo

cd ~/aaire

# 1. Create protected backup
cp .env .env.PRODUCTION
chmod 400 .env.PRODUCTION
echo "✅ Created protected production .env backup"

# 2. Add .env to .gitignore to prevent git overwrites
if ! grep -q "^\.env$" .gitignore 2>/dev/null; then
    echo ".env" >> .gitignore
    echo "✅ Added .env to .gitignore"
fi

# 3. Create restore script
cat > restore_env.sh << 'EOF'
#!/bin/bash
if [ -f .env.PRODUCTION ]; then
    cp .env.PRODUCTION .env
    echo "✅ .env restored from production backup"
else
    echo "❌ No production backup found"
fi
EOF
chmod +x restore_env.sh
echo "✅ Created restore_env.sh script"

# 4. Add git hook to prevent .env overwrites
mkdir -p .git/hooks
cat > .git/hooks/post-merge << 'EOF'
#!/bin/bash
# Restore .env after git operations
if [ -f .env.PRODUCTION ]; then
    cp .env.PRODUCTION .env
    echo "🔒 Auto-restored .env from production backup"
fi
EOF
chmod +x .git/hooks/post-merge
echo "✅ Created git hook to auto-restore .env"

echo
echo "🎉 **PROTECTION COMPLETE**"
echo "   • .env backed up to .env.PRODUCTION (read-only)"
echo "   • Added to .gitignore to prevent git overwrites"
echo "   • Created restore_env.sh for manual restore"
echo "   • Git hook will auto-restore after pulls"
echo
echo "🔧 **To manually restore .env:**"
echo "   ./restore_env.sh"