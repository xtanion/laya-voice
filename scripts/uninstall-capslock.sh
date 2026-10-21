#!/bin/zsh
hidutil property --set '{"UserKeyMapping":[]}'
launchctl bootout "gui/$(id -u)/ai.laya.capslock" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/ai.laya.capslock.plist"
echo "Caps Lock restored."
