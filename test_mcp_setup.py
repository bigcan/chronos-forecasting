#!/usr/bin/env python3
"""
Test script to verify MCP setup for Chronos Forecasting project.
This script checks if the MCP configuration is properly set up.
"""

import json
import os
import sys
from pathlib import Path

def test_mcp_configuration():
    """Test the MCP configuration files."""
    print("🔧 Testing MCP Configuration for Chronos Forecasting")
    print("=" * 60)
    
    # Check if mcp.json exists
    mcp_config_path = Path("mcp.json")
    if mcp_config_path.exists():
        print("✅ mcp.json found")
        try:
            with open(mcp_config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ mcp.json is valid JSON")
            print(f"📋 Configured servers: {list(config.get('mcpServers', {}).keys())}")
        except json.JSONDecodeError as e:
            print(f"❌ mcp.json has invalid JSON: {e}")
            return False
    else:
        print("❌ mcp.json not found")
        return False
    
    # Check Cursor settings
    cursor_settings_path = Path(".claude/settings.local.json")
    if cursor_settings_path.exists():
        print("✅ Cursor settings found")
        try:
            with open(cursor_settings_path, 'r') as f:
                settings = json.load(f)
            permissions = settings.get('permissions', {}).get('allow', [])
            mcp_permissions = [p for p in permissions if p.startswith('mcp__')]
            print(f"✅ Cursor settings are valid JSON")
            print(f"📋 MCP permissions: {len(mcp_permissions)} configured")
        except json.JSONDecodeError as e:
            print(f"❌ Cursor settings have invalid JSON: {e}")
            return False
    else:
        print("❌ Cursor settings not found")
        return False
    
    # Check workspace structure
    print("\n📁 Workspace Structure Check:")
    important_dirs = [
        "src/chronos",
        "gold_futures_analysis", 
        "log_return_approach",
        "scripts",
        "test"
    ]
    
    for dir_path in important_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"⚠️  {dir_path}/ (not found)")
    
    # Check for Node.js
    print("\n🔧 Environment Check:")
    try:
        import subprocess
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js available: {result.stdout.strip()}")
        else:
            print("❌ Node.js not available")
    except FileNotFoundError:
        print("❌ Node.js not found in PATH")
    
    # Check npm
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ npm available: {result.stdout.strip()}")
        else:
            print("❌ npm not available")
    except FileNotFoundError:
        print("❌ npm not found in PATH")
    
    print("\n" + "=" * 60)
    print("🎉 MCP Setup Test Complete!")
    print("\nNext steps:")
    print("1. Restart Cursor to apply MCP configuration")
    print("2. Test filesystem operations in your workspace")
    print("3. Configure API keys if needed (see MCP_SETUP.md)")
    print("4. Start using MCP servers for enhanced development")
    
    return True

def main():
    """Main test function."""
    success = test_mcp_configuration()
    if success:
        print("\n✅ MCP setup appears to be configured correctly!")
        sys.exit(0)
    else:
        print("\n❌ MCP setup has issues that need to be resolved.")
        sys.exit(1)

if __name__ == "__main__":
    main() 