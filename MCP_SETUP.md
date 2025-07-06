# MCP (Model Context Protocol) Setup for Cursor

This workspace has been configured with comprehensive MCP servers to enhance your development experience with the Chronos forecasting project.

## What's Configured

### Core MCP Servers

1. **Filesystem Server** (`@modelcontextprotocol/server-filesystem`)
   - Full file system access within the workspace
   - Read, write, delete, rename files
   - Search and list directories
   - Perfect for navigating and managing your Chronos project files

2. **Git Server** (`@modelcontextprotocol/server-git`)
   - Git status, diff, commits, branches
   - Repository management and version control
   - Essential for tracking changes in your forecasting models

3. **GitHub Server** (`@modelcontextprotocol/server-github`)
   - Search repositories, get issues, pull requests
   - Repository information and collaboration
   - Requires GitHub Personal Access Token

4. **Database Servers**
   - **PostgreSQL** (`@modelcontextprotocol/server-postgres`) - For production data
   - **SQLite** (`@modelcontextprotocol/server-sqlite`) - For local development
   - Query and execute database operations

5. **Search and Web**
   - **Brave Search** (`@modelcontextprotocol/server-brave-search`) - Web search capabilities
   - **Puppeteer** (`@modelcontextprotocol/server-puppeteer`) - Web automation and scraping
   - Useful for gathering financial data and research

6. **Memory and Thinking**
   - **Memory Server** (`@modelcontextprotocol/server-memory`) - Persistent memory across sessions
   - **Sequential Thinking** (`@modelcontextprotocol/server-sequential-thinking`) - Enhanced reasoning

## Configuration Files

- `mcp.json` - Main MCP server configuration
- `.claude/settings.local.json` - Cursor permissions and settings

## Setup Instructions

### 1. Install Node.js and npm
Make sure you have Node.js installed (version 16 or higher recommended).

### 2. Configure API Keys (Optional)

For enhanced functionality, add your API keys to the `mcp.json` file:

```json
{
  "mcpServers": {
    "github": {
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your_github_token_here"
      }
    },
    "brave-search": {
      "env": {
        "BRAVE_API_KEY": "your_brave_api_key_here"
      }
    },
    "postgres": {
      "env": {
        "POSTGRES_CONNECTION_STRING": "your_postgres_connection_string_here"
      }
    }
  }
}
```

### 3. Restart Cursor
After making changes to the MCP configuration, restart Cursor to apply the new settings.

## Usage Examples

### Filesystem Operations
- Navigate through your Chronos project structure
- Read and analyze forecasting results
- Create new analysis scripts
- Manage data files and models

### Git Operations
- Check project status and changes
- Review commit history
- Manage branches for different forecasting approaches

### Database Operations
- Query financial data
- Store forecasting results
- Analyze historical performance

### Web Research
- Search for financial market data
- Gather economic indicators
- Research forecasting methodologies

## Benefits for Chronos Forecasting

1. **Enhanced File Management**: Easily navigate through your complex project structure with multiple analysis directories
2. **Version Control**: Track changes to your forecasting models and configurations
3. **Data Access**: Connect to databases for financial data and results storage
4. **Research Capabilities**: Search for market data and economic indicators
5. **Memory Persistence**: Remember previous analysis results and configurations
6. **Automation**: Web scraping for real-time financial data

## Troubleshooting

### Common Issues

1. **MCP Servers Not Loading**
   - Ensure Node.js is installed
   - Check that npm packages can be installed
   - Restart Cursor after configuration changes

2. **Permission Errors**
   - Verify the permissions in `.claude/settings.local.json`
   - Check that the workspace path is correct in `mcp.json`

3. **API Key Issues**
   - Ensure API keys are properly formatted
   - Check that services are accessible from your network

### Getting Help

- Check the MCP documentation: https://modelcontextprotocol.io/
- Review server-specific documentation for each package
- Test individual servers to ensure they're working correctly

## Security Notes

- Keep API keys secure and don't commit them to version control
- Use environment variables for sensitive configuration
- Regularly rotate API keys for production use
- Review permissions and only enable what you need

## Next Steps

1. Test the filesystem server by exploring your project structure
2. Configure your GitHub token for repository access
3. Set up database connections if needed
4. Explore web search capabilities for financial research
5. Use memory server to persist analysis results

Your Chronos forecasting project is now equipped with powerful MCP capabilities to enhance your development workflow! 