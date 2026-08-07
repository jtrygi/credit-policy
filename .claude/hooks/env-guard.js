#!/usr/bin/env node
const path = require('path');

const PROJECT_ROOT = path.resolve('C:/Users/jhtry/Projects/credit-policy').toLowerCase();
const ENV_NAME_RE = /^\.env(\..+)?$/i;
const ENV_EXAMPLE_RE = /^\.env\.example$/i;

function deny(reason) {
  process.stdout.write(JSON.stringify({
    hookSpecificOutput: {
      hookEventName: 'PreToolUse',
      permissionDecision: 'deny',
      permissionDecisionReason: reason
    }
  }));
  process.exit(0);
}

function isBlockedEnvPath(candidatePath, base) {
  const resolved = path.resolve(base, candidatePath);
  const lower = resolved.toLowerCase();
  if (!lower.startsWith(PROJECT_ROOT + path.sep) && lower !== PROJECT_ROOT) return false;
  const base_ = path.basename(resolved);
  if (ENV_EXAMPLE_RE.test(base_)) return false;
  return ENV_NAME_RE.test(base_);
}

let input = '';
process.stdin.on('data', (d) => { input += d; });
process.stdin.on('end', () => {
  let data;
  try {
    data = JSON.parse(input);
  } catch (e) {
    process.exit(0);
  }

  const toolName = data.tool_name;
  const toolInput = data.tool_input || {};
  const base = data.cwd || PROJECT_ROOT;

  if (['Read', 'Write', 'Edit', 'NotebookEdit'].includes(toolName)) {
    const value = toolInput.file_path || toolInput.notebook_path || toolInput.path;
    if (typeof value === 'string' && isBlockedEnvPath(value, base)) {
      return deny(`Blocked: "${toolName}" targets "${value}" — .env files are off-limits to the assistant in this project (use .env.example instead).`);
    }
  }

  if (toolName === 'Bash' || toolName === 'PowerShell') {
    const cmd = String(toolInput.command || '');
    if (/(^|[\s;|&])(cat|type|more|less|Get-Content|gc)\b[^|;&\n]*\.env(\.[A-Za-z0-9_-]+)?\b/i.test(cmd) &&
        !/\.env\.example\b/i.test(cmd)) {
      return deny(`Blocked: command appears to read a .env file: "${cmd}"`);
    }
  }

  process.exit(0);
});
