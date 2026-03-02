// mini-openclaw Web UI
const messagesEl = document.getElementById('messages');
const inputForm = document.getElementById('input-form');
const userInput = document.getElementById('user-input');
const statusEl = document.getElementById('status');
const agentTreeEl = document.getElementById('agent-tree');
const toolLogEl = document.getElementById('tool-log');

let ws = null;
let sessionId = null;

function connect() {
    // First create a session via REST
    fetch('/api/sessions', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            sessionId = data.session_id;
            connectWebSocket(sessionId);
        })
        .catch(err => {
            addMessage('system', `Failed to create session: ${err.message}`);
            statusEl.textContent = 'Disconnected';
            statusEl.className = 'status disconnected';
        });
}

function connectWebSocket(sid) {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws/sessions/${sid}`);

    ws.onopen = () => {
        statusEl.textContent = 'Connected';
        statusEl.className = 'status connected';
        addMessage('system', `Connected. Session: ${sid.slice(0, 20)}...`);
    };

    ws.onclose = () => {
        statusEl.textContent = 'Disconnected';
        statusEl.className = 'status disconnected';
        setTimeout(connect, 3000);
    };

    ws.onerror = () => {
        statusEl.textContent = 'Error';
        statusEl.className = 'status disconnected';
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleServerMessage(data);
    };
}

function handleServerMessage(data) {
    switch (data.type) {
        case 'assistant':
            addMessage(data.success ? 'assistant' : 'error', data.content);
            if (data.tokens_used) {
                addMessage('system',
                    `tokens: ${data.tokens_used} | tools: ${data.tool_calls_made} | ` +
                    `children: ${data.children_spawned.length}`);
            }
            break;
        case 'tool_call':
            addToolLog('call', `${data.tool}(${JSON.stringify(data.args).slice(0, 60)})`);
            break;
        case 'tool_result':
            addToolLog(data.success ? 'result-ok' : 'result-fail',
                `  \u2192 ${data.success ? 'OK' : 'FAIL'} ${data.tool}`);
            break;
        case 'agent_spawned':
            addAgentNode(data);
            break;
        case 'agent_completed':
            updateAgentState(data.agent_id, 'completed');
            break;
        case 'agent_failed':
            updateAgentState(data.agent_id, 'failed');
            break;
        case 'approval_request':
            showApprovalDialog(data);
            break;
        case 'approval_resolved':
            removeApprovalDialog(data.request_id);
            addToolLog(data.approved ? 'result-ok' : 'result-fail',
                `  \u2192 ${data.approved ? 'APPROVED' : 'DENIED'} ${data.tool}`);
            break;
        case 'tools_list':
            let text = 'Available tools:\n';
            data.tools.forEach(t => {
                const flag = t.requires_approval ? ' [approval required]' : '';
                text += `  ${t.name}: ${t.description}${flag}\n`;
            });
            addMessage('system', text);
            break;
        case 'hitl_pending':
            if (data.requests.length === 0) {
                addMessage('system', 'No pending approval requests.');
            } else {
                let msg = 'Pending approval requests:\n';
                data.requests.forEach(r => {
                    msg += `  ${r.request_id}: ${r.tool_name} - ${r.description}\n`;
                });
                addMessage('system', msg);
            }
            break;
        case 'error':
            addMessage('error', data.message);
            break;
        case 'session_created':
            sessionId = data.session_id;
            break;
    }
}

// --- HITL Approval Dialog ---

function showApprovalDialog(data) {
    addToolLog('approval', '\u26a0 APPROVAL NEEDED: ' + data.tool);

    // Remove existing dialog for same request if any
    removeApprovalDialog(data.request_id);

    const overlay = document.createElement('div');
    overlay.className = 'approval-overlay';
    overlay.id = 'approval-' + data.request_id;

    const argsStr = JSON.stringify(data.arguments, null, 2);
    const argsPreview = argsStr.length > 300 ? argsStr.slice(0, 300) + '...' : argsStr;

    overlay.innerHTML =
        '<div class="approval-dialog">' +
            '<div class="approval-title">\u26a0 APPROVAL REQUIRED</div>' +
            '<div class="approval-field">' +
                '<span class="approval-label">Tool:</span> ' +
                '<span class="approval-tool">' + escapeHtml(data.tool) + '</span>' +
            '</div>' +
            '<div class="approval-field">' +
                '<span class="approval-label">Description:</span> ' +
                '<span>' + escapeHtml(data.description) + '</span>' +
            '</div>' +
            '<div class="approval-args">' +
                '<span class="approval-label">Arguments:</span>' +
                '<pre>' + escapeHtml(argsPreview) + '</pre>' +
            '</div>' +
            '<div class="approval-buttons">' +
                '<button class="btn-approve" data-rid="' + data.request_id + '">Approve</button>' +
                '<button class="btn-always" data-rid="' + data.request_id + '">Always (session)</button>' +
                '<button class="btn-deny" data-rid="' + data.request_id + '">Deny</button>' +
            '</div>' +
        '</div>';

    // Attach event listeners
    overlay.querySelector('.btn-approve').addEventListener('click', function() {
        respondApproval(this.dataset.rid, true, false);
    });
    overlay.querySelector('.btn-always').addEventListener('click', function() {
        respondApproval(this.dataset.rid, true, true);
    });
    overlay.querySelector('.btn-deny').addEventListener('click', function() {
        respondApproval(this.dataset.rid, false, false);
    });

    document.body.appendChild(overlay);
}

function removeApprovalDialog(requestId) {
    const el = document.getElementById('approval-' + requestId);
    if (el) el.remove();
}

function respondApproval(requestId, approved, remember) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'approval_response',
            request_id: requestId,
            approved: approved,
            remember: remember,
        }));
    }
    removeApprovalDialog(requestId);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// --- End HITL ---

function addMessage(type, content) {
    const div = document.createElement('div');
    div.className = 'message ' + type;
    div.textContent = content;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addToolLog(cls, text) {
    const div = document.createElement('div');
    div.className = 'tool-entry ' + cls;
    div.textContent = text;
    toolLogEl.appendChild(div);
    toolLogEl.scrollTop = toolLogEl.scrollHeight;
}

const agentNodes = {};

function addAgentNode(data) {
    const div = document.createElement('div');
    div.className = 'agent-node depth-' + Math.min(data.depth, 2);
    div.id = 'agent-' + data.agent_id;
    div.innerHTML =
        '<span class="agent-state running">running</span> ' +
        '<strong>Agent</strong> d=' + data.depth +
        '<br><small>' + data.tools.join(', ') + '</small>';

    if (data.parent_id && agentNodes[data.parent_id]) {
        agentNodes[data.parent_id].appendChild(div);
    } else {
        agentTreeEl.appendChild(div);
    }
    agentNodes[data.agent_id] = div;
}

function updateAgentState(agentId, state) {
    const node = agentNodes[agentId];
    if (node) {
        const stateEl = node.querySelector('.agent-state');
        if (stateEl) {
            stateEl.className = 'agent-state ' + state;
            stateEl.textContent = state;
        }
    }
}

// Input handling
inputForm.addEventListener('submit', function(e) {
    e.preventDefault();
    const text = userInput.value.trim();
    if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

    userInput.value = '';

    if (text.startsWith('/')) {
        ws.send(JSON.stringify({ type: 'command', command: text }));
        addMessage('system', text);
    } else {
        ws.send(JSON.stringify({ type: 'message', content: text }));
        addMessage('user', text);
    }
});

// Start
connect();
