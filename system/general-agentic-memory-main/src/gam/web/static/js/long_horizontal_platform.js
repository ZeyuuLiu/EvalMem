// ══════════════════════════════════════════════════════════════════════════
// State
// ══════════════════════════════════════════════════════════════════════════
let sessionId = null;
let gamDir = null;
let selectedFiles = [];
let isRunning = false;
let isChatProcessing = false;
let searchQueries = [];

// ══════════════════════════════════════════════════════════════════════════
// File Upload
// ══════════════════════════════════════════════════════════════════════════
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');

uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', e => { if (e.target.files.length) handleFiles(e.target.files); });

function handleFiles(fileList) {
    const files = Array.from(fileList);
    files.forEach(file => {
        const ext = file.name.split('.').pop().toLowerCase();
        if (['pdf', 'txt', 'md'].includes(ext)) {
            selectedFiles.push(file);
        }
    });
    renderDocList();
}

function renderDocList() {
    const docList = document.getElementById('doc-list');
    const badge = document.getElementById('doc-count-badge');
    docList.innerHTML = '';

    if (!selectedFiles.length) {
        badge.style.display = 'none';
        return;
    }

    badge.style.display = 'inline-flex';
    badge.textContent = selectedFiles.length;

    selectedFiles.forEach((file, i) => {
        const item = document.createElement('div');
        item.className = 'doc-item';
        item.innerHTML = `
            <span class="name">📄 ${esc(file.name)}</span>
            <span class="size">${formatSize(file.size)}</span>
            <button class="remove-btn" onclick="removeFile(${i})">✕</button>
        `;
        docList.appendChild(item);
    });
}

function removeFile(i) {
    selectedFiles.splice(i, 1);
    renderDocList();
}

// ══════════════════════════════════════════════════════════════════════════
// Upload & Request
// ══════════════════════════════════════════════════════════════════════════
async function startRequest() {
    const question = document.getElementById('question-input').value.trim();
    if (!selectedFiles.length) return alert('Please upload documents first.');
    if (!question) return alert('Please enter a research question.');
    if (isRunning) return;

    isRunning = true;
    document.getElementById('request-btn').disabled = true;
    // setRequestStatus('Uploading...', 'badge-warning');
    setConvStatus('requesting', 'Uploading docs...');

    try {
        // 1. Upload documents
        const formData = new FormData();
        selectedFiles.forEach(f => formData.append('files', f));
        const uploadRes = await fetch('/api/long_horizontal/upload_docs', { method: 'POST', body: formData });
        const uploadData = await uploadRes.json();
        if (!uploadData.success) throw new Error(uploadData.error);

        sessionId = uploadData.session_id;
        gamDir = uploadData.gam_dir;
        // setRequestStatus('Requesting...', 'badge-warning');
        setConvStatus('requesting', 'Agent requesting...');

        // 2. Clear conversation
        clearConversation();
        clearQueryEvolution();
        searchQueries = [];

        // 3. Start agent request (SSE)
        const requestForm = new FormData();
        requestForm.append('session_id', sessionId);
        requestForm.append('question', question);

        const response = await fetch('/api/long_horizontal/start_request', { method: 'POST', body: requestForm });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const ev = JSON.parse(line.slice(6));
                        handleAgentEvent(ev);
                        if (ev.type === 'complete' || ev.type === 'error') break;
                    } catch (_) {}
                }
            }
        }
    } catch (err) {
        // setRequestStatus('Error', 'badge-warning');
        setConvStatus('', 'Error');
        addConvError(err.message);
    } finally {
        isRunning = false;
        document.getElementById('request-btn').disabled = false;
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Agent Event Handler
// ══════════════════════════════════════════════════════════════════════════
function handleAgentEvent(ev) {
    switch (ev.type) {
        case 'init':
            showSystemPrompt(ev.system_prompt);
            addConvUserQuery(ev.user_query);
            break;

        case 'round':
            addConvRoundMarker(ev.round, ev.max_rounds);
            break;

        case 'assistant':
            addConvAssistant(ev.content, ev.has_tool_calls);
            break;

        case 'tool_call':
            addConvToolCall(ev);
            if (ev.name === 'search' && ev.args && ev.args.query) {
                addQueryEvolution(ev.args.query, searchQueries.length);
                searchQueries.push(ev.args.query);
            }
            break;

        case 'tool_result':
            addConvToolResult(ev);
            break;

        case 'memorize_update':
            handleMemorizeUpdate(ev);
            break;

        case 'nudge':
            addConvNudge(ev.content);
            break;

        case 'complete':
            addConvFinalAnswer(ev.answer);
            setRunStatus('Done', 'badge-success');
            setConvStatus('', 'Complete');
            if (ev.gam_dir) { gamDir = ev.gam_dir; refreshGAMTree(); }
            break;

        case 'error':
            addConvError(ev.error);
            setRunStatus('Error', 'badge-warning');
            setConvStatus('', 'Error');
            break;
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Conversation Timeline Rendering
// ══════════════════════════════════════════════════════════════════════════
function clearConversation() {
    document.getElementById('conversation-timeline').innerHTML = '';
    document.getElementById('system-prompt-banner').style.display = 'none';
    document.getElementById('conv-empty-state').style.display = 'none';
}

function showSystemPrompt(text) {
    const banner = document.getElementById('system-prompt-banner');
    const content = document.getElementById('system-prompt-content');
    banner.style.display = 'block';
    content.textContent = text;
}

function toggleSystemPrompt() {
    const content = document.getElementById('system-prompt-content');
    const btn = content.parentElement.querySelector('.system-prompt-toggle');
    content.classList.toggle('expanded');
    btn.textContent = content.classList.contains('expanded') ? '▲ Collapse' : '▼ Expand';
}

function addConvUserQuery(text) {
    const timeline = document.getElementById('conversation-timeline');
    const row = document.createElement('div');
    row.className = 'conv-row request';
    row.innerHTML = `
        <div class="conv-bubble user-query">
            <div class="conv-bubble-label"><span class="label-icon">👤</span> User Query</div>
            <div class="conv-bubble-body">${esc(text)}</div>
        </div>
    `;
    timeline.appendChild(row);
    scrollConv();
}

function addConvRoundMarker(round, max) {
    const timeline = document.getElementById('conversation-timeline');
    const marker = document.createElement('div');
    marker.className = 'conv-round-marker';
    marker.innerHTML = `<div class="line"></div><div class="label">🔄 Round ${round}/${max}</div><div class="line"></div>`;
    timeline.appendChild(marker);
    scrollConv();
}

function addConvAssistant(content, hasToolCalls) {
    if (!content) return;
    const timeline = document.getElementById('conversation-timeline');
    const row = document.createElement('div');
    row.className = 'conv-row request';
    const label = hasToolCalls ? '🤖 Assistant (thinking)' : '🤖 Assistant';
    const truncated = content.length > 500;
    const displayContent = truncated ? content.slice(0, 500) + '...' : content;
    row.innerHTML = `
        <div class="conv-bubble assistant-msg">
            <div class="conv-bubble-label"><span class="label-icon">🤖</span> ${label}</div>
            <div class="conv-bubble-body">${esc(displayContent)}</div>
            ${truncated ? `<button class="conv-bubble-toggle" onclick="toggleBubble(this, '${escAttr(content)}')">▼ Show more</button>` : ''}
        </div>
    `;
    timeline.appendChild(row);
    scrollConv();
}

function addConvToolCall(ev) {
    const timeline = document.getElementById('conversation-timeline');
    const row = document.createElement('div');
    row.className = 'conv-row request';

    const icons = { search: '🔍', memorize: '💾', recall: '📖' };
    const icon = icons[ev.name] || '🔧';
    const cssClass = ev.name + '-call';

    let argsDisplay = '';
    if (ev.name === 'search') {
        argsDisplay = `query: "${esc(ev.args.query || '')}"`;
    } else if (ev.name === 'memorize') {
        argsDisplay = `indices: [${(ev.args.search_indices || []).join(', ')}]`;
        if (ev.args.question) argsDisplay += `\nquestion: "${esc(ev.args.question)}"`;
    } else if (ev.name === 'recall') {
        argsDisplay = `question: "${esc(ev.args.question || '')}"`;
    } else {
        argsDisplay = JSON.stringify(ev.args, null, 2);
    }

    row.innerHTML = `
        <div class="conv-bubble tool-call ${cssClass}">
            <div class="conv-bubble-label"><span class="label-icon">${icon}</span> ${ev.name}()</div>
            <div class="conv-bubble-body">${esc(argsDisplay)}</div>
        </div>
    `;
    timeline.appendChild(row);
    scrollConv();
}

function addConvToolResult(ev) {
    const timeline = document.getElementById('conversation-timeline');
    const row = document.createElement('div');
    row.className = 'conv-row response';

    const cssClass = ev.name + '-result';
    const id = ev.search_index !== undefined ? `tool-result-search-${ev.search_index}` : `tool-result-${ev.tool_call_id}`;

    let content = ev.content || '';
    const truncated = content.length > 600;
    const displayContent = truncated ? content.slice(0, 600) + '...' : content;

    row.innerHTML = `
        <div class="conv-bubble tool-result ${cssClass}" id="${id}">
            <div class="conv-bubble-label"><span class="label-icon">📨</span> ${ev.name} result${ev.search_index !== undefined ? ` #${ev.search_index}` : ''}</div>
            <div class="conv-bubble-body">${esc(displayContent)}</div>
            ${truncated ? `<button class="conv-bubble-toggle" onclick="toggleResultBubble(this, '${id}')">▼ Show more</button>` : ''}
        </div>
    `;
    row.dataset.fullContent = content;
    timeline.appendChild(row);
    scrollConv();
}

function handleMemorizeUpdate(ev) {
    const id = `tool-result-search-${ev.search_index}`;
    const bubble = document.getElementById(id);
    if (!bubble) return;

    bubble.classList.add('memorized-update');
    const label = bubble.querySelector('.conv-bubble-label');
    if (label) {
        label.innerHTML = `<span class="label-icon">💾</span> search result #${ev.search_index} <span class="memorize-badge">MEMORIZED</span>`;
    }

    // Prefer gam_answer (the actual compressed content) over new_content (which has a tag prefix)
    const newContent = ev.gam_answer || ev.new_content || '';
    const body = bubble.querySelector('.conv-bubble-body');
    if (body) {
        const truncated = newContent.length > 600;
        body.textContent = truncated ? newContent.slice(0, 600) + '...' : newContent;
        body.classList.remove('expanded');

        // Update the parent row's fullContent so toggle button shows correct content
        const row = bubble.closest('.conv-row');
        if (row) row.dataset.fullContent = newContent;

        // Remove any existing toggle button and add a new one if still needed
        const oldToggle = bubble.querySelector('.conv-bubble-toggle');
        if (oldToggle) oldToggle.remove();
        if (truncated) {
            const btn = document.createElement('button');
            btn.className = 'conv-bubble-toggle';
            btn.textContent = '▼ Show more';
            btn.setAttribute('onclick', `toggleResultBubble(this, '${id}')`);
            bubble.appendChild(btn);
        }
    }

    // Update query evolution
    markQueryMemorized(ev.search_index);

    // Refresh GAM tree
    refreshGAMTree();

    scrollConv();
}

function addConvNudge(content) {
    const timeline = document.getElementById('conversation-timeline');
    const div = document.createElement('div');
    div.className = 'conv-nudge';
    div.textContent = '⚡ ' + content;
    timeline.appendChild(div);
    scrollConv();
}

function addConvFinalAnswer(answer) {
    const timeline = document.getElementById('conversation-timeline');
    const div = document.createElement('div');
    div.className = 'conv-final-answer';
    div.innerHTML = `
        <div class="label">✅ Final Answer</div>
        <div class="body">${renderMarkdown(answer || '')}</div>
    `;
    timeline.appendChild(div);

    div.querySelectorAll('pre code').forEach(block => {
        if (!block.className || !block.className.includes('language-')) block.classList.add('language-plaintext');
        hljs.highlightElement(block);
    });

    scrollConv();
}

function addConvError(msg) {
    const timeline = document.getElementById('conversation-timeline');
    const div = document.createElement('div');
    div.className = 'conv-nudge';
    div.style.borderColor = 'var(--accent-danger)';
    div.style.background = 'rgba(220,38,38,0.08)';
    div.style.color = 'var(--accent-danger)';
    div.textContent = '❌ Error: ' + msg;
    timeline.appendChild(div);
    scrollConv();
}

function toggleBubble(btn, fullContent) {
    const body = btn.previousElementSibling;
    if (body.classList.contains('expanded')) {
        body.classList.remove('expanded');
        body.textContent = fullContent.slice(0, 500) + '...';
        btn.textContent = '▼ Show more';
    } else {
        body.classList.add('expanded');
        body.textContent = fullContent;
        btn.textContent = '▲ Show less';
    }
}

function toggleResultBubble(btn, id) {
    const row = btn.closest('.conv-row');
    const body = btn.previousElementSibling;
    const full = row ? row.dataset.fullContent : '';
    if (body.classList.contains('expanded')) {
        body.classList.remove('expanded');
        body.textContent = (full || '').slice(0, 600) + '...';
        btn.textContent = '▼ Show more';
    } else {
        body.classList.add('expanded');
        body.textContent = full || '';
        btn.textContent = '▲ Show less';
    }
}

function scrollConv() {
    const container = document.getElementById('conversation-container');
    if (container) container.scrollTop = container.scrollHeight;
}

// ══════════════════════════════════════════════════════════════════════════
// Query Evolution
// ══════════════════════════════════════════════════════════════════════════
function clearQueryEvolution() {
    document.getElementById('query-evolution-list').innerHTML =
        '<div class="empty-state"><span class="icon">🔍</span><p>Agent queries will appear here</p></div>';
}

function addQueryEvolution(query, index) {
    const list = document.getElementById('query-evolution-list');
    const empty = list.querySelector('.empty-state');
    if (empty) empty.remove();

    const item = document.createElement('div');
    item.className = 'query-item';
    item.id = `query-item-${index}`;
    item.innerHTML = `
        <div class="query-index">${index}</div>
        <div class="query-text">${esc(query)}</div>
    `;
    list.appendChild(item);
    list.scrollTop = list.scrollHeight;
}

function markQueryMemorized(searchIndex) {
    const item = document.getElementById(`query-item-${searchIndex}`);
    if (!item) return;
    item.classList.add('memorized');
    if (!item.querySelector('.query-badge')) {
        const badge = document.createElement('span');
        badge.className = 'query-badge';
        badge.textContent = 'MEMO';
        item.appendChild(badge);
    }
}

// ══════════════════════════════════════════════════════════════════════════
// GAM Tree
// ══════════════════════════════════════════════════════════════════════════
async function refreshGAMTree() {
    if (!gamDir) return;
    try {
        const res = await fetch(`/api/browse?path=${encodeURIComponent(gamDir)}`);
        const data = await res.json();
        if (data.error) return;
        renderGAMTree(data.tree, gamDir);
    } catch (_) {}
}

function renderGAMTree(node, basePath, container = null, relPath = '', isRoot = true) {
    if (!container) {
        container = document.getElementById('gam-tree-view');
        container.innerHTML = '';
    }
    const nodePath = isRoot ? '' : (relPath ? `${relPath}/${node.name}` : node.name);
    const nodeDiv = document.createElement('div');
    nodeDiv.className = 'tree-node';

    const itemDiv = document.createElement('div');
    itemDiv.className = 'tree-item';
    const hasChildren = node.is_dir && node.children && node.children.length > 0;
    const shouldExpand = isRoot;

    itemDiv.innerHTML = `
        <span class="toggle">${hasChildren ? (shouldExpand ? '▼' : '▶') : ''}</span>
        <span class="icon">${node.is_dir ? '📁' : '📄'}</span>
        <span class="name">${node.name}</span>
    `;

    itemDiv.addEventListener('click', e => {
        e.stopPropagation();
        if (hasChildren) {
            const cc = nodeDiv.querySelector('.tree-children');
            const tg = itemDiv.querySelector('.toggle');
            if (cc) {
                const collapsed = cc.classList.toggle('collapsed');
                tg.textContent = collapsed ? '▶' : '▼';
            }
        }
    });

    nodeDiv.appendChild(itemDiv);
    if (hasChildren) {
        const childContainer = document.createElement('div');
        childContainer.className = 'tree-children' + (shouldExpand ? '' : ' collapsed');
        node.children.forEach(child => renderGAMTree(child, basePath, childContainer, nodePath, false));
        nodeDiv.appendChild(childContainer);
    }
    container.appendChild(nodeDiv);
}

// ══════════════════════════════════════════════════════════════════════════
// Right Panel Chat
// ══════════════════════════════════════════════════════════════════════════
function askSuggestion(q) {
    document.getElementById('chat-input').value = q;
    sendChatMessage();
}

function handleChatKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChatMessage(); }
}

function autoResizeInput(ta) {
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 60) + 'px';
}

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send-btn');
    const question = input.value.trim();
    if (!question || isChatProcessing) return;
    if (!gamDir) {
        addChatMessage('assistant', '⚠️ Please run the agent first to build an GAM.');
        return;
    }

    input.value = '';
    input.style.height = 'auto';
    addChatMessage('user', question);

    isChatProcessing = true;
    sendBtn.disabled = true;
    setChatStatus('thinking', 'Researching...');
    addChatTypingIndicator();

    try {
        const fd = new FormData();
        fd.append('question', question);
        fd.append('gam_path', gamDir);

        const response = await fetch('/api/long_horizontal/chat', { method: 'POST', body: fd });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let finalResult = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const ev = JSON.parse(line.slice(6));
                        if (ev.type === 'complete') finalResult = ev;
                        else if (ev.type === 'error') finalResult = { error: ev.error };
                        else addChatAgentAction(ev);
                    } catch (_) {}
                }
            }
        }

        removeChatTypingIndicator();
        if (finalResult && finalResult.answer) {
            addChatMessage('assistant', finalResult.answer, finalResult.sources);
        } else if (finalResult && finalResult.error) {
            addChatMessage('assistant', '❌ Error: ' + finalResult.error);
        } else {
            addChatMessage('assistant', '❌ No response received.');
        }
    } catch (err) {
        removeChatTypingIndicator();
        addChatMessage('assistant', '❌ Error: ' + err.message);
    } finally {
        isChatProcessing = false;
        sendBtn.disabled = false;
        setChatStatus('', 'Ready');
    }
}

function addChatMessage(role, content, sources = null) {
    const container = document.getElementById('chat-messages');
    const empty = document.getElementById('chat-empty-state');
    if (empty) empty.style.display = 'none';

    const div = document.createElement('div');
    div.className = `chat-message ${role}`;
    const avatar = role === 'user' ? '👤' : '🤖';

    let sourcesHtml = '';
    if (sources && sources.length) {
        sourcesHtml = `<div class="message-sources">
            <div class="message-sources-title">📎 Sources:</div>
            ${sources.map(s => {
                const d = String(s).split('/').pop() || s;
                return `<span class="message-source-item">${esc(d)}</span>`;
            }).join('')}
        </div>`;
    }

    const contentHtml = role === 'assistant' ? renderMarkdown(content) : `<p>${esc(content)}</p>`;

    div.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">${contentHtml}${sourcesHtml}</div>
    `;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;

    div.querySelectorAll('pre code').forEach(block => {
        if (!block.className || !block.className.includes('language-')) block.classList.add('language-plaintext');
        hljs.highlightElement(block);
    });
}

function addChatTypingIndicator() {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'chat-message assistant';
    div.id = 'typing-indicator';
    div.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="typing-indicator"><span></span><span></span><span></span></div>
            <div class="agent-actions" id="agent-actions-log"></div>
        </div>
    `;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function removeChatTypingIndicator() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

function addChatAgentAction(ev) {
    const log = document.getElementById('agent-actions-log');
    if (!log) return;
    const item = document.createElement('div');
    item.className = 'agent-action-item';
    if (ev.type === 'tool_call') {
        const icon = ev.tool === 'ls' ? '📂' : ev.tool === 'cat' ? '📄' : ev.tool === 'grep' ? '🔍' : '🔧';
        item.innerHTML = `<span>${icon}</span><span>${esc(ev.display || ev.tool)}</span>`;
    } else if (ev.type === 'thinking') {
        item.innerHTML = `<span>💭</span><span>${esc(ev.message || 'Thinking...')}</span>`;
    } else if (ev.type === 'round') {
        item.innerHTML = `<span>🔄</span><span>Round ${ev.round}/${ev.max_rounds}</span>`;
    }
    log.appendChild(item);
    log.scrollTop = log.scrollHeight;
}

// ══════════════════════════════════════════════════════════════════════════
// Status Helpers
// ══════════════════════════════════════════════════════════════════════════
function setRunStatus(text, cls) {
    const el = document.getElementById('run-status');
    el.textContent = text;
    el.className = 'badge ' + (cls || 'badge-info');
}

function setConvStatus(dotClass, text) {
    const dot = document.getElementById('conv-status-dot');
    const txt = document.getElementById('conv-status-text');
    dot.className = 'dot' + (dotClass ? ' ' + dotClass : '');
    txt.textContent = text;
}

function setChatStatus(dotClass, text) {
    const dot = document.getElementById('chat-status-dot');
    const txt = document.getElementById('chat-status-text');
    dot.className = 'dot' + (dotClass ? ' ' + dotClass : '');
    txt.textContent = text;
}

// ══════════════════════════════════════════════════════════════════════════
// Reset
// ══════════════════════════════════════════════════════════════════════════
function resetAll() {
    if (!confirm('Start a new session?')) return;
    sessionId = null;
    gamDir = null;
    selectedFiles = [];
    searchQueries = [];
    isRunning = false;
    isChatProcessing = false;

    renderDocList();
    document.getElementById('question-input').value = '';
    clearConversation();
    document.getElementById('conv-empty-state').style.display = 'flex';
    clearQueryEvolution();
    document.getElementById('gam-tree-view').innerHTML = '<div class="empty-state"><span class="icon">📂</span><p>GAM tree after memorize</p></div>';

    const chatContainer = document.getElementById('chat-messages');
    chatContainer.innerHTML = '';
    const chatEmpty = document.getElementById('chat-empty-state');
    if (chatEmpty) { chatContainer.appendChild(chatEmpty); chatEmpty.style.display = 'flex'; }

    setRunStatus('Ready', 'badge-info');
    setConvStatus('', 'Idle');
    setChatStatus('', 'Ready');
}

// ══════════════════════════════════════════════════════════════════════════
// Utilities
// ══════════════════════════════════════════════════════════════════════════
function esc(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
}

function escAttr(text) {
    return (text || '').replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/\n/g, '\\n');
}

function formatSize(bytes) {
    if (!bytes) return '0 B';
    const u = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return (bytes / Math.pow(1024, i)).toFixed(i > 0 ? 1 : 0) + ' ' + u[i];
}

function renderMarkdown(text) {
    try { return marked.parse(text || ''); }
    catch (_) { return `<p>${esc(text)}</p>`; }
}
