// ========== State ==========
let currentGAMDir = null;
let autoRefreshInterval = null;
let isProcessing = false;
let selectedVideoFile = null;
let selectedSrtFile = null;
let lastTreeData = null;

let treeState = {
    expandedPaths: new Set(),
    selectedPath: null,
};

// ========== Upload Handlers ==========
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const srtUploadZone = document.getElementById('srt-upload-zone');
const srtInput = document.getElementById('srt-input');

// Video upload
uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleVideoFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleVideoFile(e.target.files[0]);
    }
});

function handleVideoFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['mp4', 'avi', 'mkv', 'mov', 'webm'].includes(ext)) {
        alert('Unsupported video format. Please use MP4, AVI, MKV, MOV, or WebM.');
        return;
    }
    selectedVideoFile = file;
    document.getElementById('file-info').style.display = 'block';
    document.getElementById('file-name').textContent = `${file.name} (${formatFileSize(file.size)})`;
}

// SRT upload
srtUploadZone.addEventListener('click', () => srtInput.click());

srtUploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    srtUploadZone.classList.add('dragover');
});

srtUploadZone.addEventListener('dragleave', () => {
    srtUploadZone.classList.remove('dragover');
});

srtUploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    srtUploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleSrtFile(e.dataTransfer.files[0]);
    }
});

srtInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleSrtFile(e.target.files[0]);
    }
});

function handleSrtFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['srt', 'vtt'].includes(ext)) {
        alert('Unsupported subtitle format. Please use SRT or VTT.');
        return;
    }
    selectedSrtFile = file;
    document.getElementById('srt-info').style.display = 'block';
    document.getElementById('srt-name').textContent = file.name;
}

// ========== Video Processing ==========
async function processVideo() {
    if (!selectedVideoFile) {
        alert('Please select a video file first.');
        return;
    }

    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressStatus = document.getElementById('progress-status');
    const logOutput = document.getElementById('log-output');
    const processBtn = document.getElementById('process-btn');

    progressContainer.classList.add('active');
    processBtn.disabled = true;
    logOutput.innerHTML = '';
    progressBar.style.width = '5%';
    progressBar.style.background = 'var(--gradient-primary)';
    progressStatus.textContent = 'Uploading video...';
    addLog('🚀 Starting Video GAM pipeline...', 'info');
    addLog(`🎥 Uploading: ${selectedVideoFile.name} (${formatFileSize(selectedVideoFile.size)})`, 'info');

    try {
        const formData = new FormData();
        formData.append('video', selectedVideoFile);
        if (selectedSrtFile) {
            formData.append('subtitle', selectedSrtFile);
            addLog(`📝 Subtitle: ${selectedSrtFile.name}`, 'info');
        }

        progressBar.style.width = '10%';
        progressStatus.textContent = 'Starting pipeline...';

        const startResponse = await fetch('/api/video/pipeline_start', {
            method: 'POST',
            body: formData
        });

        const startResult = await startResponse.json();

        if (!startResult.success) {
            throw new Error(startResult.error || 'Failed to start video pipeline');
        }

        const taskId = startResult.task_id;
        currentGAMDir = startResult.gam_dir;

        // Show output path
        document.getElementById('output-paths').style.display = 'block';
        document.getElementById('gam-path-display').textContent = startResult.gam_dir;

        addLog(`📂 Output directory: ${startResult.gam_dir}`, 'info');

        // Start auto-refresh
        startAutoRefresh();

        progressBar.style.width = '20%';
        progressStatus.textContent = 'Processing video...';

        // Poll for task completion
        let pollCount = 0;
        const maxPolls = 1800; // Max 30 minutes for video

        while (pollCount < maxPolls) {
            await new Promise(resolve => setTimeout(resolve, 2000));
            pollCount++;

            const statusResponse = await fetch(`/api/video/pipeline_status?task_id=${taskId}`);
            const statusResult = await statusResponse.json();

            if (statusResult.status === 'completed') {
                stopAutoRefresh();
                progressBar.style.width = '100%';
                progressStatus.textContent = 'Complete!';

                addLog('', '');
                addLog(`✅ Video GAM built successfully!`, 'success');
                if (statusResult.segment_count) {
                    addLog(`   Segments: ${statusResult.segment_count}`, 'success');
                }
                addLog(`   Output: ${currentGAMDir}`, 'info');

                document.getElementById('process-status').textContent = 'Done';
                document.getElementById('process-status').className = 'badge badge-success';

                refreshTree();
                break;

            } else if (statusResult.status === 'error') {
                stopAutoRefresh();
                throw new Error(statusResult.error || 'Video pipeline failed');

            } else if (statusResult.status === 'running') {
                const stage = statusResult.stage || 'processing';
                if (stage === 'uploading') {
                    progressBar.style.width = '15%';
                    progressStatus.textContent = 'Uploading video...';
                } else if (stage === 'probing') {
                    progressBar.style.width = '30%';
                    progressStatus.textContent = 'Analyzing video content...';
                } else if (stage === 'segmenting') {
                    progressBar.style.width = '50%';
                    progressStatus.textContent = 'Segmenting video...';
                } else if (stage === 'describing') {
                    progressBar.style.width = '70%';
                    progressStatus.textContent = 'Generating descriptions...';
                } else if (stage === 'organizing') {
                    progressBar.style.width = '85%';
                    progressStatus.textContent = 'Organizing GAM structure...';
                }

                if (statusResult.message) {
                    addLog(`   ${statusResult.message}`, 'info');
                }
            }
        }

        if (pollCount >= maxPolls) {
            stopAutoRefresh();
            throw new Error('Video pipeline timed out');
        }

    } catch (err) {
        stopAutoRefresh();
        progressBar.style.width = '100%';
        progressBar.style.background = 'var(--accent-danger)';
        progressStatus.textContent = 'Error!';
        addLog(`❌ Error: ${err.message}`, 'error');

        document.getElementById('process-status').textContent = 'Error';
        document.getElementById('process-status').className = 'badge badge-warning';
    } finally {
        processBtn.disabled = false;
    }
}

// ========== Auto-Refresh ==========
function startAutoRefresh() {
    if (autoRefreshInterval) return;
    isProcessing = true;
    document.getElementById('auto-refresh-indicator').style.display = 'inline';

    autoRefreshInterval = setInterval(() => {
        if (currentGAMDir) {
            incrementalRefreshTree();
        }
    }, 3000);
}

function stopAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
    }
    isProcessing = false;
    document.getElementById('auto-refresh-indicator').style.display = 'none';
    lastTreeData = null;
}

async function incrementalRefreshTree() {
    if (!currentGAMDir) return;

    try {
        const response = await fetch(`/api/browse?path=${encodeURIComponent(currentGAMDir)}`);
        const data = await response.json();
        if (data.error) return;

        const newStats = `${data.stats.dirs}-${data.stats.files}`;
        const oldStats = lastTreeData ? `${lastTreeData.stats.dirs}-${lastTreeData.stats.files}` : '';

        if (newStats !== oldStats) {
            lastTreeData = data;
            renderTree(data.tree, currentGAMDir);
            updateStats(data);
        }
    } catch (err) {
        console.error('Incremental refresh failed:', err);
    }
}

// ========== Tree ==========
async function refreshTree(resetState = false) {
    if (!currentGAMDir) {
        document.getElementById('tree-view').innerHTML = `
            <div class="empty-state">
                <span class="icon">📂</span>
                <h3>No output</h3>
                <p>Upload & build video to view GAM structure</p>
            </div>
        `;
        return;
    }

    try {
        const response = await fetch(`/api/browse?path=${encodeURIComponent(currentGAMDir)}`);
        const data = await response.json();

        if (data.error) throw new Error(data.error);

        if (resetState) {
            treeState.expandedPaths.clear();
            treeState.selectedPath = null;
        }

        lastTreeData = data;
        renderTree(data.tree, currentGAMDir);
        updateStats(data);

    } catch (err) {
        console.error('Failed to refresh tree:', err);
        document.getElementById('tree-view').innerHTML = `
            <div class="empty-state">
                <span class="icon">❌</span>
                <h3>Error</h3>
                <p>${err.message}</p>
            </div>
        `;
    }
}

function updateStats(data) {
    document.getElementById('stat-dirs').textContent = data.stats.dirs;
    document.getElementById('stat-files').textContent = data.stats.files;
    // Count video files
    let videoCount = 0;
    function countVideos(node) {
        if (!node.is_dir && node.name && /\.(mp4|avi|mkv|mov|webm)$/i.test(node.name)) {
            videoCount++;
        }
        if (node.children) node.children.forEach(countVideos);
    }
    if (data.tree) countVideos(data.tree);
    document.getElementById('stat-videos').textContent = videoCount;
}

function isVideoFile(name) {
    return /\.(mp4|avi|mkv|mov|webm)$/i.test(name);
}

function getFileIcon(node) {
    if (node.is_dir) return '📁';
    if (isVideoFile(node.name)) return '🎥';
    if (/\.srt$/i.test(node.name)) return '📝';
    if (/\.md$/i.test(node.name)) return '📖';
    if (/\.json$/i.test(node.name)) return '📋';
    return '📄';
}

function getIconClass(node) {
    if (node.is_dir) return 'folder';
    if (isVideoFile(node.name)) return 'video';
    return 'file';
}

function renderTree(node, basePath, container = null, relativePath = '', isRoot = true) {
    if (!container) {
        container = document.getElementById('tree-view');
        container.innerHTML = '';
    }

    const nodePath = isRoot ? '' : (relativePath ? `${relativePath}/${node.name}` : node.name);

    const nodeDiv = document.createElement('div');
    nodeDiv.className = 'tree-node';

    const itemDiv = document.createElement('div');
    itemDiv.className = 'tree-item';

    const hasChildren = node.is_dir && node.children && node.children.length > 0;
    const shouldExpand = isRoot || treeState.expandedPaths.has(nodePath);
    const isSelected = treeState.selectedPath === nodePath;

    const icon = getFileIcon(node);
    const iconClass = getIconClass(node);

    itemDiv.innerHTML = `
        <span class="toggle">${hasChildren ? (shouldExpand ? '▼' : '▶') : ''}</span>
        <span class="icon ${iconClass}">${icon}</span>
        <span class="name">${node.name}</span>
    `;

    if (isSelected) itemDiv.classList.add('selected');

    itemDiv.addEventListener('click', (e) => {
        e.stopPropagation();

        if (hasChildren) {
            const childContainer = nodeDiv.querySelector('.tree-children');
            const toggle = itemDiv.querySelector('.toggle');
            if (childContainer) {
                const isCollapsed = childContainer.classList.toggle('collapsed');
                toggle.textContent = isCollapsed ? '▶' : '▼';
                if (isCollapsed) {
                    treeState.expandedPaths.delete(nodePath);
                } else {
                    treeState.expandedPaths.add(nodePath);
                }
            }
        }

        document.querySelectorAll('.tree-item').forEach(i => i.classList.remove('selected'));
        itemDiv.classList.add('selected');
        treeState.selectedPath = nodePath;

        const fullPath = nodePath ? `${basePath}/${nodePath}` : basePath;

        if (!node.is_dir && isVideoFile(node.name)) {
            playVideo(fullPath, node.name);
        } else {
            showPreviewByFullPath(fullPath, node.is_dir);
        }
    });

    nodeDiv.appendChild(itemDiv);

    if (hasChildren) {
        const childContainer = document.createElement('div');
        childContainer.className = 'tree-children' + (shouldExpand ? '' : ' collapsed');
        node.children.forEach(child => {
            renderTree(child, basePath, childContainer, nodePath, false);
        });
        nodeDiv.appendChild(childContainer);
    }

    container.appendChild(nodeDiv);
}

// ========== Video Player ==========
// ========== Unified Display Helpers ==========
function showVideoView(fullPath, name) {
    const displayName = name || fullPath.split('/').pop();

    // Update bar
    document.getElementById('display-bar-icon').textContent = '🎥';
    document.getElementById('display-bar-name').textContent = displayName;

    // Hide other views, show video
    document.getElementById('display-empty').style.display = 'none';
    document.getElementById('display-content').style.display = 'none';
    const videoView = document.getElementById('display-video');
    videoView.style.display = 'flex';

    const player = document.getElementById('video-player');
    const videoUrl = `/api/video/serve?path=${encodeURIComponent(fullPath)}`;
    player.src = videoUrl;
    player.load();
}

function showContentView(html) {
    // Hide other views, show content
    document.getElementById('display-empty').style.display = 'none';
    document.getElementById('display-video').style.display = 'none';

    // Pause video if playing
    const player = document.getElementById('video-player');
    player.pause();

    const contentEl = document.getElementById('display-content');
    contentEl.style.display = 'block';
    contentEl.innerHTML = html;

    contentEl.querySelectorAll('pre code').forEach((block) => {
        if (!block.className || !block.className.includes('language-')) {
            block.classList.add('language-plaintext');
        }
        hljs.highlightElement(block);
    });
}

function showEmptyView() {
    const player = document.getElementById('video-player');
    player.pause();
    player.src = '';

    document.getElementById('display-video').style.display = 'none';
    document.getElementById('display-content').style.display = 'none';
    document.getElementById('display-empty').style.display = 'flex';

    document.getElementById('display-bar-icon').textContent = '📄';
    document.getElementById('display-bar-name').textContent = 'Select a file to preview';
}

// ========== playVideo (called from tree click) ==========
function playVideo(fullPath, name) {
    syncTreeToPath(fullPath);
    showVideoView(fullPath, name);
}

// ========== Content Preview (called from tree click) ==========
async function showPreviewByFullPath(fullPath, isDir) {
    try {
        syncTreeToPath(fullPath);

        const response = await fetch(`/api/file_content?path=${encodeURIComponent(fullPath)}`);
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        const pathParts = fullPath.split('/');
        const displayName = pathParts[pathParts.length - 1] || '/';

        // Update bar
        document.getElementById('display-bar-icon').textContent = data.is_dir ? '📁' : '📄';
        document.getElementById('display-bar-name').textContent = displayName;

        let html = '';

        if (data.is_dir) {
            html = '<div style="font-family: var(--font-mono, monospace); font-size: 12px;">';
            html += '<div style="margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border-color, #e0e3e8); color: var(--text-muted, #9a9ab0);">';
            html += `📁 <strong>${escapeHtml(data.name)}</strong>`;
            const dirs = (data.children || []).filter(c => c.is_dir).length;
            const files = (data.children || []).filter(c => !c.is_dir).length;
            html += ` &nbsp;—&nbsp; ${dirs} folder${dirs !== 1 ? 's' : ''}, ${files} file${files !== 1 ? 's' : ''}`;
            html += '</div>';

            if (!data.children || data.children.length === 0) {
                html += '<p style="color: var(--text-muted, #9a9ab0);">Empty directory</p>';
            } else {
                const sorted = [...(data.children || [])].sort((a, b) => {
                    if (a.is_dir !== b.is_dir) return a.is_dir ? -1 : 1;
                    return a.name.localeCompare(b.name);
                });
                html += '<table style="width: 100%; border-collapse: collapse;">';
                sorted.forEach(child => {
                    const icon = child.is_dir ? '📁' : (isVideoFile(child.name) ? '🎥' : '📄');
                    const nameStyle = child.is_dir
                        ? 'color: var(--accent-primary); cursor: pointer;'
                        : (isVideoFile(child.name)
                            ? 'color: #e44d8a; cursor: pointer;'
                            : 'color: var(--text-primary, #1a1a2e); cursor: pointer;');
                    const sizeStr = child.is_dir ? '—' : formatFileSize(child.size || 0);
                    const childFullPath = data.full_path ? `${data.full_path}/${child.name}` : child.name;

                    html += `<tr style="border-bottom: 1px solid var(--border-color, #e0e3e8);">`;
                    html += `<td style="padding: 4px 8px; white-space: nowrap;">${icon}</td>`;

                    if (!child.is_dir && isVideoFile(child.name)) {
                        html += `<td style="padding: 4px 8px; ${nameStyle}" onclick="playVideo('${escapeHtml(childFullPath).replace(/'/g, "\\'")}', '${escapeHtml(child.name).replace(/'/g, "\\'")}')">${escapeHtml(child.name)}</td>`;
                    } else {
                        html += `<td style="padding: 4px 8px; ${nameStyle}" onclick="showPreviewByFullPath('${escapeHtml(childFullPath).replace(/'/g, "\\'")}', ${child.is_dir})">${escapeHtml(child.name)}</td>`;
                    }

                    html += `<td style="padding: 4px 8px; text-align: right; color: var(--text-muted, #9a9ab0); white-space: nowrap;">${sizeStr}</td>`;
                    html += `</tr>`;
                });
                html += '</table>';
            }
            html += '</div>';
        } else {
            const content = data.content || '';
            if (data.name && (data.name.endsWith('.md') || data.name.endsWith('.markdown'))) {
                html = marked.parse(content);
            } else if (data.name && data.name.endsWith('.json')) {
                try {
                    const formatted = JSON.stringify(JSON.parse(content), null, 2);
                    html = `<pre><code class="language-json">${escapeHtml(formatted)}</code></pre>`;
                } catch {
                    html = `<pre><code>${escapeHtml(content)}</code></pre>`;
                }
            } else {
                html = `<pre><code>${escapeHtml(content)}</code></pre>`;
            }
        }

        showContentView(html);
    } catch (err) {
        console.error('Failed to load preview:', err);
        showContentView(`<p style="color: var(--accent-danger);">Error: ${err.message}</p>`);
    }
}

function syncTreeToPath(fullPath) {
    if (!currentGAMDir) return;

    let relativePath = '';
    if (fullPath.startsWith(currentGAMDir + '/')) {
        relativePath = fullPath.slice(currentGAMDir.length + 1);
    } else if (fullPath === currentGAMDir) {
        relativePath = '';
    } else {
        return;
    }

    if (relativePath) {
        const parts = relativePath.split('/');
        let ancestor = '';
        for (let i = 0; i < parts.length - 1; i++) {
            ancestor = ancestor ? `${ancestor}/${parts[i]}` : parts[i];
            treeState.expandedPaths.add(ancestor);
        }
    }

    treeState.selectedPath = relativePath;

    if (lastTreeData) {
        renderTree(lastTreeData.tree, currentGAMDir);
    }

    setTimeout(() => {
        const selectedItem = document.querySelector('.tree-item.selected');
        if (selectedItem) {
            selectedItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }, 50);
}

function clearPreview() {
    showEmptyView();
}

// ========== Load Modal ==========
let browserCurrentPath = '/share/project/chaofan/code/memory/gam/examples/output';

function showLoadModal() {
    document.getElementById('load-modal').classList.add('show');
    browserNavigate(browserCurrentPath);
}

function hideLoadModal() {
    document.getElementById('load-modal').classList.remove('show');
}

document.getElementById('load-modal').addEventListener('click', (e) => {
    if (e.target.id === 'load-modal') hideLoadModal();
});

async function browserNavigate(path) {
    browserCurrentPath = path;
    document.getElementById('browser-current-path').textContent = path;
    document.getElementById('folder-list').innerHTML = '<div class="folder-loading">Loading...</div>';

    try {
        const response = await fetch(`/api/list_dir?path=${encodeURIComponent(path)}`);
        const data = await response.json();

        if (data.error) {
            document.getElementById('folder-list').innerHTML = `
                <div class="folder-empty">
                    <div class="icon">❌</div>
                    <div>${data.error}</div>
                </div>
            `;
            return;
        }

        const list = document.getElementById('folder-list');
        list.innerHTML = '';

        if (data.items.length === 0) {
            list.innerHTML = `
                <div class="folder-empty">
                    <div class="icon">📭</div>
                    <div>Empty folder</div>
                </div>
            `;
            return;
        }

        data.items.sort((a, b) => {
            if (a.is_dir && !b.is_dir) return -1;
            if (!a.is_dir && b.is_dir) return 1;
            return a.name.localeCompare(b.name);
        });

        data.items.forEach(item => {
            const div = document.createElement('div');
            div.className = 'folder-item';
            const fullPath = path.endsWith('/') ? path + item.name : path + '/' + item.name;

            if (item.is_dir) {
                div.innerHTML = `
                    <span class="folder-icon">📁</span>
                    <span class="folder-name">${item.name}</span>
                    <div class="folder-actions">
                        <button class="folder-action-btn gam-btn" onclick="event.stopPropagation(); setGamLoadPath('${fullPath}')" title="Select as GAM Dir">🗂️</button>
                    </div>
                `;
                div.onclick = () => browserNavigate(fullPath);
            } else {
                div.innerHTML = `
                    <span class="folder-icon">📄</span>
                    <span class="folder-name" style="color: var(--text-muted);">${item.name}</span>
                `;
            }

            list.appendChild(div);
        });

    } catch (err) {
        document.getElementById('folder-list').innerHTML = `
            <div class="folder-empty">
                <div class="icon">❌</div>
                <div>Error: ${err.message}</div>
            </div>
        `;
    }
}

function browserGoUp() {
    const parts = browserCurrentPath.split('/').filter(p => p);
    if (parts.length > 1) {
        parts.pop();
        browserNavigate('/' + parts.join('/'));
    } else if (parts.length === 1) {
        browserNavigate('/');
    }
}

function browserRefresh() {
    browserNavigate(browserCurrentPath);
}

function setGamLoadPath(path) {
    document.getElementById('load-gam-path').value = path;
}

async function loadDirectory() {
    const gamPath = document.getElementById('load-gam-path').value.trim();

    if (!gamPath) {
        alert('Please enter a directory path');
        return;
    }

    try {
        const response = await fetch(`/api/browse?path=${encodeURIComponent(gamPath)}`);
        const data = await response.json();

        if (data.error) {
            alert(`Could not load directory: ${data.error}`);
            return;
        }

        currentGAMDir = gamPath;
        hideLoadModal();

        document.getElementById('output-paths').style.display = 'block';
        document.getElementById('gam-path-display').textContent = currentGAMDir;

        document.getElementById('process-status').textContent = 'Loaded';
        document.getElementById('process-status').className = 'badge badge-success';

        const progressContainer = document.getElementById('progress-container');
        const logOutput = document.getElementById('log-output');
        progressContainer.classList.add('active');
        logOutput.innerHTML = '';
        document.getElementById('progress-bar').style.width = '100%';
        document.getElementById('progress-status').textContent = 'Loaded successfully';

        addLog(`📂 Loaded Video GAM: ${gamPath}`, 'success');

        refreshTree();
    } catch (err) {
        alert(`Error loading directory: ${err.message}`);
    }
}

// ========== Reset ==========
function resetAll() {
    if (confirm('Start a new session? This will clear current state.')) {
        stopAutoRefresh();

        currentGAMDir = null;
        selectedVideoFile = null;
        selectedSrtFile = null;
        lastTreeData = null;

        treeState.expandedPaths.clear();
        treeState.selectedPath = null;

        // Reset upload UI
        document.getElementById('file-info').style.display = 'none';
        document.getElementById('srt-info').style.display = 'none';
        document.getElementById('file-input').value = '';
        document.getElementById('srt-input').value = '';

        // Reset output
        document.getElementById('output-paths').style.display = 'none';
        document.getElementById('progress-container').classList.remove('active');

        // Reset tree
        document.getElementById('tree-view').innerHTML = `
            <div class="empty-state">
                <span class="icon">📂</span>
                <h3>No output</h3>
                <p>Upload & build video to view GAM structure</p>
            </div>
        `;

        // Reset display area
        showEmptyView();

        // Reset stats
        document.getElementById('process-status').textContent = 'Ready';
        document.getElementById('process-status').className = 'badge badge-info';
        document.getElementById('stat-dirs').textContent = '0';
        document.getElementById('stat-files').textContent = '0';
        document.getElementById('stat-videos').textContent = '0';
    }
}

// ========== Chat ==========
let isChatProcessing = false;

function updateChatStatus(status, text) {
    const dot = document.getElementById('chat-status-dot');
    const statusText = document.getElementById('chat-status-text');

    dot.className = 'dot';
    if (status === 'thinking') dot.classList.add('thinking');
    else if (status === 'disconnected') dot.classList.add('disconnected');

    statusText.textContent = text;
}

function askSuggestion(question) {
    document.getElementById('chat-input').value = question;
    sendMessage();
}

function handleChatKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function autoResizeInput(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 60) + 'px';
}

function addMessage(role, content, sources = null) {
    const messagesContainer = document.getElementById('chat-messages');
    const emptyState = document.getElementById('chat-empty-state');

    if (emptyState) emptyState.style.display = 'none';

    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}`;

    const avatar = role === 'user' ? '👤' : '🤖';

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = `
            <div class="message-sources">
                <div class="message-sources-title">📎 Sources:</div>
                ${sources.map(s => {
                    const sourceValue = String(s);
                    const displayName = sourceValue.split('/').pop() || sourceValue;
                    return `<span class="message-source-item" data-source="${encodeURIComponent(sourceValue)}" title="${escapeHtml(sourceValue)}">${escapeHtml(displayName)}</span>`;
                }).join('')}
            </div>
        `;
    }

    let contentHtml = role === 'assistant' ? marked.parse(content) : `<p>${escapeHtml(content)}</p>`;

    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            ${contentHtml}
            ${sourcesHtml}
        </div>
    `;

    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    // Bind source click handlers
    messageDiv.querySelectorAll('.message-source-item').forEach((item) => {
        item.addEventListener('click', () => {
            const sourceValue = decodeURIComponent(item.dataset.source || '');
            navigateToSource(sourceValue);
        });
    });

    messageDiv.querySelectorAll('pre code').forEach(block => {
        if (!block.className || !block.className.includes('language-')) {
            block.classList.add('language-plaintext');
        }
        hljs.highlightElement(block);
    });

    return messageDiv;
}

function addTypingIndicator() {
    const messagesContainer = document.getElementById('chat-messages');
    const emptyState = document.getElementById('chat-empty-state');
    if (emptyState) emptyState.style.display = 'none';

    const typingDiv = document.createElement('div');
    typingDiv.className = 'chat-message assistant';
    typingDiv.id = 'typing-indicator';
    typingDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
            <div class="agent-actions" id="agent-actions-log"></div>
        </div>
    `;

    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function removeTypingIndicator() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

function addAgentAction(eventData) {
    const actionsLog = document.getElementById('agent-actions-log');
    if (!actionsLog) return;

    const item = document.createElement('div');
    item.className = 'agent-action-item';

    if (eventData.type === 'tool_call') {
        const icon = eventData.tool === 'ls' ? '📂' :
                     eventData.tool === 'cat' ? '📄' :
                     eventData.tool === 'grep' ? '🔍' :
                     eventData.tool === 'inspect_video' ? '🎥' : '🔧';
        item.innerHTML = `<span class="agent-action-icon">${icon}</span><span class="agent-action-cmd">${escapeHtml(eventData.display || eventData.tool)}</span>`;
    } else if (eventData.type === 'thinking') {
        item.classList.add('action-thinking');
        item.innerHTML = `<span class="agent-action-icon">💭</span><span class="agent-action-cmd">${escapeHtml(eventData.message || 'Thinking...')}</span>`;
    } else if (eventData.type === 'round') {
        item.classList.add('action-round');
        item.innerHTML = `<span class="agent-action-icon">🔄</span><span class="agent-action-cmd">Round ${eventData.round}/${eventData.max_rounds}</span>`;
    } else {
        return;
    }

    actionsLog.appendChild(item);
    actionsLog.scrollTop = actionsLog.scrollHeight;

    const messagesContainer = document.getElementById('chat-messages');
    if (messagesContainer) messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send-btn');
    const question = input.value.trim();

    if (!question || isChatProcessing) return;

    if (!currentGAMDir) {
        addMessage('assistant', '⚠️ Please build or load a Video GAM first before asking questions.');
        return;
    }

    input.value = '';
    input.style.height = 'auto';

    addMessage('user', question);

    isChatProcessing = true;
    sendBtn.disabled = true;
    updateChatStatus('thinking', 'Researching...');
    addTypingIndicator();

    try {
        const formData = new FormData();
        formData.append('question', question);
        formData.append('gam_path', currentGAMDir);

        const response = await fetch('/api/video/research_stream', {
            method: 'POST',
            body: formData
        });

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
                        const eventData = JSON.parse(line.slice(6));
                        if (eventData.type === 'complete') {
                            finalResult = eventData;
                        } else if (eventData.type === 'error') {
                            finalResult = { error: eventData.error };
                        } else {
                            addAgentAction(eventData);
                        }
                    } catch (parseErr) {
                        // Ignore
                    }
                }
            }
        }

        removeTypingIndicator();

        if (finalResult && finalResult.answer) {
            addMessage('assistant', finalResult.answer, finalResult.sources);
        } else if (finalResult && finalResult.error) {
            addMessage('assistant', `❌ Error: ${finalResult.error}`);
        } else {
            addMessage('assistant', '❌ Error: No response received from agent');
        }

    } catch (err) {
        removeTypingIndicator();
        addMessage('assistant', `❌ Error: ${err.message}`);
    } finally {
        isChatProcessing = false;
        sendBtn.disabled = false;
        updateChatStatus('ready', 'Ready');
    }
}

async function navigateToSource(sourceName) {
    if (!currentGAMDir) return;

    try {
        const response = await fetch(`/api/find_source?gam_dir=${encodeURIComponent(currentGAMDir)}&source_name=${encodeURIComponent(sourceName)}`);
        const data = await response.json();

        if (data.found && data.full_path) {
            if (isVideoFile(data.full_path.split('/').pop())) {
                playVideo(data.full_path, data.full_path.split('/').pop());
            } else {
                showPreviewByFullPath(data.full_path, false);
            }
        } else {
            alert(`Source "${sourceName}" not found.`);
        }
    } catch (err) {
        console.error('Failed to navigate to source:', err);
    }
}

// ========== Utility ==========
function addLog(message, type = '') {
    const logOutput = document.getElementById('log-output');
    const line = document.createElement('div');
    line.className = `log-line ${type}`;
    line.textContent = message;
    logOutput.appendChild(line);
    logOutput.scrollTop = logOutput.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return (bytes / Math.pow(1024, i)).toFixed(i > 0 ? 1 : 0) + ' ' + units[i];
}
