// State
        let currentChunkDir = null;
        let currentChunkBaseDir = null;
        let currentGAMDir = null;
        let currentViewDir = null;  // 'chunk' or 'gam'
        let selectedFiles = [];
        let autoRefreshInterval = null;  // For auto-refresh during build
        let isProcessing = false;  // Track processing state

        // Tab switching
        function switchTab(tab) {
            document.querySelectorAll('.input-tab').forEach(t => t.classList.remove('active'));
            document.querySelector(`.input-tab[data-tab="${tab}"]`).classList.add('active');
            
            document.getElementById('tab-text').style.display = tab === 'text' ? 'block' : 'none';
            document.getElementById('tab-file').style.display = tab === 'file' ? 'block' : 'none';
        }

        // Directory switching
        function switchDirectory(dir) {
            const changed = currentViewDir !== dir;
            currentViewDir = dir;
            
            // Update button states
            document.getElementById('btn-chunk-dir').classList.remove('active', 'chunk-active', 'gam-active');
            document.getElementById('btn-gam-dir').classList.remove('active', 'chunk-active', 'gam-active');
            
            if (dir === 'chunk') {
                document.getElementById('btn-chunk-dir').classList.add('active', 'chunk-active');
            } else {
                document.getElementById('btn-gam-dir').classList.add('active', 'gam-active');
            }
            
            // 切换目录时重置状态
            if (changed) {
                treeState.expandedPaths.clear();
                treeState.selectedPath = null;
                lastTreeData = null;
                clearPreview();
            }
            
            refreshTree();
        }
        
        // Auto-refresh functions for real-time directory updates during build
        let lastTreeData = null;  // 存储上次的树数据，用于增量比较
        
        function startAutoRefresh() {
            if (autoRefreshInterval) return;  // Already running
            
            isProcessing = true;
            document.getElementById('auto-refresh-indicator').style.display = 'inline';
            
            autoRefreshInterval = setInterval(() => {
                if (currentViewDir && (currentChunkDir || currentChunkBaseDir || currentGAMDir)) {
                    incrementalRefreshTree();  // 增量刷新
                }
            }, 3000);  // 每 3 秒刷新一次（降低频率）
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
        
        // 增量刷新：只在有变化时更新，保持用户的展开和选中状态
        async function incrementalRefreshTree() {
            let dirPath = null;
            
            if (currentViewDir === 'chunk' && (currentChunkBaseDir || currentChunkDir)) {
                dirPath = currentChunkBaseDir || currentChunkDir;
            } else if (currentViewDir === 'gam' && currentGAMDir) {
                dirPath = currentGAMDir;
            }
            
            if (!dirPath) return;

            try {
                const endpoint = currentViewDir === 'chunk' ? '/api/browse_chunks' : '/api/browse';
                const response = await fetch(`${endpoint}?path=${encodeURIComponent(dirPath)}`);
                const data = await response.json();

                if (data.error) return;

                // 检查是否有变化（简单比较节点数量）
                const newStats = `${data.stats.dirs}-${data.stats.files}`;
                const oldStats = lastTreeData ? `${lastTreeData.stats.dirs}-${lastTreeData.stats.files}` : '';
                
                if (newStats !== oldStats) {
                    // 有变化，重新渲染但保持状态
                    lastTreeData = data;
                    renderTree(data.tree, dirPath);
                    
                    // 更新统计
                    document.getElementById('stat-dirs').textContent = data.stats.dirs;
                    document.getElementById('stat-files').textContent = data.stats.files;
                }
            } catch (err) {
                console.error('Incremental refresh failed:', err);
            }
        }

        // File upload handlers
        const uploadZone = document.getElementById('upload-zone');
        const fileInput = document.getElementById('file-input');

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
                handleFiles(e.dataTransfer.files);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                handleFiles(e.target.files);
            }
        });

        function handleFiles(fileList) {
            const files = Array.from(fileList || []);
            const validFiles = [];
            const invalidFiles = [];
            files.forEach((file) => {
                const ext = file.name.split('.').pop().toLowerCase();
                if (['pdf', 'txt', 'md'].includes(ext)) {
                    validFiles.push(file);
                } else {
                    invalidFiles.push(file.name);
                }
            });

            if (invalidFiles.length) {
                alert(`Unsupported file types: ${invalidFiles.join(', ')}`);
            }

            if (!validFiles.length) {
                return;
            }

            selectedFiles = validFiles;
            renderSelectedFiles();
        }

        function renderSelectedFiles() {
            const fileInfo = document.getElementById('file-info');
            const fileList = document.getElementById('file-list');
            fileList.innerHTML = '';

            if (!selectedFiles.length) {
                fileInfo.style.display = 'none';
                return;
            }

            fileInfo.style.display = 'block';
            selectedFiles.forEach((file, index) => {
                const item = document.createElement('div');
                item.style.display = 'flex';
                item.style.alignItems = 'center';
                item.style.justifyContent = 'space-between';
                item.style.gap = '8px';
                item.style.padding = '6px 8px';
                item.style.borderRadius = '6px';
                item.style.background = 'var(--bg-primary)';
                item.style.border = '1px solid var(--border-color)';
                item.innerHTML = `
                    <span class="badge badge-success" style="font-size: 10px;">📄 ${escapeHtml(file.name)}</span>
                    <button class="btn btn-secondary btn-sm" style="padding: 2px 6px; font-size: 10px;" onclick="removeSelectedFile(${index})">
                        ✕
                    </button>
                `;
                fileList.appendChild(item);
            });
        }

        function removeSelectedFile(index) {
            if (index < 0 || index >= selectedFiles.length) return;
            selectedFiles.splice(index, 1);
            renderSelectedFiles();
        }

        function clearSelectedFiles() {
            selectedFiles = [];
            const fileInput = document.getElementById('file-input');
            if (fileInput) {
                fileInput.value = '';
            }
            renderSelectedFiles();
        }

        function updateChunkingParamsVisibility() {
            const splitWithinFile = document.getElementById('split-within-file');
            const params = document.getElementById('chunking-params');
            if (!splitWithinFile || !params) return;
            params.style.display = splitWithinFile.checked ? 'block' : 'none';
        }

        // Process content - Chunk then GAM pipeline
        async function processContent() {
            const activeTab = document.querySelector('.input-tab.active').dataset.tab;
            let hasContent = false;
            const incrementalAdd = document.getElementById('incremental-add').checked;

            if (activeTab === 'text') {
                const content = document.getElementById('content-input').value.trim();
                if (content) hasContent = true;
            } else if (activeTab === 'file') {
                if (selectedFiles.length) hasContent = true;
            }

            if (!hasContent) {
                alert('Please enter some content or select a file');
                return;
            }
            
            if (incrementalAdd && !currentGAMDir) {
                alert('Incremental add requires an existing GAM. Please Build or Load an GAM first.');
                return;
            }

            // Show progress
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
            progressStatus.textContent = 'Starting...';
            addLog(incrementalAdd ? '➕ Starting incremental add...' : '🚀 Starting Chunk + GAM pipeline...', 'info');

            try {
                // Prepare form data
                const formData = new FormData();

                const windowSize = parseInt(document.getElementById('pipeline-window-size').value) || 8000;
                const overlapSize = parseInt(document.getElementById('pipeline-overlap-size').value) || 1000;
                const splitWithinFile = document.getElementById('split-within-file').checked;
                formData.append('window_size', windowSize);
                formData.append('overlap_size', overlapSize);
                formData.append('split_within_file', splitWithinFile ? '1' : '0');
                
                if (activeTab === 'file') {
                    selectedFiles.forEach(file => formData.append('files', file));
                    if (selectedFiles.length === 1) {
                        addLog(`📄 Uploading file: ${selectedFiles[0].name}`, 'info');
                    } else {
                        addLog(`📄 Uploading ${selectedFiles.length} files`, 'info');
                    }
                    if (!splitWithinFile) {
                        addLog('✂️ File splitting disabled: each file will be one chunk', 'warning');
                    }
                } else {
                    const content = document.getElementById('content-input').value;
                    formData.append('content', content);
                    addLog(`📝 Processing text content (${content.length} chars)`, 'info');
                    if (!splitWithinFile) {
                        addLog('✂️ Text splitting disabled: content will be one chunk', 'warning');
                    }
                }

                if (incrementalAdd) {
                    formData.append('gam_dir', currentGAMDir);
                }

                progressBar.style.width = '10%';
                progressStatus.textContent = incrementalAdd ? 'Starting incremental add...' : 'Starting pipeline...';

                // Start async pipeline and get task_id + paths immediately
                const startResponse = await fetch(incrementalAdd ? '/api/pipeline_add' : '/api/pipeline_start', {
                    method: 'POST',
                    body: formData
                });

                const startResult = await startResponse.json();

                if (!startResult.success) {
                    throw new Error(startResult.error || 'Failed to start pipeline');
                }

                const taskId = startResult.task_id;
                currentChunkDir = startResult.chunk_dir;
                currentChunkBaseDir = startResult.chunk_base_dir || null;
                currentGAMDir = startResult.gam_dir || currentGAMDir;

                // Show output paths immediately
                document.getElementById('output-paths').style.display = 'block';
                document.getElementById('chunk-path-display').textContent = startResult.chunk_base_dir || startResult.chunk_dir;
                document.getElementById('gam-path-display').textContent = startResult.gam_dir;

                addLog(`📂 Output directories created:`, 'info');
                addLog(`   Chunks: ${startResult.chunk_dir}`, 'info');
                addLog(`   GAM: ${startResult.gam_dir}`, 'info');

                // Switch to GAM view and start auto-refresh
                switchDirectory('gam');
                startAutoRefresh();

                progressBar.style.width = '20%';
                progressStatus.textContent = 'Processing...';

                // Poll for task completion
                let pollCount = 0;
                const maxPolls = 600;  // Max 10 minutes (600 * 1s)
                
                while (pollCount < maxPolls) {
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    pollCount++;

                    const statusResponse = await fetch(`/api/pipeline_status?task_id=${taskId}`);
                    const statusResult = await statusResponse.json();

                    if (statusResult.status === 'completed') {
                        // Stop auto-refresh
                        stopAutoRefresh();

                        progressBar.style.width = '100%';
                        progressStatus.textContent = 'Complete!';

                        // Log stages
                        addLog('', '');
                        addLog('📦 Stage 1: Chunking', 'warning');
                        addLog(`   Chunks created: ${statusResult.chunk_count || 0}`, 'success');
                        addLog(`   Output: ${currentChunkBaseDir || currentChunkDir}`, 'info');

                        addLog('', '');
                        addLog(incrementalAdd ? '🗂️ Stage 2: GAM Incremental Add' : '🗂️ Stage 2: GAM Organization', 'warning');
                        if (statusResult.gam_actions) {
                            statusResult.gam_actions.forEach(action => {
                                addLog(`   ${action}`, 'success');
                            });
                        }
                        addLog(`   Output: ${currentGAMDir}`, 'info');

                        addLog('', '');
                        addLog('✅ Pipeline completed successfully!', 'success');

                        // Update status
                        document.getElementById('process-status').textContent = 'Done';
                        document.getElementById('process-status').className = 'badge badge-success';

                        // Final refresh
                        refreshTree();
                        
                        // Clear cached input after successful build/add
                        clearSelectedFiles();
                        document.getElementById('content-input').value = '';
                        break;

                    } else if (statusResult.status === 'error') {
                        stopAutoRefresh();
                        throw new Error(statusResult.error || 'Pipeline failed');

                    } else if (statusResult.status === 'running') {
                        // Update progress based on stage
                        const stage = statusResult.stage || 'processing';
                        if (stage === 'chunking') {
                            progressBar.style.width = '40%';
                            progressStatus.textContent = 'Chunking content...';
                        } else if (stage === 'gam' || stage === 'gam_add') {
                            progressBar.style.width = '70%';
                            progressStatus.textContent = incrementalAdd
                                ? 'Adding chunks into GAM...'
                                : 'Building GAM structure...';
                        }
                        
                        // Log updates if any
                        if (statusResult.message) {
                            addLog(`   ${statusResult.message}`, 'info');
                        }
                    }
                }

                if (pollCount >= maxPolls) {
                    stopAutoRefresh();
                    throw new Error('Pipeline timed out');
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

        document.getElementById('split-within-file').addEventListener('change', updateChunkingParamsVisibility);
        updateChunkingParamsVisibility();

        function addLog(message, type = '') {
            const logOutput = document.getElementById('log-output');
            const line = document.createElement('div');
            line.className = `log-line ${type}`;
            line.textContent = message;
            logOutput.appendChild(line);
            logOutput.scrollTop = logOutput.scrollHeight;
        }

        // Refresh tree (resetState = true 会重置展开状态)
        async function refreshTree(resetState = false) {
            let dirPath = null;
            
            if (currentViewDir === 'chunk' && (currentChunkBaseDir || currentChunkDir)) {
                dirPath = currentChunkBaseDir || currentChunkDir;
            } else if (currentViewDir === 'gam' && currentGAMDir) {
                dirPath = currentGAMDir;
            }
            
            if (!dirPath) {
                document.getElementById('tree-view').innerHTML = `
                    <div class="empty-state">
                        <span class="icon">📂</span>
                        <h3>No output</h3>
                        <p>Process content to view files</p>
                    </div>
                `;
                return;
            }

            try {
                const endpoint = currentViewDir === 'chunk' ? '/api/browse_chunks' : '/api/browse';
                const response = await fetch(`${endpoint}?path=${encodeURIComponent(dirPath)}`);
                const data = await response.json();

                if (data.error) {
                    throw new Error(data.error);
                }

                // 如果需要重置状态
                if (resetState) {
                    treeState.expandedPaths.clear();
                    treeState.selectedPath = null;
                }
                
                lastTreeData = data;
                renderTree(data.tree, dirPath);
                
                document.getElementById('stat-dirs').textContent = data.stats.dirs;
                document.getElementById('stat-files').textContent = data.stats.files;

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

        // 存储当前树的状态（展开的节点、选中的节点）
        let treeState = {
            expandedPaths: new Set(),
            selectedPath: null,
        };

        function renderTree(node, basePath, container = null, relativePath = '', isRoot = true) {
            if (!container) {
                container = document.getElementById('tree-view');
                container.innerHTML = '';
            }

            // 对于根节点，relativePath 保持为空字符串
            // 对于子节点，构建相对路径
            const nodePath = isRoot ? '' : (relativePath ? `${relativePath}/${node.name}` : node.name);
            
            const nodeDiv = document.createElement('div');
            nodeDiv.className = 'tree-node';
            nodeDiv.dataset.nodePath = nodePath;

            const itemDiv = document.createElement('div');
            itemDiv.className = 'tree-item';
            itemDiv.dataset.path = nodePath;
            itemDiv.dataset.basePath = basePath;
            itemDiv.dataset.isDir = node.is_dir ? 'true' : 'false';

            const hasChildren = node.is_dir && node.children && node.children.length > 0;
            
            // 检查是否应该展开（根节点默认展开，或者之前已展开）
            const shouldExpand = isRoot || treeState.expandedPaths.has(nodePath);
            
            // 检查是否是选中状态
            const isSelected = treeState.selectedPath === nodePath;
            
            itemDiv.innerHTML = `
                <span class="toggle">${hasChildren ? (shouldExpand ? '▼' : '▶') : ''}</span>
                <span class="icon ${node.is_dir ? 'folder' : 'file'}">${node.is_dir ? '📁' : '📄'}</span>
                <span class="name">${node.name}</span>
            `;
            
            if (isSelected) {
                itemDiv.classList.add('selected');
            }

            itemDiv.addEventListener('click', (e) => {
                e.stopPropagation();
                
                // Toggle children
                if (hasChildren) {
                    const childContainer = nodeDiv.querySelector('.tree-children');
                    const toggle = itemDiv.querySelector('.toggle');
                    if (childContainer) {
                        const isCollapsed = childContainer.classList.toggle('collapsed');
                        toggle.textContent = isCollapsed ? '▶' : '▼';
                        
                        // 更新展开状态
                        if (isCollapsed) {
                            treeState.expandedPaths.delete(nodePath);
                        } else {
                            treeState.expandedPaths.add(nodePath);
                        }
                    }
                }

                // Select item
                document.querySelectorAll('.tree-item').forEach(i => i.classList.remove('selected'));
                itemDiv.classList.add('selected');
                treeState.selectedPath = nodePath;

                // Show preview - 使用完整路径
                const fullPath = nodePath ? `${basePath}/${nodePath}` : basePath;
                showPreviewByFullPath(fullPath, node.is_dir);
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

        // 同步左侧目录树的选中和展开状态
        function syncTreeToPath(fullPath) {
            // 获取当前目录树的 basePath
            let basePath = null;
            if (currentViewDir === 'chunk') {
                basePath = currentChunkBaseDir || currentChunkDir;
            } else if (currentViewDir === 'gam') {
                basePath = currentGAMDir;
            }
            if (!basePath) return;

            // 计算相对路径
            let relativePath = '';
            if (fullPath.startsWith(basePath + '/')) {
                relativePath = fullPath.slice(basePath.length + 1);
            } else if (fullPath === basePath) {
                relativePath = '';
            } else {
                return; // 不在当前树的范围内
            }

            // 展开所有祖先路径
            if (relativePath) {
                const parts = relativePath.split('/');
                let ancestor = '';
                for (let i = 0; i < parts.length - 1; i++) {
                    ancestor = ancestor ? `${ancestor}/${parts[i]}` : parts[i];
                    treeState.expandedPaths.add(ancestor);
                }
                // 如果目标本身是目录，也展开它
                // （后面 showPreviewByFullPath 会传入 isDir）
            }

            // 更新选中状态
            treeState.selectedPath = relativePath;

            // 重新渲染树以反映展开/选中状态
            if (lastTreeData) {
                renderTree(lastTreeData.tree, basePath);
            }

            // 滚动到选中的节点
            setTimeout(() => {
                const selectedItem = document.querySelector('.tree-item.selected');
                if (selectedItem) {
                    selectedItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            }, 50);
        }

        async function showPreviewByFullPath(fullPath, isDir) {
            try {
                // 同步左侧目录树
                syncTreeToPath(fullPath);

                const response = await fetch(`/api/file_content?path=${encodeURIComponent(fullPath)}`);
                const data = await response.json();

                if (data.error) {
                    throw new Error(data.error);
                }

                // 从完整路径提取显示名称
                const pathParts = fullPath.split('/');
                const displayName = pathParts[pathParts.length - 1] || '/';
                document.getElementById('preview-path').textContent = displayName;
                
                // Hide empty state, show content
                document.getElementById('preview-empty-state').style.display = 'none';
                const preview = document.getElementById('content-preview');
                preview.style.display = 'block';
                
                if (data.is_dir) {
                    // For directories, show directory tree listing
                    let html = '<div style="font-family: var(--font-mono, monospace); font-size: 12px;">';
                    html += '<div style="margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border-color, #e0e3e8); color: var(--text-muted, #9a9ab0);">';
                    html += `📁 <strong>${escapeHtml(data.name)}</strong>`;
                    const dirs = (data.children || []).filter(c => c.is_dir).length;
                    const files = (data.children || []).filter(c => !c.is_dir).length;
                    html += ` &nbsp;—&nbsp; ${dirs} folder${dirs !== 1 ? 's' : ''}, ${files} file${files !== 1 ? 's' : ''}`;
                    html += '</div>';
                    
                    if (!data.children || data.children.length === 0) {
                        html += '<p style="color: var(--text-muted, #9a9ab0);">Empty directory</p>';
                    } else {
                        // Sort: directories first, then files
                        const sorted = [...(data.children || [])].sort((a, b) => {
                            if (a.is_dir !== b.is_dir) return a.is_dir ? -1 : 1;
                            return a.name.localeCompare(b.name);
                        });
                        html += '<table style="width: 100%; border-collapse: collapse;">';
                        sorted.forEach(child => {
                            const icon = child.is_dir ? '📁' : '📄';
                            const nameStyle = child.is_dir 
                                ? 'color: var(--accent-primary, #6366f1); cursor: pointer;' 
                                : 'color: var(--text-primary, #1a1a2e); cursor: pointer;';
                            const sizeStr = child.is_dir ? '—' : formatFileSize(child.size || 0);
                            const childFullPath = data.full_path ? `${data.full_path}/${child.name}` : child.name;
                            html += `<tr style="border-bottom: 1px solid var(--border-color, #e0e3e8);">`;
                            html += `<td style="padding: 4px 8px; white-space: nowrap;">${icon}</td>`;
                            html += `<td style="padding: 4px 8px; ${nameStyle}" onclick="showPreviewByFullPath('${escapeHtml(childFullPath).replace(/'/g, "\\'")}', ${child.is_dir})">${escapeHtml(child.name)}</td>`;
                            html += `<td style="padding: 4px 8px; text-align: right; color: var(--text-muted, #9a9ab0); white-space: nowrap;">${sizeStr}</td>`;
                            html += `</tr>`;
                        });
                        html += '</table>';
                    }
                    html += '</div>';
                    preview.innerHTML = html;
                } else {
                    // For files
                    const content = data.content || '';
                    if (data.name && (data.name.endsWith('.md') || data.name.endsWith('.markdown'))) {
                        preview.innerHTML = marked.parse(content);
                    } else if (data.name && data.name.endsWith('.json')) {
                        try {
                            const formatted = JSON.stringify(JSON.parse(content), null, 2);
                            preview.innerHTML = `<pre><code class="language-json">${escapeHtml(formatted)}</code></pre>`;
                        } catch {
                            preview.innerHTML = `<pre><code>${escapeHtml(content)}</code></pre>`;
                        }
                    } else {
                        preview.innerHTML = `<pre><code>${escapeHtml(content)}</code></pre>`;
                    }
                }

                // Highlight code blocks
                preview.querySelectorAll('pre code').forEach((block) => {
                    if (!block.className || !block.className.includes('language-')) {
                        block.classList.add('language-plaintext');
                    }
                    hljs.highlightElement(block);
                });
            } catch (err) {
                console.error('Failed to load preview:', err);
                document.getElementById('preview-empty-state').style.display = 'none';
                const preview = document.getElementById('content-preview');
                preview.style.display = 'block';
                preview.innerHTML = `<p style="color: var(--accent-danger);">Error: ${err.message}</p>`;
            }
        }

        function clearPreview() {
            document.getElementById('preview-path').textContent = 'Select a file to preview';
            document.getElementById('preview-empty-state').style.display = 'flex';
            document.getElementById('content-preview').style.display = 'none';
        }

        // Navigate to a source file from chat messages
        async function navigateToSource(sourceName) {
            if (!currentGAMDir) {
                alert('GAM directory not loaded. Please build or load an GAM first.');
                return;
            }
            
            // Try to find the file in GAM directory
            try {
                const response = await fetch(`/api/find_source?gam_dir=${encodeURIComponent(currentGAMDir)}&source_name=${encodeURIComponent(sourceName)}`);
                const data = await response.json();
                
                if (data.found && data.full_path) {
                    // Show the file in preview
                    showPreviewByFullPath(data.full_path, false);
                    
                    // Optionally scroll to the middle panel on mobile
                    const middlePanel = document.querySelector('.middle-panel');
                    if (middlePanel && window.innerWidth < 1200) {
                        middlePanel.scrollIntoView({ behavior: 'smooth' });
                    }
                } else {
                    // File not found in GAM, try in chunks directory
                    const chunkSearchDir = currentChunkBaseDir || currentChunkDir;
                    if (chunkSearchDir) {
                        const chunkResponse = await fetch(`/api/find_source?gam_dir=${encodeURIComponent(chunkSearchDir)}&source_name=${encodeURIComponent(sourceName)}`);
                        const chunkData = await chunkResponse.json();
                        
                        if (chunkData.found && chunkData.full_path) {
                            showPreviewByFullPath(chunkData.full_path, false);
                            return;
                        }
                    }
                    alert(`Source file "${sourceName}" not found.`);
                }
            } catch (err) {
                console.error('Failed to navigate to source:', err);
                alert(`Error finding source: ${err.message}`);
            }
        }

        function resetAll() {
            if (confirm('Start a new session? This will clear current paths.')) {
                // Stop any ongoing auto-refresh
                stopAutoRefresh();
                
                currentChunkDir = null;
                currentChunkBaseDir = null;
                currentGAMDir = null;
                currentViewDir = null;
                clearSelectedFiles();
                lastTreeData = null;
                
                // 重置树状态
                treeState.expandedPaths.clear();
                treeState.selectedPath = null;
                
                document.getElementById('content-input').value = '';
                document.getElementById('output-paths').style.display = 'none';
                document.getElementById('progress-container').classList.remove('active');
                
                document.getElementById('tree-view').innerHTML = `
                    <div class="empty-state">
                        <span class="icon">📂</span>
                        <h3>No output</h3>
                        <p>Process content to view files</p>
                    </div>
                `;
                
                // Clear preview section
                clearPreview();
                
                document.getElementById('process-status').textContent = 'Ready';
                document.getElementById('process-status').className = 'badge badge-info';
                
                document.getElementById('stat-dirs').textContent = '0';
                document.getElementById('stat-files').textContent = '0';
                
                // Reset directory buttons
                document.getElementById('btn-chunk-dir').classList.remove('active', 'chunk-active', 'gam-active');
                document.getElementById('btn-gam-dir').classList.remove('active', 'chunk-active', 'gam-active');
            }
        }

        function getChunkBaseDir(path) {
            if (!path) return null;
            const normalized = path.replace(/\/+$/, '');
            if (/\/chunks_\d+$/.test(normalized)) {
                return normalized.replace(/\/chunks_\d+$/, '');
            }
            return normalized;
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

        // ========== Load Directory Functions ==========
        let browserCurrentPath = '/share/project/chaofan/code/memory/gam/examples/output/web_test';
        
        function showLoadModal() {
            document.getElementById('load-modal').classList.add('show');
            loadRecentSessions();
            browserNavigate(browserCurrentPath);
        }

        function hideLoadModal() {
            document.getElementById('load-modal').classList.remove('show');
        }

        // Close modal when clicking overlay
        document.getElementById('load-modal').addEventListener('click', (e) => {
            if (e.target.id === 'load-modal') {
                hideLoadModal();
            }
        });

        // Folder Browser Functions
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
                
                // Sort: directories first, then by name
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
                                <button class="folder-action-btn chunk-btn" onclick="event.stopPropagation(); setChunkPath('${fullPath}')" title="Set as Chunk Dir">📦</button>
                                <button class="folder-action-btn gam-btn" onclick="event.stopPropagation(); setGamPath('${fullPath}')" title="Set as GAM Dir">🗂️</button>
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
        
        function setChunkPath(path) {
            document.getElementById('load-chunk-path').value = path;
        }
        
        function setGamPath(path) {
            document.getElementById('load-gam-path').value = path;
        }

        async function loadRecentSessions() {
            try {
                const response = await fetch('/api/recent_sessions');
                const data = await response.json();
                
                const section = document.getElementById('quick-load-section');
                const list = document.getElementById('quick-load-list');
                
                if (data.success && data.sessions && data.sessions.length > 0) {
                    section.style.display = 'block';
                    list.innerHTML = '';
                    
                    data.sessions.slice(0, 5).forEach(session => {
                        const item = document.createElement('div');
                        item.className = 'quick-load-item';
                        item.style.cssText = 'padding: 8px 10px; margin-bottom: 6px;';
                        item.innerHTML = `
                            <span class="icon" style="font-size: 14px;">📁</span>
                            <div class="info">
                                <div class="name" style="font-size: 11px;">${session.name}</div>
                            </div>
                        `;
                        item.onclick = () => {
                            document.getElementById('load-chunk-path').value = session.chunk_dir || '';
                            document.getElementById('load-gam-path').value = session.gam_dir || '';
                        };
                        list.appendChild(item);
                    });
                } else {
                    section.style.display = 'none';
                }
            } catch (err) {
                console.error('Failed to load recent sessions:', err);
                document.getElementById('quick-load-section').style.display = 'none';
            }
        }

        async function loadDirectories() {
            const chunkPath = document.getElementById('load-chunk-path').value.trim();
            const gamPath = document.getElementById('load-gam-path').value.trim();
            
            if (!chunkPath && !gamPath) {
                alert('Please enter at least one directory path');
                return;
            }
            
            // Validate paths exist
            let validChunk = false;
            let validGam = false;
            
            if (chunkPath) {
                try {
                    const response = await fetch(`/api/browse?path=${encodeURIComponent(chunkPath)}`);
                    const data = await response.json();
                    if (!data.error) {
                        validChunk = true;
                        currentChunkDir = chunkPath;
                        currentChunkBaseDir = getChunkBaseDir(chunkPath);
                    }
                } catch (err) {
                    console.error('Chunk path validation failed:', err);
                }
            }
            
            if (gamPath) {
                try {
                    const response = await fetch(`/api/browse?path=${encodeURIComponent(gamPath)}`);
                    const data = await response.json();
                    if (!data.error) {
                        validGam = true;
                        currentGAMDir = gamPath;
                        if (!validChunk) {
                            currentChunkBaseDir = getChunkBaseDir(gamPath);
                        }
                    }
                } catch (err) {
                    console.error('GAM path validation failed:', err);
                }
            }
            
            if (!validChunk && !validGam) {
                alert('Neither path could be loaded. Please check the paths exist.');
                return;
            }
            
            // Update UI
            hideLoadModal();
            
            // Show output paths
            document.getElementById('output-paths').style.display = 'block';
            document.getElementById('chunk-path-display').textContent = currentChunkBaseDir || currentChunkDir || 'Not loaded';
            document.getElementById('gam-path-display').textContent = currentGAMDir || 'Not loaded';
            
            // Update status
            document.getElementById('process-status').textContent = 'Loaded';
            document.getElementById('process-status').className = 'badge badge-success';
            
            // Show progress container for logs
            const progressContainer = document.getElementById('progress-container');
            const logOutput = document.getElementById('log-output');
            progressContainer.classList.add('active');
            logOutput.innerHTML = '';
            document.getElementById('progress-bar').style.width = '100%';
            document.getElementById('progress-status').textContent = 'Loaded successfully';
            
            addLog(`📂 Loaded existing directories:`, 'info');
            if (validChunk) addLog(`   Chunks: ${chunkPath}`, 'success');
            if (validGam) addLog(`   GAM: ${gamPath}`, 'success');
            
            // Switch to appropriate view and refresh
            if (validGam) {
                switchDirectory('gam');
            } else if (validChunk) {
                switchDirectory('chunk');
            }
        }

        // ========== Chat Functions ==========
        let chatMessages = [];
        let isChatProcessing = false;

        function updateChatStatus(status, text) {
            const dot = document.getElementById('chat-status-dot');
            const statusText = document.getElementById('chat-status-text');
            
            dot.className = 'dot';
            if (status === 'thinking') {
                dot.classList.add('thinking');
            } else if (status === 'disconnected') {
                dot.classList.add('disconnected');
            }
            
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
            
            // Hide empty state
            if (emptyState) {
                emptyState.style.display = 'none';
            }
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `chat-message ${role}`;
            
            const avatar = role === 'user' ? '👤' : '🤖';
            
            let sourcesHtml = '';
            if (sources && sources.length > 0) {
                sourcesHtml = `
                    <div class="message-sources">
                        <div class="message-sources-title">📎 Sources: <span style="font-weight: normal; font-size: 10px; color: var(--text-muted);">(点击查看)</span></div>
                        ${sources.map(s => {
                            const sourceValue = String(s);
                            const displayName = sourceValue.split('/').pop() || sourceValue;
                            const encoded = encodeURIComponent(sourceValue);
                            return `<span class="message-source-item" data-source="${encoded}" title="${escapeHtml(displayName)}">${escapeHtml(displayName)}</span>`;
                        }).join('')}
                    </div>
                `;
            }
            
            // Parse markdown for assistant messages
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
                    const encoded = item.dataset.source || '';
                    const sourceValue = decodeURIComponent(encoded);
                    navigateToSource(sourceValue);
                });
            });
            
            // Highlight code blocks
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
            
            if (emptyState) {
                emptyState.style.display = 'none';
            }
            
            const typingDiv = document.createElement('div');
            typingDiv.className = 'chat-message assistant';
            typingDiv.id = 'typing-indicator';
            typingDiv.innerHTML = `
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    <div class="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                    <div class="agent-actions" id="agent-actions-log"></div>
                </div>
            `;
            
            messagesContainer.appendChild(typingDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function removeTypingIndicator() {
            const typingDiv = document.getElementById('typing-indicator');
            if (typingDiv) {
                typingDiv.remove();
            }
        }

        function addAgentAction(eventData) {
            const actionsLog = document.getElementById('agent-actions-log');
            if (!actionsLog) return;
            
            const item = document.createElement('div');
            item.className = 'agent-action-item';
            
            if (eventData.type === 'tool_call') {
                const icon = eventData.tool === 'ls' ? '📂' : 
                             eventData.tool === 'cat' ? '📄' : 
                             eventData.tool === 'grep' ? '🔍' : '🔧';
                item.innerHTML = `<span class="agent-action-icon">${icon}</span><span class="agent-action-cmd">${escapeHtml(eventData.display || eventData.tool)}</span>`;
            } else if (eventData.type === 'thinking') {
                item.classList.add('action-thinking');
                item.innerHTML = `<span class="agent-action-icon">💭</span><span class="agent-action-cmd">${escapeHtml(eventData.message || 'Thinking...')}</span>`;
            } else if (eventData.type === 'round') {
                item.classList.add('action-round');
                item.innerHTML = `<span class="agent-action-icon">🔄</span><span class="agent-action-cmd">Round ${eventData.round}/${eventData.max_rounds}</span>`;
            } else {
                return; // ignore unknown types
            }
            
            actionsLog.appendChild(item);
            
            // Auto-scroll the actions log
            actionsLog.scrollTop = actionsLog.scrollHeight;
            
            // Also scroll the messages container
            const messagesContainer = document.getElementById('chat-messages');
            if (messagesContainer) {
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
        }

        async function sendMessage() {
            const input = document.getElementById('chat-input');
            const sendBtn = document.getElementById('chat-send-btn');
            const question = input.value.trim();
            
            if (!question || isChatProcessing) return;
            
            // Check if GAM is loaded
            if (!currentGAMDir) {
                addMessage('assistant', '⚠️ Please build an GAM first before asking questions. Upload content and click "Chunk & Build GAM" to get started.');
                return;
            }
            
            // Clear input
            input.value = '';
            input.style.height = 'auto';
            
            // Add user message
            addMessage('user', question);
            
            // Update status
            isChatProcessing = true;
            sendBtn.disabled = true;
            updateChatStatus('thinking', 'Researching...');
            addTypingIndicator();
            
            try {
                const formData = new FormData();
                formData.append('question', question);
                formData.append('gam_path', currentGAMDir);
                
                const response = await fetch('/api/research_stream', {
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
                    
                    // Parse SSE events from buffer
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';  // Keep incomplete line in buffer
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const eventData = JSON.parse(line.slice(6));
                                
                                if (eventData.type === 'complete') {
                                    finalResult = eventData;
                                } else if (eventData.type === 'error') {
                                    finalResult = { error: eventData.error };
                                } else {
                                    // Show agent action in the typing indicator area
                                    addAgentAction(eventData);
                                }
                            } catch (parseErr) {
                                // Ignore unparseable lines
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

        function clearChat() {
            const messagesContainer = document.getElementById('chat-messages');
            const emptyState = document.getElementById('chat-empty-state');
            
            messagesContainer.innerHTML = '';
            if (emptyState) {
                messagesContainer.appendChild(emptyState);
                emptyState.style.display = 'flex';
            }
            
            chatMessages = [];
        }