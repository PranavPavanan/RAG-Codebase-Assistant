/**
 * RAG GitHub Assistant — Frontend Application
 */

class ApiClient {
  constructor(baseUrl) { this.baseUrl = baseUrl; }
  async req(path, opts = {}) {
    const res = await fetch(`${this.baseUrl}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...opts
    });
    if (!res.ok) {
      let detail = '';
      try { const j = await res.json(); detail = j.detail || JSON.stringify(j); } catch {}
      throw new Error(detail || `${res.status} ${res.statusText}`);
    }
    return res.json();
  }
  health() { return this.req('/health'); }
  searchRepositories(query, language, minStars) { return this.req('/search/repositories', { method: 'POST', body: JSON.stringify({ query, language, min_stars: minStars }) }); }
  startIndexing(repository_url, branch) { return this.req('/index/start', { method: 'POST', body: JSON.stringify({ repository_url, branch }) }); }
  indexStatus(taskId) { return this.req(`/index/status/${taskId}`); }
  indexStats() { return this.req('/index/stats'); }
  clearIndex() { return this.req('/index/current', { method: 'DELETE' }); }
  chatQuery(query, conversation_id, session_id) { return this.req('/chat/query', { method: 'POST', body: JSON.stringify({ query, conversation_id, session_id }) }); }
}

class RAGApp {
  constructor() {
    this.api = new ApiClient('http://localhost:8000/api');
    this.state = {
      selectedRepository: null,
      searchResults: [],
      isSearching: false,
      indexStats: null,
      currentIndexingTask: null,
      isIndexing: false,
      messages: [],
      conversationId: null,
      sessionId: null,
      currentConversation: null,
      conversations: [],
      isQuerying: false,
      route: 'search',
      elapsedMs: 0,
      elapsedTimer: null,
      theme: 'system'
    };
    this.init();
  }

  init() {
    this.ensureToastContainer();
    this.initTheme();
    this.bindNav();
    this.bindEvents();
    this.bindWelcomeChips();
    this.bindMobileSidebar();
    this.loadIndexStats();

    const savedRoute = localStorage.getItem('rag.route');
    if (savedRoute && ['search', 'index', 'chat'].includes(savedRoute)) {
      this.switchPanel(savedRoute === 'chat' && !this.isIndexed() ? 'search' : savedRoute);
    } else {
      this.switchPanel('search');
    }
    this.setupScrollToBottom();
    this.setupGlobalShortcuts();
    this.icons();
  }

  icons() { if (typeof lucide !== 'undefined') lucide.createIcons(); }

  // ── Navigation ────────────────────────────────────────────
  bindNav() {
    document.querySelectorAll('.nav-item').forEach(btn => {
      btn.addEventListener('click', (e) => {
        if (!e.currentTarget.disabled) this.switchPanel(e.currentTarget.dataset.route);
      });
    });
  }

  switchPanel(route) {
    this.state.route = route;
    try { localStorage.setItem('rag.route', route); } catch {}

    document.querySelectorAll('.nav-item').forEach(el =>
      el.classList.toggle('active', el.dataset.route === route)
    );

    const panels = ['panel-search', 'panel-index', 'panel-chat'];
    panels.forEach(id => {
      const el = document.getElementById(id);
      if (!el) return;
      el.classList.remove('active');
      el.style.display = 'none';
    });

    const map = { search: 'panel-search', index: 'panel-index', chat: 'panel-chat' };
    const panel = document.getElementById(map[route]);
    if (panel) {
      if (route === 'chat') {
        panel.style.display = 'flex';
        panel.style.flexDirection = 'column';
      } else {
        panel.style.display = '';
      }
      // Trigger reflow for animation
      void panel.offsetWidth;
      panel.classList.add('active');
    }

    const titles = { search: 'Search GitHub', index: 'Repository Indexing', chat: 'Chat about Code' };
    const headline = document.getElementById('headline-text');
    if (headline) headline.textContent = titles[route] || '';

    if (route === 'index') this.updateIndexDisplay();
    if (route === 'chat') this.updateChatInterface();

    // Close mobile sidebar
    this.closeMobileSidebar();
  }

  // ── Mobile Sidebar ────────────────────────────────────────
  bindMobileSidebar() {
    const btn = document.getElementById('mobile-menu-btn');
    const overlay = document.getElementById('sidebar-overlay');
    if (btn) btn.addEventListener('click', () => this.openMobileSidebar());
    if (overlay) overlay.addEventListener('click', () => this.closeMobileSidebar());
  }
  openMobileSidebar() {
    document.getElementById('sidebar')?.classList.add('open');
    document.getElementById('sidebar-overlay')?.classList.add('show');
  }
  closeMobileSidebar() {
    document.getElementById('sidebar')?.classList.remove('open');
    document.getElementById('sidebar-overlay')?.classList.remove('show');
  }

  // ── Toasts ────────────────────────────────────────────────
  ensureToastContainer() {
    if (!document.querySelector('.toast-container')) {
      const c = document.createElement('div'); c.className = 'toast-container'; document.body.appendChild(c);
    }
  }
  showToast(message, timeout = 2500) {
    this.ensureToastContainer();
    const c = document.querySelector('.toast-container');
    const t = document.createElement('div'); t.className = 'toast'; t.textContent = message;
    c.appendChild(t);
    setTimeout(() => {
      t.classList.add('dismissing');
      setTimeout(() => t.remove(), 280);
    }, timeout);
  }

  // ── Helpers ───────────────────────────────────────────────
  showError(id, msg) {
    const el = document.getElementById(id);
    if (el) { el.textContent = msg; el.classList.remove('hidden'); }
  }
  hideError(id) {
    const el = document.getElementById(id);
    if (el) el.classList.add('hidden');
  }
  setLoading(btnId, isLoading, text) {
    const btn = document.getElementById(btnId); if (!btn) return;
    const spinner = btn.querySelector('.btn-spinner');
    const label = btn.querySelector('.btn-text');
    if (isLoading) {
      btn.disabled = true;
      spinner?.classList.remove('hidden');
      if (label && text) label.textContent = text;
    } else {
      btn.disabled = false;
      spinner?.classList.add('hidden');
      if (label) label.textContent = btn.dataset.originalText || label.textContent;
    }
  }

  // ── Search ────────────────────────────────────────────────
  async handleSearch() {
    const q = document.getElementById('search-query').value.trim();
    if (!q) return this.showError('search-error', 'Please enter a search query');
    this.hideError('search-error');
    this.state.isSearching = true;
    this.setLoading('search-btn', true, 'Searching...');
    try {
      const res = await this.api.searchRepositories(q);
      this.state.searchResults = res.repositories || [];
      this.displaySearchResults();
      if (this.state.searchResults.length === 0) this.showError('search-error', 'No repositories found');
    } catch (e) {
      this.showError('search-error', e.message || 'Failed to search repositories');
      this.state.searchResults = [];
    } finally {
      this.state.isSearching = false; this.setLoading('search-btn', false);
    }
  }

  displaySearchResults() {
    const container = document.getElementById('search-results');
    const list = document.getElementById('results-list');
    const title = document.getElementById('results-title');
    if (this.state.searchResults.length === 0) { container.classList.add('hidden'); return; }
    title.textContent = `${this.state.searchResults.length} result${this.state.searchResults.length !== 1 ? 's' : ''}`;
    list.innerHTML = '';
    this.state.searchResults.forEach((repo, i) => {
      const item = document.createElement('div');
      item.className = 'result-item';
      item.style.animationDelay = `${i * 0.04}s`;
      item.innerHTML = `
        <div class="result-header">
          <div class="result-name">
            <span>${repo.full_name}</span>
            <a href="${repo.html_url}" target="_blank" rel="noopener noreferrer" class="result-link" onclick="event.stopPropagation()">
              <i data-lucide="external-link"></i>
            </a>
          </div>
          <button class="btn btn-sm btn-outline">Select</button>
        </div>
        ${repo.description ? `<p class="result-description">${repo.description}</p>` : ''}
        <div class="result-stats">
          ${repo.language ? `<span class="repo-stat"><span class="language-dot"></span><span class="language-text">${repo.language}</span></span>` : ''}
          <span class="repo-stat"><i data-lucide="star"></i><span>${(repo.stars || 0).toLocaleString()}</span></span>
          ${repo.forks ? `<span class="repo-stat"><i data-lucide="git-fork"></i><span>${repo.forks.toLocaleString()}</span></span>` : ''}
        </div>`;
      item.addEventListener('click', () => this.selectRepository(repo));
      list.appendChild(item);
    });
    container.classList.remove('hidden');
    this.icons();
  }

  selectRepository(repo) {
    this.state.selectedRepository = repo;
    this.displaySelectedRepository();
    this.switchPanel('index');
    this.showToast(`Selected: ${repo.full_name}`);
  }

  displaySelectedRepository() {
    const sel = document.getElementById('selected-repo');
    if (!this.state.selectedRepository) { sel.classList.add('hidden'); return; }
    document.getElementById('repo-name').textContent = this.state.selectedRepository.full_name;
    document.getElementById('repo-link').href = this.state.selectedRepository.html_url;
    document.getElementById('repo-description').textContent = this.state.selectedRepository.description || '';
    document.getElementById('repo-stars').textContent = (this.state.selectedRepository.stars || 0).toLocaleString();
    const lang = document.getElementById('repo-language');
    if (this.state.selectedRepository.language) {
      lang.classList.remove('hidden');
      lang.querySelector('.language-text').textContent = this.state.selectedRepository.language;
    } else { lang.classList.add('hidden'); }
    sel.classList.remove('hidden');
    this.icons();
  }

  clearSelection() { this.state.selectedRepository = null; this.displaySelectedRepository(); }

  // ── Indexing ──────────────────────────────────────────────
  async handleStartIndexing() {
    const url = this.state.selectedRepository?.html_url || document.getElementById('repo-url').value.trim();
    if (!url) return this.showError('indexing-error', 'Please select a repository or enter a URL');
    this.hideError('indexing-error');
    this.state.isIndexing = true;
    this.setLoading('start-indexing', true, 'Starting...');
    try {
      const res = await this.api.startIndexing(url);
      this.state.currentIndexingTask = res;
      this.displayIndexingProgress();
      this.startIndexingPolling();
    } catch (e) {
      this.showError('indexing-error', e.message || 'Failed to start indexing');
      this.state.isIndexing = false;
    } finally { this.setLoading('start-indexing', false); }
  }

  displayIndexingProgress() {
    document.getElementById('indexing-progress').classList.remove('hidden');
    this.updateIndexingStatus();
  }

  updateIndexingStatus() {
    if (!this.state.currentIndexingTask) return;
    const t = this.state.currentIndexingTask;
    const icon = document.getElementById('progress-icon');
    const statusEl = document.getElementById('progress-status');
    const perc = document.getElementById('progress-percentage');
    const fill = document.getElementById('progress-fill');
    const files = document.getElementById('files-processed');
    const total = document.getElementById('total-files');
    const err = document.getElementById('progress-error');
    const startQ = document.getElementById('start-querying');

    statusEl.textContent = t.status;
    statusEl.className = `progress-status ${t.status}`;

    const iconMap = { completed: 'check-circle-2', failed: 'x-circle', running: 'loader-2', pending: 'clock' };
    icon.setAttribute('data-lucide', iconMap[t.status] || 'clock');
    icon.style.animation = t.status === 'running' ? 'spin 0.9s linear infinite' : '';

    const pct = t.progress?.percentage ?? 0;
    perc.textContent = `${pct.toFixed(1)}%`;
    fill.style.width = `${pct}%`;
    files.textContent = t.progress?.files_processed ?? 0;
    total.textContent = t.progress?.total_files ?? 0;

    if (t.error) { err.textContent = t.error; err.classList.remove('hidden'); } else { err.classList.add('hidden'); }
    if (t.status === 'completed') { startQ.classList.remove('hidden'); this.state.isIndexing = false; }
    else { startQ.classList.add('hidden'); }
    this.icons();
  }

  startIndexingPolling() {
    if (!this.state.currentIndexingTask?.task_id) return;
    const it = setInterval(async () => {
      try {
        const st = await this.api.indexStatus(this.state.currentIndexingTask.task_id);
        this.state.currentIndexingTask = st;
        this.updateIndexingStatus();
        if (st.status === 'completed' || st.status === 'failed') {
          clearInterval(it);
          this.state.isIndexing = false;
          await this.loadIndexStats();
          if (st.status === 'completed') this.showToast('✓ Indexing complete — ready to chat!');
        }
      } catch (e) { console.error('Failed to fetch indexing status:', e); }
    }, 2000);
  }

  async handleClearIndex() {
    if (!confirm('Are you sure you want to clear the index?')) return;
    try {
      await this.api.clearIndex();
      this.state.indexStats = null;
      this.state.currentIndexingTask = null;
      this.hideError('indexing-error');
      this.updateIndexDisplay();
      this.switchPanel('search');
      this.updateIndexAwareness();
      this.showToast('Index cleared');
    } catch (e) { this.showError('indexing-error', e.message || 'Failed to clear index'); }
  }

  async loadIndexStats() {
    try {
      const stats = await this.api.indexStats();
      this.state.indexStats = stats;
      this.updateIndexDisplay();
      this.updateChatInterface();
      this.updateIndexAwareness();
    } catch {
      this.state.indexStats = null;
      this.updateIndexAwareness();
    }
  }

  isIndexed() {
    const s = this.state.indexStats;
    return !!(s?.is_indexed || (s?.file_count > 0) || (s?.vector_count > 0) || (s?.repository_name && s?.file_count >= 0));
  }

  updateIndexAwareness() {
    const chatBtn = document.querySelector('.nav-item[data-route="chat"]');
    const isIndexed = this.isIndexed();
    if (chatBtn) chatBtn.disabled = !isIndexed;

    const chip = document.getElementById('status-chip');
    const text = document.getElementById('status-text');
    const banner = document.getElementById('repo-banner');

    if (isIndexed) {
      chip?.classList.add('ready');
      if (text) text.textContent = 'Ready';
      if (banner) {
        banner.classList.remove('hidden');
        document.getElementById('repo-banner-name').textContent = this.state.indexStats.repository_name || 'Unknown';
        document.getElementById('repo-banner-files').textContent = `${this.state.indexStats.file_count} files`;
      }
    } else {
      chip?.classList.remove('ready');
      if (text) text.textContent = 'No index';
      banner?.classList.add('hidden');
    }
    this.icons();
  }

  updateIndexDisplay() {
    const cur = document.getElementById('current-index');
    const prog = document.getElementById('indexing-progress');
    const sel = document.getElementById('selected-repo-display');
    const urlGrp = document.getElementById('url-input-group');
    const startBtn = document.getElementById('start-indexing');
    const isIndexed = this.isIndexed();

    if (isIndexed) {
      cur.classList.remove('hidden');
      document.getElementById('index-repo-name').textContent = this.state.indexStats.repository_name || '—';
      document.getElementById('index-file-count').textContent = (this.state.indexStats.file_count || 0).toLocaleString();
      document.getElementById('index-vector-count').textContent = (this.state.indexStats.vector_count || 0).toLocaleString();
      const t = startBtn?.querySelector('.btn-text'); if (t) t.textContent = 'Re-index Repository';
    } else {
      cur.classList.add('hidden');
      const t = startBtn?.querySelector('.btn-text'); if (t) t.textContent = 'Start Indexing';
    }

    if (this.state.currentIndexingTask) prog.classList.remove('hidden'); else prog.classList.add('hidden');
    if (this.state.selectedRepository) {
      sel.classList.remove('hidden');
      document.getElementById('selected-repo-name').textContent = this.state.selectedRepository.full_name;
      urlGrp.classList.add('hidden');
    } else { sel.classList.add('hidden'); urlGrp.classList.remove('hidden'); }
  }

  // ── Chat ──────────────────────────────────────────────────
  updateChatInterface() {
    const noIdx = document.getElementById('no-index-message');
    const welcome = document.getElementById('welcome-message');
    const input = document.getElementById('chat-input');
    const submit = document.getElementById('chat-submit');
    const repoBar = document.getElementById('repo-info-bar');
    const clearBtn = document.getElementById('clear-chat');

    if (this.isIndexed()) {
      noIdx.classList.add('hidden');
      if (this.state.messages.length === 0) welcome.classList.remove('hidden');
      else welcome.classList.add('hidden');

      input.disabled = this.state.isQuerying;
      input.placeholder = `Ask about ${this.state.indexStats?.repository_name || 'the code'}…`;
      submit.disabled = this.state.isQuerying;

      const sendIcon = submit.querySelector('[data-lucide="send"]');
      const spinIcon = submit.querySelector('[data-lucide="loader-2"]');
      if (this.state.isQuerying) { sendIcon?.classList.add('hidden'); spinIcon?.classList.remove('hidden'); }
      else { sendIcon?.classList.remove('hidden'); spinIcon?.classList.add('hidden'); }

      if (this.state.messages.length > 0) clearBtn.disabled = false;

      if (this.state.indexStats?.repository_name) {
        document.getElementById('chat-repo-name').textContent = this.state.indexStats.repository_name;
        repoBar.classList.remove('hidden');
      }
    } else {
      noIdx.classList.remove('hidden');
      welcome.classList.add('hidden');
      input.disabled = true;
      input.placeholder = 'Index a repository first…';
      submit.disabled = true;
      if (clearBtn) clearBtn.disabled = true;
      repoBar.classList.add('hidden');
    }
  }

  async handleChatSubmit() {
    const input = document.getElementById('chat-input');
    const query = input.value.trim();
    if (!query || this.state.isQuerying) return;
    if (!this.isIndexed()) return this.showError('chat-error', 'Please index a repository first');

    const userMsg = { role: 'user', content: query, timestamp: new Date().toISOString() };
    this.addMessage(userMsg);
    input.value = '';
    this.hideError('chat-error');
    this.state.isQuerying = true;
    this.startElapsedTimer();
    this.updateChatInterface();
    this.displayMessages();

    try {
      const res = await this.api.chatQuery(query, this.state.conversationId, this.state.sessionId);
      if (!this.state.sessionId) this.state.sessionId = res.session_id;
      if (!this.state.conversationId) {
        this.state.conversationId = res.conversation_id;
        if (!this.state.currentConversation) {
          const conv = { id: res.conversation_id, title: 'New Chat', messages: [userMsg], createdAt: new Date().toISOString(), lastActivity: new Date().toISOString() };
          this.state.currentConversation = conv;
          this.state.conversations.push(conv);
          this.updateConversationTabs();
        }
      }
      const assistantMsg = { role: 'assistant', content: res.answer || res.response, timestamp: new Date().toISOString(), sources: res.sources || res.references };
      this.addMessage(assistantMsg);
      this.displayMessages();

      if (this.state.currentConversation?.title === 'New Chat' && this.state.currentConversation.messages.length <= 2) {
        const title = query.slice(0, 30);
        this.state.currentConversation.title = title.length < query.length ? `${title}…` : title;
        this.updateConversationTabs();
      }
    } catch (e) {
      this.showError('chat-error', e.message || 'Failed to process query');
    } finally {
      this.state.isQuerying = false;
      this.stopElapsedTimer();
      this.updateChatInterface();
    }
  }

  addMessage(m) {
    this.state.messages.push(m);
    if (this.state.currentConversation) {
      this.state.currentConversation.messages.push(m);
      this.state.currentConversation.lastActivity = new Date().toISOString();
    }
    this.displayMessages();
  }

  displayMessages() {
    const box = document.getElementById('chat-messages');
    const noIdx = document.getElementById('no-index-message');
    const welcome = document.getElementById('welcome-message');

    if (this.state.messages.length === 0) {
      if (this.isIndexed()) { noIdx.classList.add('hidden'); welcome.classList.remove('hidden'); }
      else { noIdx.classList.remove('hidden'); welcome.classList.add('hidden'); }
      // Remove any existing messages
      box.querySelectorAll('.message, .generating-message').forEach(n => n.remove());
      return;
    }
    noIdx.classList.add('hidden');
    welcome.classList.add('hidden');
    box.querySelectorAll('.message, .generating-message').forEach(n => n.remove());
    this.state.messages.forEach(m => box.appendChild(this.createMessageElement(m)));
    if (this.state.isQuerying) box.appendChild(this.createGeneratingElement());
    box.scrollTop = box.scrollHeight;
    this.icons();
  }

  createMessageElement(m) {
    const wrap = document.createElement('div');
    wrap.className = `message ${m.role}`;

    // Avatar
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    if (m.role === 'user') {
      avatar.innerHTML = '<i data-lucide="user-2"></i>';
    } else {
      avatar.innerHTML = '<i data-lucide="cpu"></i>';
    }

    const content = document.createElement('div');
    content.className = 'message-content';

    const text = document.createElement('div');
    text.className = 'message-text';
    text.textContent = m.content;
    content.appendChild(text);

    if (m.sources && m.sources.length > 0) {
      const sources = document.createElement('div');
      sources.className = 'message-sources';
      const srcTitle = document.createElement('div');
      srcTitle.className = 'message-sources-title';
      srcTitle.textContent = `Sources (${m.sources.length})`;
      sources.appendChild(srcTitle);

      m.sources.forEach(s => {
        const item = document.createElement('div');
        item.className = 'source-item';
        const filePath = s.file || s.file_path || s.path || 'unknown';
        const lineStart = s.line_start || s.start_line || 1;
        const lineEnd = s.line_end || s.end_line || lineStart;
        const score = s.score !== undefined ? ` · ${(s.score * 100).toFixed(0)}% match` : '';

        const header = document.createElement('div');
        header.className = 'source-header';
        header.innerHTML = `<i data-lucide="file-code-2"></i><span class="source-file">${filePath}</span><span class="source-lines">L${lineStart}–${lineEnd}${score}</span>`;

        const copyBtn = document.createElement('button');
        copyBtn.className = 'btn btn-ghost btn-sm';
        copyBtn.type = 'button';
        copyBtn.style.marginLeft = 'auto';
        copyBtn.innerHTML = '<i data-lucide="copy"></i>';
        copyBtn.title = 'Copy source';
        copyBtn.addEventListener('click', async (ev) => {
          ev.stopPropagation();
          try { await navigator.clipboard.writeText(s.content || ''); this.showToast('Copied!'); } catch { this.showToast('Copy failed'); }
        });
        header.appendChild(copyBtn);
        item.appendChild(header);

        if (s.content) {
          const code = document.createElement('div');
          code.className = 'source-code';
          code.textContent = s.content.length > 300 ? s.content.substring(0, 300) + '…' : s.content;
          item.appendChild(code);
        }
        sources.appendChild(item);
      });
      content.appendChild(sources);
    }

    const ts = document.createElement('div');
    ts.className = 'message-timestamp';
    ts.textContent = new Date(m.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    content.appendChild(ts);

    wrap.appendChild(avatar);
    wrap.appendChild(content);
    return wrap;
  }

  createGeneratingElement() {
    const d = document.createElement('div');
    d.className = 'generating-message';
    d.innerHTML = `
      <div class="typing-dots">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
      <div class="generating-info">
        <span class="generating-label">AI is thinking…</span>
        <span class="generating-timer">${Math.floor(this.state.elapsedMs / 1000)}s</span>
      </div>`;
    return d;
  }

  startElapsedTimer() {
    this.state.elapsedMs = 0;
    this.state.elapsedTimer = setInterval(() => {
      this.state.elapsedMs += 1000;
      const el = document.querySelector('.generating-timer');
      if (el) el.textContent = `${Math.floor(this.state.elapsedMs / 1000)}s`;
    }, 1000);
  }
  stopElapsedTimer() {
    if (this.state.elapsedTimer) { clearInterval(this.state.elapsedTimer); this.state.elapsedTimer = null; }
  }

  // ── Conversations ─────────────────────────────────────────
  createNewConversation() {
    const c = { id: `conv-${Date.now()}`, title: 'New Chat', messages: [], createdAt: new Date().toISOString(), lastActivity: new Date().toISOString() };
    this.state.conversations.push(c);
    this.state.currentConversation = c;
    this.state.messages = [];
    this.state.conversationId = c.id;
    this.updateConversationTabs();
    this.displayMessages();
    this.updateChatInterface();
    document.getElementById('chat-input')?.focus();
  }

  updateConversationTabs() {
    const cont = document.getElementById('conversation-tabs');
    const list = document.getElementById('tabs-list');
    if (this.state.conversations.length === 0) { cont.classList.add('hidden'); return; }
    cont.classList.remove('hidden');
    list.innerHTML = '';
    this.state.conversations.forEach(c => {
      const el = document.createElement('div');
      el.className = `conversation-tab ${c.id === this.state.currentConversation?.id ? 'active' : ''}`;
      const title = this.generateConversationTitle(c);
      el.innerHTML = `<i data-lucide="message-square"></i><span class="conversation-tab-title">${title}</span>${this.state.conversations.length > 1 ? `<button class="conversation-tab-close" data-cid="${c.id}"><i data-lucide="x"></i></button>` : ''}`;
      el.addEventListener('click', () => this.switchToConversation(c.id));
      const close = el.querySelector('.conversation-tab-close');
      if (close) close.addEventListener('click', (e) => { e.stopPropagation(); this.closeConversation(c.id); });
      list.appendChild(el);
    });
    this.icons();
  }

  generateConversationTitle(c) {
    if (c.title !== 'New Chat') return c.title;
    const first = c.messages.find(m => m.role === 'user');
    if (first) { const t = first.content.slice(0, 28); return t.length < first.content.length ? `${t}…` : t; }
    return 'New Chat';
  }
  switchToConversation(id) {
    const c = this.state.conversations.find(x => x.id === id);
    if (c) { this.state.currentConversation = c; this.state.messages = c.messages; this.state.conversationId = c.id; this.updateConversationTabs(); this.displayMessages(); }
  }
  closeConversation(id) {
    if (this.state.conversations.length <= 1) return;
    if (this.state.currentConversation?.id === id) {
      const other = this.state.conversations.find(x => x.id !== id);
      if (other) this.switchToConversation(other.id);
    }
    this.state.conversations = this.state.conversations.filter(x => x.id !== id);
    this.updateConversationTabs();
  }
  clearCurrentChat() {
    if (!confirm('Clear current conversation?')) return;
    this.state.messages = [];
    this.state.conversationId = null;
    this.displayMessages();
    this.hideError('chat-error');
  }

  // ── Welcome chips ─────────────────────────────────────────
  bindWelcomeChips() {
    document.querySelectorAll('.welcome-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const query = chip.dataset.query;
        const input = document.getElementById('chat-input');
        if (input && !input.disabled) {
          input.value = query;
          input.focus();
        }
      });
    });
  }

  // ── Scroll to bottom ──────────────────────────────────────
  setupScrollToBottom() {
    const chatContainer = document.querySelector('.chat-container');
    if (!chatContainer) return;
    let btn = document.getElementById('scroll-bottom-btn');
    if (!btn) {
      btn = document.createElement('button');
      btn.id = 'scroll-bottom-btn';
      btn.className = 'scroll-bottom-btn';
      btn.innerHTML = '<i data-lucide="chevrons-down"></i><span>Scroll</span>';
      btn.addEventListener('click', () => {
        const box = document.getElementById('chat-messages');
        if (box) box.scrollTop = box.scrollHeight;
      });
      chatContainer.appendChild(btn);
      this.icons();
    }
    const box = document.getElementById('chat-messages');
    if (!box) return;
    box.addEventListener('scroll', () => {
      const nearBottom = (box.scrollHeight - box.scrollTop - box.clientHeight) < 100;
      btn.classList.toggle('show', !nearBottom);
    });
  }

  // ── Global shortcuts ──────────────────────────────────────
  setupGlobalShortcuts() {
    document.addEventListener('keydown', (e) => {
      if (e.key === '/' && this.state.route === 'chat') {
        const input = document.getElementById('chat-input');
        if (input && !input.disabled && document.activeElement !== input) { e.preventDefault(); input.focus(); }
      }
      if (e.key === 'Escape') this.closeMobileSidebar();
    });
  }

  // ── Events ────────────────────────────────────────────────
  bindEvents() {
    document.getElementById('theme-toggle')?.addEventListener('click', () => this.toggleTheme());
    document.getElementById('search-form')?.addEventListener('submit', (e) => { e.preventDefault(); this.handleSearch(); });
    document.getElementById('clear-selection')?.addEventListener('click', () => this.clearSelection());
    document.getElementById('start-indexing')?.addEventListener('click', () => this.handleStartIndexing());
    document.getElementById('clear-index')?.addEventListener('click', () => this.handleClearIndex());
    document.getElementById('start-querying')?.addEventListener('click', () => this.switchPanel('chat'));
    document.getElementById('chat-form')?.addEventListener('submit', (e) => { e.preventDefault(); this.handleChatSubmit(); });
    document.getElementById('new-chat-btn')?.addEventListener('click', () => this.createNewConversation());
    document.getElementById('new-chat-header')?.addEventListener('click', () => this.createNewConversation());
    document.getElementById('clear-chat')?.addEventListener('click', () => this.clearCurrentChat());
    document.getElementById('chat-input')?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this.handleChatSubmit(); }
    });
  }

  // ── Theme ─────────────────────────────────────────────────
  initTheme() {
    try {
      const stored = localStorage.getItem('rag.theme');
      this.state.theme = (stored === 'light' || stored === 'dark') ? stored : 'system';
      this.applyTheme(this.state.theme);
    } catch { this.applyTheme('system'); }
  }
  getSystemTheme() { return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'; }
  applyTheme(mode) {
    const root = document.documentElement;
    const btn = document.getElementById('theme-toggle');
    const label = document.querySelector('.theme-switch-label');
    let effective = 'light';
    if (mode === 'dark') {
      root.setAttribute('data-theme', 'dark'); btn?.setAttribute('data-mode', 'dark'); effective = 'dark';
    } else if (mode === 'light') {
      root.setAttribute('data-theme', 'light'); btn?.setAttribute('data-mode', 'light'); effective = 'light';
    } else {
      root.removeAttribute('data-theme');
      const sys = this.getSystemTheme();
      btn?.setAttribute('data-mode', sys); effective = sys;
    }
    if (label) label.textContent = effective.charAt(0).toUpperCase() + effective.slice(1);
    this.icons();
  }
  toggleTheme() {
    const current = this.state.theme === 'system' ? this.getSystemTheme() : this.state.theme;
    const next = current === 'dark' ? 'light' : 'dark';
    this.state.theme = next;
    try { localStorage.setItem('rag.theme', next); } catch {}
    this.applyTheme(next);
  }
}

document.addEventListener('DOMContentLoaded', () => { window.ragApp = new RAGApp(); });
if (typeof module !== 'undefined' && module.exports) { module.exports = RAGApp; }
