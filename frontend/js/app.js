let FIELD_META = {};
const VALID_AGG = { Q: ['mean', 'max', 'min', 'count', 'bin'], N: ['count', 'none'], T: ['count', 'bin', 'none'] };
const BADGE_CLASS = {
    distribution: 'badge-distribution',
    trend: 'badge-trend',
    correlation: 'badge-correlation',
    'top/bottom k': 'badge-topbottom',
    'co-correlation': 'badge-co-correlation',
    comparison: 'badge-comparison',
};

let isDark = true;
let editMode = false;
let chartInstances = {};
let recChartInstances = [];
let uploadedRows = [];
let currentCharts = [];
let recommendations = [];
let topics = [];
let attributes = [];
let currentDatasetName = null;

function showToast(msg, type = 'info') {
    const c = document.getElementById('toastContainer');
    const t = document.createElement('div');
    t.className = `toast toast-${type}`;
    const icons = { success: 'fa-check-circle', info: 'fa-info-circle', warn: 'fa-exclamation-triangle' };
    t.innerHTML = `<i class="fas ${icons[type] || icons.info}"></i> ${msg}`;
    c.appendChild(t);
    setTimeout(() => { t.style.opacity = '0'; t.style.transform = 'translateX(20px)'; t.style.transition = 'all 0.3s'; }, 2800);
    setTimeout(() => t.remove(), 3200);
}

function chartColors() {
    return {
        grid: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)',
        tick: isDark ? '#94A3B8' : '#64748B',
        tooltipBg: isDark ? '#1E293B' : '#FFFFFF',
        tooltipText: isDark ? '#E2E8F0' : '#1E293B',
    };
}

function toNumber(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
}

function inferType(values, key) {
    if (/(date|time|year|month|day)/i.test(key)) return 'T';
    const present = values.filter(v => v !== '' && v != null);
    if (!present.length) return 'N';
    const numeric = present.filter(v => toNumber(v) !== null).length;
    return numeric / present.length >= 0.85 ? 'Q' : 'N';
}

function displayName(key) {
    return String(key || '').replaceAll('_', ' ');
}

function formatBytes(bytes) {
    if (!bytes) return '0 KB';
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function splitCSVLine(line) {
    const cells = [];
    let current = '';
    let quoted = false;
    for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (ch === '"' && line[i + 1] === '"') { current += '"'; i++; }
        else if (ch === '"') quoted = !quoted;
        else if (ch === ',' && !quoted) { cells.push(current); current = ''; }
        else current += ch;
    }
    cells.push(current);
    return cells;
}

function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/).filter(Boolean);
    if (lines.length < 2) return [];
    const headers = splitCSVLine(lines[0]).map(h => h.trim());
    return lines.slice(1).map(line => {
        const cells = splitCSVLine(line);
        const row = {};
        headers.forEach((h, i) => {
            const raw = (cells[i] ?? '').trim();
            row[h] = raw !== '' && !Number.isNaN(Number(raw)) ? Number(raw) : raw;
        });
        return row;
    });
}

function setEmptyState() {
    FIELD_META = {};
    uploadedRows = [];
    currentCharts = [];
    recommendations = [];
    topics = [];
    attributes = [];
    currentDatasetName = null;
    updateDatasetStats([], [], null);
    renderAttributes();
    renderTopics();
    renderPreviewTable();
    renderDashboardCharts();
    renderRecommendations();
}

function updateDatasetStats(rows, columns, file) {
    document.getElementById('rowCount').textContent = rows.length;
    document.getElementById('columnCount').textContent = columns.length;
    document.getElementById('dataHealth').textContent = rows.length ? 'Ready' : '0%';
    document.getElementById('datasetFileName').textContent = file?.name || currentDatasetName || 'No file uploaded';
    document.getElementById('datasetFileSize').textContent = file ? formatBytes(file.size) : '0 KB';
    document.getElementById('datasetCrumb').textContent = currentDatasetName || 'No dataset';
}

function updateMetadata(rows, profile, file = null) {
    const columns = profile?.columns?.length
        ? profile.columns
        : Object.keys(rows[0] || {}).map((name, index) => ({ name, index, type: inferType(rows.map(r => r[name]), name), cardinality: new Set(rows.map(r => r[name])).size }));
    FIELD_META = {};
    attributes = columns.map(c => {
        FIELD_META[c.name] = { type: c.type, label: `${c.name} (${c.type})` };
        return { name: displayName(c.name), key: c.name, type: c.type, desc: `${c.cardinality ?? ''} unique` };
    });
    updateEditorOptions(columns);
    updateDatasetStats(rows, columns, file);
}

function updateEditorOptions(columns) {
    const fields = columns.map(c => `<option value="${c.name}">${c.name} (${c.type})</option>`).join('');
    ['xAxis', 'yAxis', 'colorField'].forEach(id => {
        const select = document.getElementById(id);
        if (!select) return;
        select.innerHTML = (id === 'colorField' ? '<option value="none">None</option>' : '') + fields;
    });
    if (!columns.length) return;
    const q = columns.find(c => c.type === 'Q')?.name || columns[0].name;
    const n = columns.find(c => c.type === 'N')?.name || columns[0].name;
    document.getElementById('xAxis').value = n;
    document.getElementById('yAxis').value = q;
    applyConstrainedSampling();
}

function renderAttributes() {
    const tc = { Q: 'bg-blue-100 dark:bg-blue-900/40 text-blue-600 dark:text-blue-400', N: 'bg-emerald-100 dark:bg-emerald-900/40 text-emerald-600 dark:text-emerald-400', T: 'bg-sky-100 dark:bg-sky-900/40 text-sky-600 dark:text-sky-400' };
    document.getElementById('attributeList').innerHTML = attributes.length
        ? attributes.map(a => `<div class="flex items-center gap-2 px-2.5 py-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800/50 cursor-pointer transition group"><span class="w-5 h-5 rounded text-[10px] font-bold flex items-center justify-center ${tc[a.type]}">${a.type}</span><span class="text-sm font-medium flex-1">${a.name}</span><span class="text-[10px] text-slate-400 opacity-0 group-hover:opacity-100 transition">${a.desc || ''}</span></div>`).join('')
        : '<div class="text-xs text-slate-400 px-2 py-3">Upload CSV de xem schema.</div>';
}

function renderTopics() {
    document.getElementById('topicList').innerHTML = topics.length
        ? topics.map((t, i) => `<div class="topic-card glass-strong rounded-xl p-3.5 cursor-pointer transition-all hover:shadow-lg ${t.active ? 'active' : ''}" onclick="selectTopic(${i})"><div class="flex items-start justify-between mb-2"><h4 class="text-sm font-bold leading-snug pr-2">${t.title}</h4><div class="flex items-center gap-1 text-xs">${t.charts.map(c => `<i class="fas ${c} text-slate-400 text-[10px]"></i>`).join('')}</div></div><div class="flex items-center gap-2 mb-2.5"><div class="flex-1 h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden"><div class="h-full rounded-full reward-bar transition-all duration-700" style="width:${Math.min(100, Math.round(t.score * 30))}%"></div></div><span class="text-xs font-bold text-brand-600 dark:text-brand-400 whitespace-nowrap">Return: ${Number(t.score).toFixed(2)}</span></div><div class="flex flex-wrap gap-1">${t.badges.map(b => `<span class="${BADGE_CLASS[b] || 'badge-correlation'} text-[10px] font-bold px-1.5 py-0.5 rounded-md">${b}</span>`).join('')}</div></div>`).join('')
        : '<div class="glass-strong rounded-xl p-4 text-sm text-slate-400">Chua co recommendation. Hay upload CSV.</div>';
}

function renderPreviewTable() {
    const rows = uploadedRows.slice(0, 30);
    const cols = Object.keys(rows[0] || {});
    const head = document.querySelector('#previewModal thead tr');
    if (head) head.innerHTML = '<th class="px-4 py-2 text-left text-xs font-semibold text-slate-400">#</th>' + cols.map(c => `<th class="px-4 py-2 text-left text-xs font-semibold text-slate-400">${c}</th>`).join('');
    document.getElementById('previewTableBody').innerHTML = rows.length
        ? rows.map((r, i) => `<tr class="border-t border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/40 transition"><td class="px-4 py-2.5 text-slate-400 text-xs">${i + 1}</td>${cols.map(c => `<td class="px-4 py-2.5 text-sm">${r[c] ?? ''}</td>`).join('')}</tr>`).join('')
        : '<tr><td class="px-4 py-6 text-sm text-slate-400" colspan="4">No dataset uploaded.</td></tr>';
}

function renderCanvasTitle(key = null) {
    const title = document.getElementById('canvasTitle');
    const sub = document.getElementById('canvasSubtitle');
    if (!currentDatasetName) {
        title.textContent = 'Upload a CSV to generate a dashboard';
        sub.textContent = 'Backend se profile du lieu, phat hien insight va sinh dashboard sau khi upload.';
        return;
    }
    title.innerHTML = `Insights about <span class="bg-gradient-to-r from-brand-500 to-teal-400 bg-clip-text text-transparent">${displayName(key || 'Dashboard')}</span> in ${currentDatasetName}`;
    sub.textContent = 'Phan tich phan phoi, tuong quan va xu huong - sinh boi DashBot backend.';
}

function updateKPIs(keyColumn = null) {
    const qKey = keyColumn && FIELD_META[keyColumn]?.type === 'Q' ? keyColumn : Object.keys(FIELD_META).find(k => FIELD_META[k].type === 'Q');
    const values = uploadedRows.map(r => toNumber(r[qKey])).filter(v => v !== null);
    const uniqueKey = keyColumn || Object.keys(FIELD_META)[0];
    const unique = new Set(uploadedRows.map(r => r[uniqueKey])).size;
    const mean = values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
    const max = values.length ? Math.max(...values) : 0;
    const min = values.length ? Math.min(...values) : 0;
    const els = document.querySelectorAll('.kpi-value');
    [mean, max, min, unique].forEach((v, i) => { if (els[i]) { els[i].dataset.target = String(Number(v.toFixed ? v.toFixed(1) : v)); els[i].textContent = Number(v.toFixed ? v.toFixed(1) : v); } });
}

function renderDashboardCharts() {
    Object.values(chartInstances).forEach(c => c.destroy());
    chartInstances = {};
    const grid = document.getElementById('chartGrid');
    if (!currentCharts.length) {
        grid.innerHTML = '<div class="col-span-3 glass-strong rounded-xl p-10 text-center text-slate-400"><div class="w-12 h-12 mx-auto mb-3 rounded-xl bg-brand-100 dark:bg-brand-900/40 flex items-center justify-center"><i class="fas fa-cloud-arrow-up text-brand-500"></i></div><p class="font-semibold text-slate-600 dark:text-slate-300">No dashboard yet</p><p class="text-sm mt-1">Upload CSV de DashBot sinh bieu do.</p></div>';
        return;
    }
    grid.innerHTML = currentCharts.map((chart, i) => {
        const badge = chart.insight_type || inferInsightLabel(chart);
        return `<div class="chart-card glass-strong rounded-xl p-4 anim-fade-up"><div class="flex items-center justify-between mb-3"><h4 class="text-sm font-bold leading-tight pr-2">${chart.title || chartTitle(chart)}</h4><span class="${BADGE_CLASS[badge] || 'badge-correlation'} text-[10px] font-bold px-2 py-0.5 rounded-full whitespace-nowrap">${badge}</span></div><div class="h-44"><canvas id="chart${i}"></canvas></div></div>`;
    }).join('');
    currentCharts.forEach((chart, i) => { chartInstances[`chart${i}`] = new Chart(document.getElementById(`chart${i}`), chartToChartJs(chart, false)); });
}

function renderRecommendations() {
    const container = document.getElementById('recContainer');
    container.innerHTML = '';
    recChartInstances.forEach(c => c.destroy());
    recChartInstances = [];
    if (!recommendations.length) {
        container.innerHTML = '<div class="glass-strong rounded-xl px-4 py-6 text-sm text-slate-400">Recommendations se xuat hien sau khi upload CSV.</div>';
        return;
    }
    recommendations.forEach((rec, i) => {
        const card = document.createElement('div');
        card.className = 'rec-card relative flex-shrink-0 w-44 glass-strong rounded-xl p-3 cursor-pointer transition hover:shadow-lg';
        card.innerHTML = `<div class="relative"><canvas id="recChart${i}" height="82"></canvas><div class="rec-overlay absolute inset-0 bg-brand-600/80 rounded-lg flex items-center justify-center"><button onclick="addToCanvas(${i})" class="w-8 h-8 bg-white rounded-full flex items-center justify-center shadow-lg hover:scale-110 transition active:scale-95"><i class="fas fa-plus text-brand-600 text-sm"></i></button></div></div><p class="text-[11px] font-semibold mt-2 leading-tight">${rec.title || chartTitle(rec)}</p>`;
        container.appendChild(card);
        recChartInstances.push(new Chart(document.getElementById(`recChart${i}`), chartToChartJs(rec, true)));
    });
}

function chartToChartJs(chart, compact = false) {
    const c = chartColors();
    const options = { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { backgroundColor: c.tooltipBg, titleColor: c.tooltipText, bodyColor: c.tooltipText } }, scales: { x: { display: !compact, grid: { display: false }, ticks: { color: c.tick, font: { size: 10 } } }, y: { display: !compact, grid: { color: c.grid }, ticks: { color: c.tick, font: { size: 10 } } } } };
    if (chart.mark === 'point') return scatterConfig(chart, options);
    if (chart.mark === 'line') return lineConfig(chart, options);
    return barConfig(chart, options);
}

function barConfig(chart, options) {
    const grouped = groupedValues(chart);
    return { type: 'bar', data: { labels: grouped.labels, datasets: [{ data: grouped.values, backgroundColor: ['#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4'], borderRadius: 6 }] }, options };
}

function lineConfig(chart, options) {
    const grouped = groupedValues(chart);
    return { type: 'line', data: { labels: grouped.labels, datasets: [{ data: grouped.values, borderColor: '#10B981', backgroundColor: 'rgba(16,185,129,0.08)', fill: true, tension: 0.35, pointRadius: 3 }] }, options };
}

function scatterConfig(chart, options) {
    const points = uploadedRows.map(r => ({ x: toNumber(r[chart.x]), y: toNumber(r[chart.y]) })).filter(p => p.x !== null && p.y !== null);
    return { type: 'scatter', data: { datasets: [{ data: points, backgroundColor: 'rgba(79,70,229,0.45)', borderColor: '#4F46E5', pointRadius: 3.5 }] }, options };
}

function groupedValues(chart) {
    if (chart.x_agg === 'bin') return histogram(uploadedRows.map(r => toNumber(r[chart.x])).filter(v => v !== null), 6);
    const groups = new Map();
    uploadedRows.forEach(r => {
        const key = r[chart.x] ?? 'Unknown';
        const value = chart.y ? toNumber(r[chart.y]) : 1;
        if (value === null) return;
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(value);
    });
    const entries = [...groups.entries()].map(([label, values]) => ({ label, value: aggregate(values, chart.y_agg || 'mean') })).sort((a, b) => String(a.label).localeCompare(String(b.label))).slice(0, 12);
    return { labels: entries.map(e => String(e.label)), values: entries.map(e => Number(e.value.toFixed ? e.value.toFixed(2) : e.value)) };
}

function histogram(values, bins = 6) {
    if (!values.length) return { labels: [], values: [] };
    const min = Math.min(...values), max = Math.max(...values), width = (max - min || 1) / bins;
    const counts = Array(bins).fill(0);
    values.forEach(v => counts[Math.min(bins - 1, Math.floor((v - min) / width))]++);
    return { labels: counts.map((_, i) => `${(min + i * width).toFixed(0)}-${(min + (i + 1) * width).toFixed(0)}`), values: counts };
}

function aggregate(values, agg) {
    if (!values.length) return 0;
    if (agg === 'count') return values.length;
    if (agg === 'max') return Math.max(...values);
    if (agg === 'min') return Math.min(...values);
    return values.reduce((a, b) => a + b, 0) / values.length;
}

function chartTitle(chart) {
    if (chart.x_agg === 'bin') return `Distribution of ${displayName(chart.x)}`;
    if (chart.mark === 'point') return `${displayName(chart.y)} vs ${displayName(chart.x)}`;
    if (chart.mark === 'line') return `${displayName(chart.y)} over ${displayName(chart.x)}`;
    return `${chart.y_agg || 'mean'} ${displayName(chart.y)} by ${displayName(chart.x)}`;
}

function inferInsightLabel(chart) {
    if (chart.x_agg === 'bin') return 'distribution';
    if (chart.mark === 'line') return 'trend';
    if (chart.mark === 'point') return 'correlation';
    return 'top/bottom k';
}

function selectTopic(index) {
    topics.forEach((t, i) => t.active = i === index);
    renderTopics();
    renderCanvasTitle(topics[index].id);
    updateKPIs(topics[index].id);
}

function toggleTheme() {
    isDark = !isDark;
    document.documentElement.classList.toggle('dark', isDark);
    document.getElementById('themeIcon').className = isDark ? 'fas fa-moon text-sm text-slate-500' : 'fas fa-sun text-sm text-amber-500';
    renderDashboardCharts();
    renderRecommendations();
}

function toggleEditMode() {
    editMode = !editMode;
    const controls = document.getElementById('editorControls'), track = document.getElementById('editToggleTrack'), thumb = document.getElementById('editToggleThumb');
    if (editMode) { controls.classList.remove('opacity-40', 'pointer-events-none'); track.classList.add('bg-brand-500'); thumb.style.transform = 'translateX(16px)'; applyConstrainedSampling(); }
    else { controls.classList.add('opacity-40', 'pointer-events-none'); track.classList.remove('bg-brand-500'); thumb.style.transform = 'translateX(0)'; document.getElementById('constraintAlert').classList.remove('visible'); }
}

function applyConstrainedSampling() {
    const xField = document.getElementById('xAxis')?.value, yField = document.getElementById('yAxis')?.value;
    if (!xField || !yField) return;
    updateAggregateDropdown('xAgg', FIELD_META[xField]?.type || 'Q');
    updateAggregateDropdown('yAgg', FIELD_META[yField]?.type || 'Q');
}

function updateAggregateDropdown(selectId, fieldType) {
    const select = document.getElementById(selectId), valid = VALID_AGG[fieldType] || VALID_AGG.Q;
    if (!select) return;
    Array.from(select.options).forEach(opt => opt.disabled = !valid.includes(opt.value));
    if (!valid.includes(select.value)) select.value = valid[0];
}

function onAxisChange() { applyConstrainedSampling(); }

function applyChartConfig() {
    if (!uploadedRows.length) { showToast('Upload CSV truoc khi them chart.', 'warn'); return; }
    const mark = document.getElementById('markType').value, x = document.getElementById('xAxis').value, xAgg = document.getElementById('xAgg').value, y = document.getElementById('yAxis').value, yAgg = document.getElementById('yAgg').value, color = document.getElementById('colorField').value;
    const chart = { mark, x, y, x_agg: xAgg, y_agg: yAgg, color: color === 'none' ? null : color, title: `Custom: ${displayName(y)} by ${displayName(x)}`, insight_type: inferInsightLabel({ mark, x_agg: xAgg }) };
    if (currentCharts.some(existing => chartSignature(existing) === chartSignature(chart))) {
        showToast('Chart nay da co tren Canvas.', 'warn');
        return;
    }
    currentCharts.unshift(chart);
    currentCharts = uniqueCharts(currentCharts).slice(0, 8);
    renderDashboardCharts();
    showToast('Da them chart cau hinh moi vao Canvas.', 'success');
}

function addToCanvas(index) {
    if (!uploadedRows.length) { showToast('Upload CSV truoc khi them chart.', 'warn'); return; }
    const rec = recommendations[index];
    if (currentCharts.some(existing => chartSignature(existing) === chartSignature(rec))) {
        showToast('Chart nay da co tren Canvas.', 'warn');
        return;
    }
    currentCharts.unshift({ ...rec, title: rec.title || chartTitle(rec) });
    currentCharts = uniqueCharts(currentCharts).slice(0, 8);
    renderDashboardCharts();
    showToast(`Da them "${rec.title || chartTitle(rec)}" vao Canvas.`, 'success');
}

function exportDashboard(format) {
    if (!currentCharts.length) { showToast('Chua co dashboard de export.', 'warn'); return; }
    toggleExportMenu();
    const blob = new Blob([JSON.stringify({ dataset: currentDatasetName, charts: currentCharts }, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `dashbot-dashboard-${format.toLowerCase()}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
}

function toggleExportMenu() { document.getElementById('exportMenu').classList.toggle('show'); }
document.addEventListener('click', e => { const m = document.getElementById('exportMenu'); if (m.classList.contains('show') && !e.target.closest('.relative')) m.classList.remove('show'); });
function openUploadModal() { document.getElementById('uploadModal').classList.add('open'); }
function closeUploadModal() { document.getElementById('uploadModal').classList.remove('open'); }
function openPreviewModal() { document.getElementById('previewModal').classList.add('open'); }
function closePreviewModal() { document.getElementById('previewModal').classList.remove('open'); }

async function handleFileUpload(file) {
    if (!file) return;
    const p = document.getElementById('uploadProgress'), b = document.getElementById('uploadBar'), pc = document.getElementById('uploadPercent');
    p.classList.remove('hidden'); b.style.width = '20%'; pc.textContent = '20%';
    try {
        const text = await file.text();
        const rows = parseCSV(text);
        if (!rows.length) throw new Error('CSV khong co du lieu hop le');
        uploadedRows = rows;
        currentDatasetName = file.name;
        updateMetadata(rows, null, file);
        b.style.width = '50%'; pc.textContent = '50%';
        const result = await requestDashboardRecommendation(file, 5);
        b.style.width = '100%'; pc.textContent = '100%';
        applyRecommendationResult(result, file);
        setTimeout(() => { closeUploadModal(); p.classList.add('hidden'); b.style.width = '0%'; }, 300);
    } catch (err) {
        p.classList.add('hidden'); b.style.width = '0%';
        showToast(`Khong xu ly duoc upload: ${err.message}`, 'warn');
    }
}

function applyRecommendationResult(result, file) {
    updateMetadata(uploadedRows, result.profile, file);
    currentCharts = uniqueCharts(result.charts.map(chart => ({ mark: chart.mark, x: chart.x, y: chart.y, color: chart.color, x_agg: chart.x_agg, y_agg: chart.y_agg, title: chart.title || chartTitle(chart), insight_type: chart.insight_type || inferInsightLabel(chart) })));
    recommendations = currentCharts.slice(0, 4).map((chart, i) => ({ ...chart, title: chart.title || `Recommendation ${i + 1}` }));
    const badges = [...new Set(result.insights.map(i => i.type))].slice(0, 4);
    topics = [{ id: result.key_column, title: `Insights about ${displayName(result.key_column)}`, score: result.reward, badges: badges.length ? badges : ['correlation'], charts: ['fa-chart-bar', 'fa-chart-line', 'fa-braille'], active: true }];
    renderCanvasTitle(result.key_column);
    renderAttributes();
    renderTopics();
    renderPreviewTable();
    updateKPIs(result.key_column);
    renderDashboardCharts();
    renderRecommendations();
    showToast(`Da render ${currentCharts.length} chart tu ${currentDatasetName}. Return: ${result.reward.toFixed(2)}`, 'success');
}

function chartSignature(chart) {
    return [
        chart.mark || '',
        chart.x || '',
        chart.y || '',
        chart.x_agg || '',
        chart.y_agg || '',
    ].join('|').toLowerCase();
}

function uniqueCharts(charts) {
    const seen = new Set();
    const unique = [];
    charts.forEach(chart => {
        const signature = chartSignature(chart);
        if (seen.has(signature)) return;
        seen.add(signature);
        unique.push(chart);
    });
    return unique;
}

document.addEventListener('DOMContentLoaded', () => {
    setEmptyState();
    document.getElementById('uploadModal').classList.remove('open');
    document.getElementById('previewModal').classList.remove('open');
});
