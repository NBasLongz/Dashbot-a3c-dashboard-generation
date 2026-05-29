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
const CHART_PALETTE = ['#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4', '#F97316', '#14B8A6'];

let isDark = false;
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
    ['xAxis', 'yAxis'].forEach(id => {
        const select = document.getElementById(id);
        if (!select) return;
        select.innerHTML = fields;
    });
    const colorSelect = document.getElementById('colorField');
    if (colorSelect) {
        const colorFields = columns
            .filter(c => c.type === 'N' && (c.cardinality ?? 0) >= 2 && (c.cardinality ?? 0) <= 12)
            .map(c => `<option value="${c.name}">${c.name} (${c.type})</option>`)
            .join('');
        colorSelect.innerHTML = '<option value="none">None</option>' + colorFields;
    }
    if (!columns.length) return;
    const q = columns.find(c => c.type === 'Q')?.name || columns[0].name;
    const n = columns.find(c => c.type === 'N')?.name || columns[0].name;
    document.getElementById('xAxis').value = n;
    document.getElementById('yAxis').value = q;
    if (colorSelect) colorSelect.value = 'none';
    applyConstrainedSampling();
}

function renderAttributes() {
    const tc = { Q: 'bg-blue-100 dark:bg-blue-900/40 text-blue-600 dark:text-blue-400', N: 'bg-emerald-100 dark:bg-emerald-900/40 text-emerald-600 dark:text-emerald-400', T: 'bg-sky-100 dark:bg-sky-900/40 text-sky-600 dark:text-sky-400' };
    document.getElementById('attributeList').innerHTML = attributes.length
        ? attributes.map(a => `<div onclick="selectColumnAsTopic('${a.key}')" class="flex items-center gap-2 px-2.5 py-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800/50 cursor-pointer transition group"><span class="w-5 h-5 rounded text-[10px] font-bold flex items-center justify-center ${tc[a.type]}">${a.type}</span><span class="text-sm font-medium flex-1">${a.name}</span><span class="text-[10px] text-slate-400 opacity-0 group-hover:opacity-100 transition">${a.desc || ''}</span></div>`).join('')
        : '<div class="text-xs text-slate-400 px-2 py-3">Upload data to view columns.</div>';
}

function renderTopics() {
    document.getElementById('topicList').innerHTML = topics.length
        ? topics.map((t, i) => `<div class="topic-card glass-strong rounded-xl p-3.5 cursor-pointer transition-all hover:shadow-lg ${t.active ? 'active' : ''}" onclick="selectTopic(${i})"><div class="flex items-start justify-between mb-2"><h4 class="text-sm font-bold leading-snug pr-2">${t.title}</h4><div class="flex items-center gap-1 text-xs">${t.charts.map(c => `<i class="fas ${c} text-slate-400 text-[10px]"></i>`).join('')}</div></div><div class="flex items-center gap-2 mb-2.5"><div class="flex-1 h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden"><div class="h-full rounded-full reward-bar transition-all duration-700" style="width:${Math.min(100, Math.round(t.score * 30))}%"></div></div><span class="text-xs font-bold text-brand-600 dark:text-brand-400 whitespace-nowrap">Return: ${Number(t.score).toFixed(2)}</span></div><div class="flex flex-wrap gap-1">${t.badges.map(b => `<span class="${BADGE_CLASS[b] || 'badge-correlation'} text-[10px] font-bold px-1.5 py-0.5 rounded-md">${b}</span>`).join('')}</div></div>`).join('')
        : '<div class="glass-strong rounded-xl p-4 text-sm text-slate-400">No topics yet. Upload data first.</div>';
}

function renderPreviewTable() {
    const rows = uploadedRows;
    const cols = Object.keys(rows[0] || {});
    const head = document.querySelector('#previewModal thead tr');
    if (head) head.innerHTML = '<th class="px-4 py-2 text-left text-xs font-semibold text-slate-400">#</th>' + cols.map(c => `<th class="px-4 py-2 text-left text-xs font-semibold text-slate-400">${c}</th>`).join('');
    document.getElementById('previewTableBody').innerHTML = rows.length
        ? rows.map((r, i) => `<tr class="border-t border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/40 transition"><td class="px-4 py-2.5 text-slate-400 text-xs">${i + 1}</td>${cols.map(c => `<td class="px-4 py-2.5 text-sm">${r[c] ?? ''}</td>`).join('')}</tr>`).join('')
        : '<tr><td class="px-4 py-6 text-sm text-slate-400" colspan="4">No dataset uploaded.</td></tr>';
    const note = document.getElementById('previewTableNote');
    if (note) {
        note.textContent = rows.length
            ? `Showing all ${rows.length.toLocaleString()} rows and ${cols.length} columns.`
            : 'Raw data will appear after upload.';
    }
}

function renderCanvasTitle(key = null, topicTitle = null) {
    const title = document.getElementById('canvasTitle');
    const sub = document.getElementById('canvasSubtitle');
    if (!currentDatasetName) {
        title.textContent = 'Upload a CSV to generate a dashboard';
        sub.textContent = 'Upload data to generate topics, charts, and recommendations.';
        return;
    }
    if (key) {
        const keyLabel = displayName(key);
        const label = topicTitle || `Insights about ${keyLabel}`;
        title.innerHTML = `${label.replace(keyLabel, `<span class="bg-gradient-to-r from-brand-500 to-teal-400 bg-clip-text text-transparent">${keyLabel}</span>`)} in ${currentDatasetName}`;
    } else {
        title.innerHTML = `<span class="bg-gradient-to-r from-brand-500 to-teal-400 bg-clip-text text-transparent">${topicTitle || 'Overview dashboard'}</span> in ${currentDatasetName}`;
    }
    sub.textContent = 'Distribution, correlation, and trend analysis generated by DashBot.';
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
        grid.innerHTML = '<div class="col-span-3 glass-strong rounded-xl p-10 text-center text-slate-400"><div class="w-12 h-12 mx-auto mb-3 rounded-xl bg-brand-100 dark:bg-brand-900/40 flex items-center justify-center"><i class="fas fa-cloud-arrow-up text-brand-500"></i></div><p class="font-semibold text-slate-600 dark:text-slate-300">No canvas yet</p><p class="text-sm mt-1">Upload data to generate dashboard charts.</p></div>';
        return;
    }
    grid.innerHTML = currentCharts.map((chart, i) => {
        const badge = chart.insight_type || inferInsightLabel(chart);
        return `<div class="chart-card glass-strong rounded-xl p-4 anim-fade-up">
            <div class="flex items-center justify-between mb-3">
                <h4 class="text-sm font-bold leading-tight pr-2">${chart.title || chartTitle(chart)}</h4>
                <div class="flex items-center gap-1.5">
                    <span class="${BADGE_CLASS[badge] || 'badge-correlation'} text-[10px] font-bold px-2 py-0.5 rounded-full whitespace-nowrap">${badge}</span>
                    <button onclick="removeFromCanvas(${i})" class="text-slate-400 hover:text-red-500 transition duration-150 p-1 rounded hover:bg-slate-100 dark:hover:bg-slate-800" title="Xóa biểu đồ">
                        <i class="fas fa-trash-alt text-[11px]"></i>
                    </button>
                </div>
            </div>
            <div class="h-44">
                <canvas id="chart${i}"></canvas>
            </div>
        </div>`;
    }).join('');
    currentCharts.forEach((chart, i) => { chartInstances[`chart${i}`] = new Chart(document.getElementById(`chart${i}`), chartToChartJs(chart, false)); });
}

function renderRecommendations() {
    const container = document.getElementById('recContainer');
    container.innerHTML = '';
    recChartInstances.forEach(c => c.destroy());
    recChartInstances = [];
    if (!recommendations.length) {
        container.innerHTML = '<div class="glass-strong rounded-xl px-4 py-6 text-sm text-slate-400">Recommendations will appear after dashboard generation.</div>';
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
    const options = baseChartOptions(compact);
    if (chart.mark === 'boxplot') return boxplotConfig(chart, options);
    if (chart.mark === 'point') return scatterConfig(chart, options);
    if (chart.mark === 'line') return lineConfig(chart, options);
    return barConfig(chart, options);
}

function baseChartOptions(compact = false) {
    const c = chartColors();
    return {
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: { top: 2, right: 4, bottom: 0, left: 0 } },
        plugins: {
            legend: { display: false, position: 'right', labels: { color: c.tick, boxWidth: 9, font: { size: 10 } } },
            tooltip: { backgroundColor: c.tooltipBg, titleColor: c.tooltipText, bodyColor: c.tooltipText },
        },
        scales: {
            x: { display: !compact, grid: { display: false }, ticks: { color: c.tick, font: { size: 10 }, maxRotation: compact ? 0 : 35, autoSkip: true } },
            y: { display: !compact, grid: { color: c.grid }, ticks: { color: c.tick, font: { size: 10 } } },
        },
    };
}

function barConfig(chart, options) {
    const colorField = chart.color && chart.color !== chart.x ? chart.color : null;
    const series = colorField ? groupedSeries(chart, colorField, { limitLabels: 12, limitSeries: 6, sort: 'value-desc' }) : null;
    if (series && series.series.length > 1) {
        options.plugins.legend.display = Boolean(options.scales.x.display);
        return {
            type: 'bar',
            data: {
                labels: series.labels,
                datasets: series.series.map((entry, i) => ({
                    label: entry.label,
                    data: entry.values,
                    backgroundColor: CHART_PALETTE[i % CHART_PALETTE.length],
                    borderRadius: 5,
                    maxBarThickness: 22,
                })),
            },
            options,
        };
    }

    const grouped = groupedValues(chart, { sort: 'value-desc', limit: 12 });
    const horizontal = shouldUseHorizontalBar(grouped.labels, chart);
    if (horizontal) {
        options.indexAxis = 'y';
        options.scales.x.grid.display = true;
        options.scales.y.grid.display = false;
        options.scales.y.ticks.autoSkip = false;
    }
    return {
        type: 'bar',
        data: {
            labels: grouped.labels,
            datasets: [{
                data: grouped.values,
                backgroundColor: grouped.labels.map((_, i) => CHART_PALETTE[i % CHART_PALETTE.length]),
                borderRadius: 6,
                maxBarThickness: horizontal ? 18 : 28,
            }],
        },
        options,
    };
}

function lineConfig(chart, options) {
    const colorField = chart.color && chart.color !== chart.x && chart.color !== chart.y
        ? chart.color
        : autoColorField(chart, 5);
    const series = colorField ? groupedSeries(chart, colorField, { limitLabels: 24, limitSeries: 5, sort: 'x', missingValue: null }) : null;
    if (series && series.series.length > 1) {
        options.plugins.legend.display = true;
        return {
            type: 'line',
            data: {
                labels: series.labels,
                datasets: series.series.map((entry, i) => ({
                    label: entry.label,
                    data: entry.values,
                    borderColor: CHART_PALETTE[i % CHART_PALETTE.length],
                    backgroundColor: `${CHART_PALETTE[i % CHART_PALETTE.length]}22`,
                    fill: false,
                    tension: 0.28,
                    pointRadius: 2,
                    borderWidth: 2,
                })),
            },
            options,
        };
    }

    const grouped = groupedValues(chart, { sort: 'x', limit: 24 });
    return {
        type: 'line',
        data: {
            labels: grouped.labels,
            datasets: [{
                data: grouped.values,
                borderColor: '#10B981',
                backgroundColor: 'rgba(16,185,129,0.08)',
                fill: true,
                tension: 0.32,
                pointRadius: 2.5,
                borderWidth: 2.5,
            }],
        },
        options,
    };
}

function scatterConfig(chart, options) {
    const colorField = chart.color && chart.color !== chart.x && chart.color !== chart.y
        ? chart.color
        : autoColorField(chart, 6);
    if (colorField) {
        const groups = new Map();
        uploadedRows.forEach(row => {
            const x = toNumber(row[chart.x]);
            const y = toNumber(row[chart.y]);
            if (x === null || y === null) return;
            const label = normalizeLabel(row[colorField]);
            if (!groups.has(label)) groups.set(label, []);
            groups.get(label).push({ x, y });
        });
        const entries = [...groups.entries()].filter(([, points]) => points.length).slice(0, 6);
        if (entries.length > 1) {
            options.plugins.legend.display = true;
            return {
                type: 'scatter',
                data: {
                    datasets: entries.map(([label, points], i) => ({
                        label,
                        data: points,
                        backgroundColor: `${CHART_PALETTE[i % CHART_PALETTE.length]}99`,
                        borderColor: CHART_PALETTE[i % CHART_PALETTE.length],
                        pointRadius: 3,
                    })),
                },
                options,
            };
        }
    }
    const points = uploadedRows.map(r => ({ x: toNumber(r[chart.x]), y: toNumber(r[chart.y]) })).filter(p => p.x !== null && p.y !== null);
    return { type: 'scatter', data: { datasets: [{ data: points, backgroundColor: 'rgba(79,70,229,0.45)', borderColor: '#4F46E5', pointRadius: 3.5 }] }, options };
}

function boxplotConfig(chart, options) {
    const orientation = boxplotOrientation(chart);
    const horizontal = orientation === 'horizontal';
    const stats = boxplotStats(chart, { limit: options.scales.x.display ? 12 : 6, orientation });
    options.plugins.legend.display = false;
    options.plugins.dashbotBoxplot = { stats, orientation, compact: !options.scales.x.display };
    if (horizontal) {
        options.indexAxis = 'y';
        options.scales.x.type = 'linear';
        options.scales.x.grid.display = true;
        options.scales.y.type = 'category';
        options.scales.y.grid.display = false;
        options.scales.y.ticks.autoSkip = false;
    } else {
        options.scales.x.type = 'category';
        options.scales.x.grid.display = false;
        options.scales.x.ticks.autoSkip = false;
        options.scales.y.type = 'linear';
    }
    return {
        type: 'bar',
        data: {
            labels: stats.map(s => s.label),
            datasets: [{
                label: 'boxplot-scale',
                data: stats.map(s => horizontal ? [s.whiskerLow, s.whiskerHigh] : [s.whiskerLow, s.whiskerHigh]),
                backgroundColor: 'rgba(0,0,0,0)',
                borderColor: 'rgba(0,0,0,0)',
                hoverBackgroundColor: 'rgba(0,0,0,0)',
                borderSkipped: false,
            }],
        },
        options,
        plugins: [DASHBOT_BOXPLOT_PLUGIN],
    };
}

const DASHBOT_BOXPLOT_PLUGIN = {
    id: 'dashbotBoxplot',
    afterDatasetsDraw(chart, _args, pluginOptions) {
        const stats = pluginOptions.stats || [];
        if (!stats.length) return;
        const horizontal = pluginOptions.orientation === 'horizontal';
        const compact = Boolean(pluginOptions.compact);
        const { ctx, scales } = chart;
        const categoryScale = horizontal ? scales.y : scales.x;
        const valueScale = horizontal ? scales.x : scales.y;
        const boxThickness = Math.max(compact ? 8 : 14, Math.min(compact ? 14 : 26, (horizontal ? chart.chartArea.height : chart.chartArea.width) / stats.length * 0.42));
        const capSize = boxThickness * 0.72;
        const lineColor = isDark ? '#94A3B8' : '#64748B';
        const medianColor = isDark ? '#F8FAFC' : '#111827';

        ctx.save();
        ctx.lineWidth = compact ? 1.2 : 1.6;
        stats.forEach((s, i) => {
            const center = categoryScale.getPixelForValue(i);
            const color = CHART_PALETTE[i % CHART_PALETTE.length];
            const q1 = valueScale.getPixelForValue(s.q1);
            const q3 = valueScale.getPixelForValue(s.q3);
            const median = valueScale.getPixelForValue(s.median);
            const low = valueScale.getPixelForValue(s.whiskerLow);
            const high = valueScale.getPixelForValue(s.whiskerHigh);

            ctx.strokeStyle = lineColor;
            ctx.fillStyle = `${color}55`;
            if (horizontal) {
                const boxLeft = Math.min(q1, q3);
                const boxRight = Math.max(q1, q3);
                const whiskerLeft = Math.min(low, high);
                const whiskerRight = Math.max(low, high);
                ctx.beginPath();
                ctx.moveTo(whiskerLeft, center);
                ctx.lineTo(boxLeft, center);
                ctx.moveTo(boxRight, center);
                ctx.lineTo(whiskerRight, center);
                ctx.moveTo(whiskerLeft, center - capSize / 2);
                ctx.lineTo(whiskerLeft, center + capSize / 2);
                ctx.moveTo(whiskerRight, center - capSize / 2);
                ctx.lineTo(whiskerRight, center + capSize / 2);
                ctx.stroke();
                ctx.fillRect(boxLeft, center - boxThickness / 2, Math.max(2, boxRight - boxLeft), boxThickness);
                ctx.strokeStyle = color;
                ctx.strokeRect(boxLeft, center - boxThickness / 2, Math.max(2, boxRight - boxLeft), boxThickness);
                ctx.strokeStyle = medianColor;
                ctx.beginPath();
                ctx.moveTo(median, center - boxThickness / 2);
                ctx.lineTo(median, center + boxThickness / 2);
                ctx.stroke();
                drawOutliers(ctx, s.outliers, valueScale, center, true, color, compact);
            } else {
                const boxTop = Math.min(q1, q3);
                const boxBottom = Math.max(q1, q3);
                const whiskerTop = Math.min(low, high);
                const whiskerBottom = Math.max(low, high);
                ctx.beginPath();
                ctx.moveTo(center, whiskerTop);
                ctx.lineTo(center, boxTop);
                ctx.moveTo(center, boxBottom);
                ctx.lineTo(center, whiskerBottom);
                ctx.moveTo(center - capSize / 2, whiskerTop);
                ctx.lineTo(center + capSize / 2, whiskerTop);
                ctx.moveTo(center - capSize / 2, whiskerBottom);
                ctx.lineTo(center + capSize / 2, whiskerBottom);
                ctx.stroke();
                ctx.fillRect(center - boxThickness / 2, boxTop, boxThickness, Math.max(2, boxBottom - boxTop));
                ctx.strokeStyle = color;
                ctx.strokeRect(center - boxThickness / 2, boxTop, boxThickness, Math.max(2, boxBottom - boxTop));
                ctx.strokeStyle = medianColor;
                ctx.beginPath();
                ctx.moveTo(center - boxThickness / 2, median);
                ctx.lineTo(center + boxThickness / 2, median);
                ctx.stroke();
                drawOutliers(ctx, s.outliers, valueScale, center, false, color, compact);
            }
        });
        ctx.restore();
    },
};

function drawOutliers(ctx, outliers, valueScale, center, horizontal, color, compact) {
    const radius = compact ? 2 : 3;
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.strokeStyle = color;
    ctx.lineWidth = compact ? 1 : 1.25;
    outliers.slice(0, compact ? 8 : 20).forEach(value => {
        const point = valueScale.getPixelForValue(value);
        ctx.beginPath();
        ctx.arc(horizontal ? point : center, horizontal ? center : point, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    });
}

function groupedValues(chart, config = {}) {
    if (chart.x_agg === 'bin') return histogram(uploadedRows.map(r => toNumber(r[chart.x])).filter(v => v !== null), 6);
    const groups = new Map();
    uploadedRows.forEach(r => {
        const key = normalizeLabel(r[chart.x]);
        const value = chart.y ? toNumber(r[chart.y]) : 1;
        if (value === null) return;
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(value);
    });
    const entries = [...groups.entries()]
        .map(([label, values]) => ({ label, value: aggregate(values, chart.y_agg || 'mean') }))
        .sort((a, b) => sortEntries(a, b, chart.x, config.sort || 'label'))
        .slice(0, config.limit || 12);
    return { labels: entries.map(e => String(e.label)), values: entries.map(e => Number(e.value.toFixed ? e.value.toFixed(2) : e.value)) };
}

function boxplotOrientation(chart) {
    const xType = FIELD_META[chart.x]?.type;
    const yType = FIELD_META[chart.y]?.type;
    const xUnique = uniqueValueCount(chart.x);
    const yUnique = uniqueValueCount(chart.y);
    if (xType === 'Q' && yType !== 'Q') return 'horizontal';
    if (xType === 'Q' && yType === 'Q' && xUnique > yUnique && yUnique <= 20) return 'horizontal';
    return 'vertical';
}

function boxplotStats(chart, config = {}) {
    const groups = new Map();
    const horizontal = config.orientation === 'horizontal';
    const categoryField = horizontal ? chart.y : chart.x;
    const valueField = horizontal ? chart.x : chart.y;
    const categoryMeta = FIELD_META[categoryField] || {};
    const shouldBinCategory = categoryMeta.type === 'Q' && uniqueValueCount(categoryField) > 12;
    if (shouldBinCategory) {
        const pairs = uploadedRows
            .map(row => ({ category: toNumber(row[categoryField]), value: toNumber(row[valueField]) }))
            .filter(pair => pair.category !== null && pair.value !== null);
        if (!pairs.length) return [];
        const xs = pairs.map(pair => pair.category);
        const min = Math.min(...xs), max = Math.max(...xs), bins = Math.min(8, config.limit || 8);
        const width = (max - min || 1) / bins;
        pairs.forEach(pair => {
            const index = Math.min(bins - 1, Math.floor((pair.category - min) / width));
            const start = min + index * width;
            const end = min + (index + 1) * width;
            const label = `${start.toFixed(0)}-${end.toFixed(0)}`;
            if (!groups.has(label)) groups.set(label, []);
            groups.get(label).push(pair.value);
        });
    } else {
        uploadedRows.forEach(row => {
            const value = toNumber(row[valueField]);
            if (value === null) return;
            const label = normalizeLabel(row[categoryField]);
            if (!groups.has(label)) groups.set(label, []);
            groups.get(label).push(value);
        });
    }

    return [...groups.entries()]
        .map(([label, values]) => ({ label, ...fiveNumberSummary(values) }))
        .filter(item => item.count > 0)
        .sort((a, b) => b.median - a.median)
        .slice(0, config.limit || 12);
}

function fiveNumberSummary(values) {
    const sorted = values.slice().sort((a, b) => a - b);
    const q1 = quantile(sorted, 0.25);
    const median = quantile(sorted, 0.5);
    const q3 = quantile(sorted, 0.75);
    const iqr = q3 - q1;
    const lowerFence = q1 - 1.5 * iqr;
    const upperFence = q3 + 1.5 * iqr;
    const nonOutliers = sorted.filter(value => value >= lowerFence && value <= upperFence);
    return {
        min: sorted[0],
        q1,
        median,
        q3,
        max: sorted[sorted.length - 1],
        whiskerLow: nonOutliers[0] ?? sorted[0],
        whiskerHigh: nonOutliers[nonOutliers.length - 1] ?? sorted[sorted.length - 1],
        outliers: sorted.filter(value => value < lowerFence || value > upperFence),
        count: sorted.length,
    };
}

function quantile(sortedValues, p) {
    if (!sortedValues.length) return 0;
    const pos = (sortedValues.length - 1) * p;
    const base = Math.floor(pos);
    const rest = pos - base;
    return sortedValues[base + 1] !== undefined
        ? sortedValues[base] + rest * (sortedValues[base + 1] - sortedValues[base])
        : sortedValues[base];
}

function groupedSeries(chart, colorField, config = {}) {
    const groups = new Map();
    uploadedRows.forEach(row => {
        const xLabel = normalizeLabel(row[chart.x]);
        const seriesLabel = normalizeLabel(row[colorField]);
        const value = chart.y ? toNumber(row[chart.y]) : 1;
        if (value === null) return;
        if (!groups.has(xLabel)) groups.set(xLabel, new Map());
        const seriesMap = groups.get(xLabel);
        if (!seriesMap.has(seriesLabel)) seriesMap.set(seriesLabel, []);
        seriesMap.get(seriesLabel).push(value);
    });

    let labelEntries = [...groups.entries()].map(([label, seriesMap]) => ({
        label,
        total: [...seriesMap.values()].flat().reduce((sum, value) => sum + value, 0),
        seriesMap,
    }));
    labelEntries = labelEntries
        .sort((a, b) => config.sort === 'x' ? sortEntries(a, b, chart.x, 'x') : b.total - a.total)
        .slice(0, config.limitLabels || 12);
    const labels = labelEntries.map(entry => String(entry.label));

    const seriesTotals = new Map();
    labelEntries.forEach(entry => {
        entry.seriesMap.forEach((values, label) => {
            seriesTotals.set(label, (seriesTotals.get(label) || 0) + values.length);
        });
    });
    const seriesLabels = [...seriesTotals.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, config.limitSeries || 6)
        .map(([label]) => label);

    if (seriesLabels.length < 2) return null;
    const series = seriesLabels.map(seriesLabel => ({
        label: String(seriesLabel),
        values: labelEntries.map(entry => {
            const values = entry.seriesMap.get(seriesLabel) || [];
            if (!values.length) return config.missingValue ?? 0;
            return Number(aggregate(values, chart.y_agg || 'mean').toFixed(2));
        }),
    }));
    return { labels, series };
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

function normalizeLabel(value) {
    if (value == null || value === '' || value === '.') return 'Unknown';
    return String(value);
}

function sortEntries(a, b, field, mode) {
    if (mode === 'value-desc') return b.value - a.value;
    if (mode === 'x') {
        const type = FIELD_META[field]?.type;
        if (type === 'T') return new Date(a.label) - new Date(b.label);
        if (type === 'Q') return Number(a.label) - Number(b.label);
    }
    return String(a.label).localeCompare(String(b.label));
}

function shouldUseHorizontalBar(labels, chart) {
    return chart.mark === 'bar' && (labels.some(label => String(label).length > 12) || labels.length > 7);
}

function uniqueValueCount(field) {
    return new Set(uploadedRows.map(row => normalizeLabel(row[field]))).size;
}

function autoColorField(chart, maxCardinality = 6) {
    const candidates = Object.keys(FIELD_META).filter(field => {
        if (field === chart.x || field === chart.y || field === chart.color) return false;
        if (FIELD_META[field]?.type !== 'N') return false;
        const values = new Set(uploadedRows.map(row => normalizeLabel(row[field])).filter(Boolean));
        return values.size >= 2 && values.size <= maxCardinality;
    });
    return candidates[0] || null;
}

function chartTitle(chart) {
    if (chart.mark === 'boxplot') return `${displayName(chart.y)} distribution by ${displayName(chart.x)}`;
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
    const oldKey = topics.find(t => t.active)?.id;
    const topic = topics[index];
    if (!topic) return;
    const newKey = topic.id;
    topics.forEach((t, i) => t.active = i === index);
    if (topic.dashboardCharts?.length) {
        currentCharts = uniqueCharts(topic.dashboardCharts.map(chart => ({ ...chart, title: chart.title || chartTitle(chart) })));
    }
    recommendations = uniqueCharts((topic.recommendations || []).map(chart => ({ ...chart, title: chart.title || chartTitle(chart) })))
        .filter(chart => !currentCharts.some(existing => isSameAnalysis(existing, chart)))
        .slice(0, 4);
    renderTopics();
    renderCanvasTitle(newKey, topic.title);
    updateKPIs(newKey);
    renderDashboardCharts();
    renderRecommendations();
    return;

    if (oldKey && oldKey !== newKey) {
        let changedAny = false;
        currentCharts.forEach(chart => {
            let changed = false;
            if (chart.x === oldKey) { chart.x = newKey; changed = true; }
            if (chart.y === oldKey) { chart.y = newKey; changed = true; }
            if (chart.color === oldKey) { chart.color = newKey; changed = true; }
            if (changed) {
                chart.title = chartTitle(chart);
                changedAny = true;
            }
        });
        if (changedAny) {
            currentCharts = uniqueCharts(currentCharts);
            currentCharts = currentCharts.filter(chart => !(chart.x && chart.y && chart.x === chart.y && chart.x_agg !== 'bin'));
            renderDashboardCharts();
            showToast(`Đã thay thế cột khóa "${oldKey}" thành "${newKey}" trên các biểu đồ.`, 'success');
        }
    }
}

function selectColumnAsTopic(key) {
    if (!key) return;
    const oldKey = topics.find(t => t.active)?.id;
    if (oldKey === key) return;

    const existingIndex = topics.findIndex(t => t.id === key);
    if (existingIndex !== -1) {
        selectTopic(existingIndex);
    } else {
        topics.forEach(t => t.active = false);
        topics.unshift({
            id: key,
            title: `Insights about ${displayName(key)}`,
            score: 1.0,
            badges: ['distribution'],
            charts: ['fa-chart-bar', 'fa-chart-line'],
            active: true
        });
        renderTopics();
        renderCanvasTitle(key);
        updateKPIs(key);
        return;

        if (oldKey) {
            let changedAny = false;
            currentCharts.forEach(chart => {
                let changed = false;
                if (chart.x === oldKey) { chart.x = key; changed = true; }
                if (chart.y === oldKey) { chart.y = key; changed = true; }
                if (chart.color === oldKey) { chart.color = key; changed = true; }
                if (changed) {
                    chart.title = chartTitle(chart);
                    changedAny = true;
                }
            });
            if (changedAny) {
                currentCharts = uniqueCharts(currentCharts);
                currentCharts = currentCharts.filter(chart => !(chart.x && chart.y && chart.x === chart.y && chart.x_agg !== 'bin'));
                renderDashboardCharts();
                showToast(`Đã thay thế cột khóa "${oldKey}" thành "${key}" trên các biểu đồ.`, 'success');
            }
        }
    }
}

function buildTopicList(result) {
    if (Array.isArray(result.topics) && result.topics.length) {
        return result.topics.map((topic, index) => normalizeTopic(topic, index === 0));
    }

    const columns = result.profile?.columns || [];
    if (!columns.length) {
        return [{
            id: result.key_column,
            title: `Insights about ${displayName(result.key_column)}`,
            score: result.reward || 0,
            badges: ['correlation'],
            charts: ['fa-chart-bar', 'fa-chart-line'],
            active: true,
        }];
    }

    const chartFields = new Map();
    const insightFields = new Map();
    const badgeFields = new Map();
    const iconFields = new Map();
    const allCharts = [...currentCharts, ...recommendations];

    allCharts.forEach(chart => {
        const fields = [chart.x, chart.y, chart.color].filter(Boolean);
        fields.forEach(field => {
            chartFields.set(field, (chartFields.get(field) || 0) + 1);
            if (!iconFields.has(field)) iconFields.set(field, new Set());
            iconFields.get(field).add(iconForMark(chart.mark));
        });
    });

    (result.insights || []).forEach(insight => {
        (insight.columns || []).forEach(field => {
            insightFields.set(field, (insightFields.get(field) || 0) + Number(insight.reward || 1));
            if (!badgeFields.has(field)) badgeFields.set(field, new Set());
            badgeFields.get(field).add(insight.type);
        });
    });

    const mainKey = result.key_column || columns[0]?.name;
    return columns
        .map(column => {
            const field = column.name;
            const chartScore = chartFields.get(field) || 0;
            const insightScore = insightFields.get(field) || 0;
            const typeBonus = column.type === 'Q' ? 0.35 : column.type === 'T' ? 0.25 : 0.15;
            const activeBonus = field === mainKey ? 1.0 : 0.0;
            const rawScore = activeBonus + typeBonus + chartScore * 0.28 + insightScore * 0.18;
            const score = field === mainKey
                ? Number(result.reward || rawScore || 1)
                : Number(Math.max(0.75, Math.min((result.reward || 2) - 0.12, rawScore)).toFixed(2));
            const badges = [...(badgeFields.get(field) || [])].slice(0, 3);
            const icons = [...(iconFields.get(field) || [])].slice(0, 3);
            return {
                id: field,
                title: `Insights about ${displayName(field)}`,
                score,
                badges: badges.length ? badges : defaultBadgesForType(column.type),
                charts: icons.length ? icons : defaultIconsForType(column.type),
                active: field === mainKey,
            };
        })
        .sort((a, b) => Number(b.active) - Number(a.active) || b.score - a.score)
        .slice(0, 8);
}

function normalizeTopic(topic, active = false) {
    const charts = uniqueCharts((topic.charts || []).map(normalizeResponseChart));
    const recommendationsForTopic = uniqueCharts((topic.recommendations || []).map(normalizeResponseChart))
        .filter(chart => !charts.some(existing => isSameAnalysis(existing, chart)))
        .slice(0, 4);
    const badges = [...new Set((topic.insights || []).map(insight => insight.type))].slice(0, 4);
    const icons = [...new Set(charts.map(chart => iconForMark(chart.mark)))].slice(0, 3);
    return {
        id: topic.key_column,
        title: topic.title || (topic.key_column ? `Insights about ${displayName(topic.key_column)}` : 'Overview dashboard'),
        score: Number(topic.reward || 0),
        badges: badges.length ? badges : ['correlation'],
        charts: icons.length ? icons : ['fa-chart-bar', 'fa-chart-line'],
        dashboardCharts: charts,
        recommendations: recommendationsForTopic,
        active: Boolean(topic.active || active),
    };
}

function iconForMark(mark) {
    if (mark === 'line') return 'fa-chart-line';
    if (mark === 'point') return 'fa-braille';
    if (mark === 'boxplot') return 'fa-chart-simple';
    return 'fa-chart-bar';
}

function defaultBadgesForType(type) {
    if (type === 'T') return ['trend'];
    if (type === 'N') return ['comparison'];
    return ['distribution', 'correlation'];
}

function defaultIconsForType(type) {
    if (type === 'T') return ['fa-chart-line'];
    if (type === 'N') return ['fa-chart-bar'];
    return ['fa-chart-bar', 'fa-braille'];
}

function toggleTheme() {
    isDark = !isDark;
    document.documentElement.classList.toggle('dark', isDark);
    document.getElementById('themeIcon').className = isDark ? 'fas fa-sun text-sm text-amber-500' : 'fas fa-moon text-sm text-slate-500';
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
    if (mark === 'boxplot' && FIELD_META[y]?.type !== 'Q') {
        showToast('Boxplot can Y-axis la cot dinh luong (Q).', 'warn');
        return;
    }
    const normalizedXAgg = mark === 'boxplot' && FIELD_META[x]?.type === 'Q' && uniqueValueCount(x) > 12 ? 'bin' : xAgg;
    const normalizedYAgg = mark === 'boxplot' ? 'none' : yAgg;
    const chart = {
        mark,
        x,
        y,
        x_agg: normalizedXAgg,
        y_agg: normalizedYAgg,
        color: color === 'none' ? null : color,
        title: chartTitle({ mark, x, y, x_agg: normalizedXAgg, y_agg: normalizedYAgg }),
        insight_type: inferInsightLabel({ mark, x_agg: normalizedXAgg }),
    };
    if (currentCharts.some(existing => isSameAnalysis(existing, chart))) {
        showToast('Chart nay da trung insight tren Canvas.', 'warn');
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
    if (currentCharts.some(existing => isSameAnalysis(existing, rec))) {
        showToast('Chart nay da trung insight tren Canvas.', 'warn');
        return;
    }
    currentCharts.unshift({ ...rec, title: rec.title || chartTitle(rec) });
    currentCharts = uniqueCharts(currentCharts).slice(0, 8);
    recommendations = recommendations.filter((chart, i) => i !== index && !currentCharts.some(existing => isSameAnalysis(existing, chart)));
    renderDashboardCharts();
    renderRecommendations();
    showToast(`Da them "${rec.title || chartTitle(rec)}" vao Canvas.`, 'success');
}

function removeFromCanvas(index) {
    if (index >= 0 && index < currentCharts.length) {
        const title = currentCharts[index].title || chartTitle(currentCharts[index]);
        currentCharts.splice(index, 1);
        renderDashboardCharts();
        showToast(`Da xoa "${title}" khoi Canvas.`, 'info');
    }
}

async function exportDashboard(format) {
    const menu = document.getElementById('exportMenu');
    if (!currentCharts.length) {
        menu.classList.remove('show');
        showToast('Chua co dashboard de export.', 'warn');
        return;
    }
    menu.classList.remove('show');
    const normalizedFormat = String(format).toUpperCase();
    const fileName = `dashbot-${safeFileName(currentDatasetName || 'dashboard')}`;

    if (normalizedFormat === 'PDF') {
        const printWindow = window.open('', '_blank');
        if (!printWindow) {
            showToast('Trinh duyet dang chan popup PDF.', 'warn');
            return;
        }
        const pngDataUrl = await createDashboardExportCanvas().then(canvas => canvas.toDataURL('image/png'));
        printWindow.document.write(`
            <!doctype html>
            <html>
            <head>
                <title>${fileName}.pdf</title>
                <style>
                    body { margin: 0; padding: 24px; background: #f8fafc; font-family: Inter, Arial, sans-serif; }
                    img { width: 100%; max-width: 1400px; display: block; margin: 0 auto; }
                    @media print { body { padding: 0; background: white; } img { max-width: 100%; } }
                </style>
            </head>
            <body>
                <img src="${pngDataUrl}" alt="DashBot dashboard export">
                <script>window.onload = () => setTimeout(() => window.print(), 250);<\/script>
            </body>
            </html>
        `);
        printWindow.document.close();
        showToast('PDF da mo o cua so in. Chon Save as PDF de luu.', 'info');
        return;
    }

    if (normalizedFormat === 'SVG') {
        downloadBlob(
            new Blob([createDashboardSvg()], { type: 'image/svg+xml;charset=utf-8' }),
            `${fileName}.svg`
        );
        showToast('Da tai dashboard SVG.', 'success');
        return;
    }

    const canvas = await createDashboardExportCanvas();
    canvas.toBlob(blob => {
        if (!blob) {
            showToast('Khong tao duoc PNG.', 'warn');
            return;
        }
        downloadBlob(blob, `${fileName}.png`);
        showToast('Da tai dashboard PNG.', 'success');
    }, 'image/png', 0.95);
}

function downloadBlob(blob, filename) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
}

function safeFileName(value) {
    return String(value).replace(/\.[^.]+$/, '').replace(/[^a-z0-9_-]+/gi, '-').replace(/^-+|-+$/g, '').toLowerCase() || 'dashboard';
}

function escapeXml(value) {
    return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&apos;');
}

function createDashboardSvg() {
    const width = 1500;
    const columns = 3;
    const gap = 28;
    const margin = 42;
    const cardWidth = (width - margin * 2 - gap * (columns - 1)) / columns;
    const cardHeight = 285;
    const headerHeight = 116;
    const rows = Math.ceil(currentCharts.length / columns);
    const height = headerHeight + margin + rows * cardHeight + Math.max(0, rows - 1) * gap;
    const bg = isDark ? '#0b1120' : '#f8fafc';
    const cardBg = isDark ? '#111827' : '#ffffff';
    const text = isDark ? '#e2e8f0' : '#1e293b';
    const muted = isDark ? '#94a3b8' : '#64748b';
    const border = isDark ? '#263244' : '#e2e8f0';
    const title = document.getElementById('canvasTitle')?.textContent || `Insights about ${currentDatasetName || 'dashboard'}`;
    const subtitle = document.getElementById('canvasSubtitle')?.textContent || 'Generated by DashBot.';
    const chartBlocks = currentCharts.map((chart, i) => {
        const col = i % columns;
        const row = Math.floor(i / columns);
        const x = margin + col * (cardWidth + gap);
        const y = headerHeight + row * (cardHeight + gap);
        const canvas = document.getElementById(`chart${i}`);
        const dataUrl = canvas ? canvas.toDataURL('image/png') : '';
        return `
            <g>
                <rect x="${x}" y="${y}" width="${cardWidth}" height="${cardHeight}" rx="14" fill="${cardBg}" stroke="${border}"/>
                <text x="${x + 22}" y="${y + 34}" font-size="17" font-weight="700" fill="${text}">${escapeXml(chart.title || chartTitle(chart))}</text>
                <text x="${x + cardWidth - 110}" y="${y + 34}" font-size="12" font-weight="700" fill="#ef4444">${escapeXml(chart.insight_type || inferInsightLabel(chart))}</text>
                <image x="${x + 22}" y="${y + 58}" width="${cardWidth - 44}" height="${cardHeight - 82}" href="${dataUrl}" preserveAspectRatio="xMidYMid meet"/>
            </g>
        `;
    }).join('');
    return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
    <rect width="100%" height="100%" fill="${bg}"/>
    <text x="${margin}" y="54" font-size="34" font-weight="800" fill="${text}">${escapeXml(title)}</text>
    <text x="${margin}" y="88" font-size="17" fill="${muted}">${escapeXml(subtitle)}</text>
    ${chartBlocks}
</svg>`;
}

async function createDashboardExportCanvas() {
    await new Promise(resolve => requestAnimationFrame(resolve));
    const width = 1500;
    const columns = 3;
    const gap = 28;
    const margin = 42;
    const cardWidth = (width - margin * 2 - gap * (columns - 1)) / columns;
    const cardHeight = 285;
    const headerHeight = 116;
    const rows = Math.ceil(currentCharts.length / columns);
    const height = headerHeight + margin + rows * cardHeight + Math.max(0, rows - 1) * gap;
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const bg = isDark ? '#0b1120' : '#f8fafc';
    const cardBg = isDark ? '#111827' : '#ffffff';
    const text = isDark ? '#e2e8f0' : '#1e293b';
    const muted = isDark ? '#94a3b8' : '#64748b';
    const border = isDark ? '#263244' : '#e2e8f0';

    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = text;
    ctx.font = '800 34px Inter, Arial, sans-serif';
    ctx.fillText(document.getElementById('canvasTitle')?.textContent || `Insights about ${currentDatasetName || 'dashboard'}`, margin, 54);
    ctx.fillStyle = muted;
    ctx.font = '17px Inter, Arial, sans-serif';
    ctx.fillText(document.getElementById('canvasSubtitle')?.textContent || 'Generated by DashBot.', margin, 88);

    currentCharts.forEach((chart, i) => {
        const col = i % columns;
        const row = Math.floor(i / columns);
        const x = margin + col * (cardWidth + gap);
        const y = headerHeight + row * (cardHeight + gap);
        drawRoundRect(ctx, x, y, cardWidth, cardHeight, 14, cardBg, border);
        ctx.fillStyle = text;
        ctx.font = '700 17px Inter, Arial, sans-serif';
        drawClippedText(ctx, chart.title || chartTitle(chart), x + 22, y + 34, cardWidth - 150);
        ctx.fillStyle = '#ef4444';
        ctx.font = '700 12px Inter, Arial, sans-serif';
        ctx.fillText(chart.insight_type || inferInsightLabel(chart), x + cardWidth - 110, y + 34);
        const source = document.getElementById(`chart${i}`);
        if (source) ctx.drawImage(source, x + 22, y + 58, cardWidth - 44, cardHeight - 82);
    });
    return canvas;
}

function drawRoundRect(ctx, x, y, width, height, radius, fill, stroke) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + width, y, x + width, y + height, radius);
    ctx.arcTo(x + width, y + height, x, y + height, radius);
    ctx.arcTo(x, y + height, x, y, radius);
    ctx.arcTo(x, y, x + width, y, radius);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1;
    ctx.stroke();
}

function drawClippedText(ctx, value, x, y, maxWidth) {
    let text = String(value);
    while (ctx.measureText(text).width > maxWidth && text.length > 4) {
        text = `${text.slice(0, -4)}...`;
    }
    ctx.fillText(text, x, y);
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
        updateUploadProgress(35, 'Profiling');
        const job = await startDashboardRecommendationJob(file, 5);
        const result = await waitForRecommendationJob(job.job_id);
        b.style.width = '100%'; pc.textContent = '100%';
        applyRecommendationResult(result, file);
        setTimeout(() => { closeUploadModal(); p.classList.add('hidden'); b.style.width = '0%'; }, 300);
    } catch (err) {
        p.classList.add('hidden'); b.style.width = '0%';
        showToast(`Khong xu ly duoc upload: ${err.message}`, 'warn');
    }
}

async function waitForRecommendationJob(jobId) {
    while (true) {
        const job = await getDashboardRecommendationJob(jobId);
        updateUploadProgress(job.progress || 50, job.message || 'Generating');
        if (job.status === 'completed') return job.result;
        if (job.status === 'failed') throw new Error(job.error || job.message || 'Recommendation failed');
        await new Promise(resolve => setTimeout(resolve, 900));
    }
}

function updateUploadProgress(percent, label = '') {
    const b = document.getElementById('uploadBar'), pc = document.getElementById('uploadPercent');
    const safePercent = Math.max(0, Math.min(100, Math.round(percent)));
    b.style.width = `${safePercent}%`;
    pc.textContent = label ? `${safePercent}% · ${label}` : `${safePercent}%`;
}

function applyRecommendationResult(result, file) {
    updateMetadata(uploadedRows, result.profile, file);
    currentCharts = uniqueCharts(result.charts.map(normalizeResponseChart));
    const canvasSignatures = new Set(currentCharts.map(chartSignature));
    const recommendationSource = Array.isArray(result.recommendations) ? result.recommendations : [];
    recommendations = uniqueCharts(recommendationSource.map(normalizeResponseChart))
        .filter(chart => !canvasSignatures.has(chartSignature(chart)) && !currentCharts.some(existing => isSameAnalysis(existing, chart)))
        .slice(0, 4);
    topics = buildTopicList(result);
    const activeTopic = topics.find(topic => topic.active) || topics[0];
    if (activeTopic?.dashboardCharts?.length) currentCharts = activeTopic.dashboardCharts;
    if (activeTopic?.recommendations) recommendations = activeTopic.recommendations;
    renderCanvasTitle(activeTopic?.id ?? result.key_column, activeTopic?.title);
    renderAttributes();
    renderTopics();
    renderPreviewTable();
    updateKPIs(activeTopic?.id ?? result.key_column);
    renderDashboardCharts();
    renderRecommendations();
    showToast(`Da render ${currentCharts.length} chart tu ${currentDatasetName}. Return: ${result.reward.toFixed(2)}`, 'success');
}

function normalizeResponseChart(chart) {
    const normalized = {
        mark: chart.mark,
        x: chart.x,
        y: chart.y,
        color: chart.color,
        x_agg: chart.x_agg,
        y_agg: chart.y_agg,
        insight_type: chart.insight_type || inferInsightLabel(chart),
    };
    return { ...normalized, title: chartTitle(normalized) };
}

function chartSignature(chart) {
    return [
        chart.mark || '',
        chart.x || '',
        chart.y || '',
        chart.color || '',
        chart.x_agg || '',
        chart.y_agg || '',
    ].join('|').toLowerCase();
}

function analysisSignature(chart) {
    if (chart.mark === 'point') {
        return ['relationship', ...[chart.x || '', chart.y || ''].sort()].join('|').toLowerCase();
    }
    if (chart.mark === 'line') {
        return ['trend', chart.x || '', chart.y || '', chart.color || ''].join('|').toLowerCase();
    }
    if (chart.mark === 'bar' && chart.x_agg === 'bin') {
        return ['distribution', chart.x || ''].join('|').toLowerCase();
    }
    return ['grouped', chart.mark || '', chart.x || '', chart.y || '', chart.color || ''].join('|').toLowerCase();
}

function isSameAnalysis(a, b) {
    return chartSignature(a) === chartSignature(b) || analysisSignature(a) === analysisSignature(b);
}

function uniqueCharts(charts) {
    const seenExact = new Set();
    const seenAnalysis = new Set();
    const unique = [];
    charts.forEach(chart => {
        const exact = chartSignature(chart);
        const analysis = analysisSignature(chart);
        if (seenExact.has(exact) || seenAnalysis.has(analysis)) return;
        seenExact.add(exact);
        seenAnalysis.add(analysis);
        unique.push(chart);
    });
    return unique;
}

document.addEventListener('DOMContentLoaded', () => {
    setEmptyState();
    document.getElementById('uploadModal').classList.remove('open');
    document.getElementById('previewModal').classList.remove('open');

    // Sidebar divider drag-to-resize setup
    const resizer = document.getElementById('sidebar-resizer');
    const topicSection = document.getElementById('topicListSection');
    if (resizer && topicSection) {
        let isDragging = false;
        resizer.addEventListener('mousedown', (e) => {
            isDragging = true;
            document.body.style.cursor = 'row-resize';
            document.body.style.userSelect = 'none';
        });
        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const sidebar = topicSection.parentElement;
            const sidebarRect = sidebar.getBoundingClientRect();
            const newHeight = e.clientY - sidebarRect.top;
            // Clamp height between 150px and sidebar_height - 180px
            const clampedHeight = Math.max(150, Math.min(newHeight, sidebarRect.height - 180));
            topicSection.style.height = `${clampedHeight}px`;
        });
        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
            }
        });
    }
});
