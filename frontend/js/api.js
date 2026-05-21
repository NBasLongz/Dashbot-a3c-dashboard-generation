const DASHBOT_QUERY = new URLSearchParams(window.location.search);
const DASHBOT_API_BASE = DASHBOT_QUERY.get('apiBase') || window.DASHBOT_API_BASE || 'http://127.0.0.1:8010';
const DASHBOT_RECOMMEND_MODE = DASHBOT_QUERY.get('mode') || window.DASHBOT_RECOMMEND_MODE || 'a3c';
const DASHBOT_SEARCH_STEPS = Number(DASHBOT_QUERY.get('searchSteps') || window.DASHBOT_SEARCH_STEPS || 1000);

async function requestDashboardRecommendation(file, maxCharts = 5) {
    const formData = new FormData();
    formData.append('file', file);
    const params = new URLSearchParams({
        max_charts: String(maxCharts),
        mode: DASHBOT_RECOMMEND_MODE,
        search_steps: String(DASHBOT_SEARCH_STEPS),
    });
    const response = await fetch(`${DASHBOT_API_BASE}/api/recommend?${params.toString()}`, {
        method: 'POST',
        body: formData,
    });
    if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `DashBot API error: ${response.status}`);
    }
    return response.json();
}

async function startDashboardRecommendationJob(file, maxCharts = 5) {
    const formData = new FormData();
    formData.append('file', file);
    const params = new URLSearchParams({
        max_charts: String(maxCharts),
        mode: DASHBOT_RECOMMEND_MODE,
        search_steps: String(DASHBOT_SEARCH_STEPS),
    });
    const response = await fetch(`${DASHBOT_API_BASE}/api/recommend/jobs?${params.toString()}`, {
        method: 'POST',
        body: formData,
    });
    if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `DashBot API error: ${response.status}`);
    }
    return response.json();
}

async function getDashboardRecommendationJob(jobId) {
    const response = await fetch(`${DASHBOT_API_BASE}/api/recommend/jobs/${jobId}`);
    if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `DashBot API error: ${response.status}`);
    }
    return response.json();
}
