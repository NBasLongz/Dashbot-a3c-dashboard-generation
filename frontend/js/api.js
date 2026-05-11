const DASHBOT_API_BASE = window.DASHBOT_API_BASE || 'http://127.0.0.1:8010';
const DASHBOT_RECOMMEND_MODE = window.DASHBOT_RECOMMEND_MODE || 'a3c';
const DASHBOT_SEARCH_STEPS = window.DASHBOT_SEARCH_STEPS || 1000;

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
