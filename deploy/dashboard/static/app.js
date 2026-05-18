Chart.defaults.color = '#6b6a5e';
Chart.defaults.borderColor = '#232b1a';
Chart.defaults.font.family = "'DM Mono', monospace";
Chart.defaults.font.size = 11;

const HONEY = '#e8a020';
const HONEY2 = '#f5c842';
const GREEN2 = '#6aaf5c';

let activeRange = '24h';
let currentThreshold = 50;
let mainChart = null;

const RANGE_LABELS = {
    '1h': 'Anzahl & Durchschnitt — pro Minute (letzte Stunde)',
    '24h': 'Anzahl & Durchschnitt — pro Stunde (letzte 24 Stunden)',
    '30d': 'Anzahl & Durchschnitt — pro Tag (letzte 30 Tage)',
    '365d': 'Anzahl & Durchschnitt — pro Tag (letztes Jahr)',
};

// ── Helpers ───────────────────────────────────────────────────────
function formatTime(iso) {
    if (!iso) return '—';
    return new Date(iso).toLocaleTimeString('de-CH', { hour: '2-digit', minute: '2-digit' });
}

function formatDate(str) {
    // Input: 2026-05-18 → 18.05.2026
    if (!str) return '—';
    const [y, m, d] = str.split('-');
    return `${d}.${m}.${y}`;
}

function formatDateTime(iso) {
    if (!iso) return '—';
    return new Date(iso).toLocaleString('de-CH', {
        day: '2-digit', month: '2-digit', year: 'numeric',
        hour: '2-digit', minute: '2-digit'
    });
}

function formatSnapshotTimestamp(raw) {
    // Input: 2026-05-18T13-59-18 → 18.05.2026 13:59
    if (!raw) return '';
    const iso = raw.replace(/T(\d{2})-(\d{2})-(\d{2})$/, 'T$1:$2:$3Z');
    const d = new Date(iso);
    const date = d.toLocaleDateString('de-CH', { day: '2-digit', month: '2-digit', year: 'numeric' });
    const time = d.toLocaleTimeString('de-CH', { hour: '2-digit', minute: '2-digit' });
    return `${date} ${time}`;
}

function formatChartLabel(label, isBar) {
    if (isBar) return formatDate(label);
    // label is full ISO UTC string e.g. 2026-05-18T13:00Z → convert to local time
    const d = new Date(label);
    return d.toLocaleTimeString('de-CH', { hour: '2-digit', minute: '2-digit' });
}

function movingAvg(data, w = 5) {
    return data.map((_, i) => {
        const slice = data.slice(Math.max(0, i - w + 1), i + 1);
        return Math.round(slice.reduce((a, b) => a + b, 0) / slice.length * 10) / 10;
    });
}

// ── Chart ─────────────────────────────────────────────────────────
function buildChart(labels, countData, avgData, isBar) {
    const ctx = document.getElementById('mainChart').getContext('2d');
    if (mainChart) { mainChart.destroy(); mainChart = null; }

    const datasets = isBar ? [
        {
            label: 'Anzahl (Peak)', data: countData, type: 'bar',
            backgroundColor: 'rgba(232,160,32,0.45)', borderColor: HONEY,
            borderWidth: 1, borderRadius: 2, order: 2,
        },
        {
            label: 'Durchschnitt', data: avgData, type: 'line',
            borderColor: GREEN2, backgroundColor: 'transparent',
            pointRadius: 2, pointHoverRadius: 5, borderWidth: 2, tension: 0.3,
            fill: false, order: 1,
        },
    ] : [
        {
            label: 'Anzahl', data: countData, type: 'line',
            borderColor: HONEY,
            backgroundColor: (ctx) => {
                const g = ctx.chart.ctx.createLinearGradient(0, 0, 0, 260);
                g.addColorStop(0, 'rgba(232,160,32,0.25)');
                g.addColorStop(1, 'rgba(232,160,32,0)');
                return g;
            },
            fill: true, tension: 0.4,
            pointRadius: countData.length > 60 ? 0 : 3, pointHoverRadius: 5,
            pointBackgroundColor: HONEY2, pointBorderColor: '#0d0f0a', pointBorderWidth: 2,
            borderWidth: 2, order: 2,
        },
        ...(avgData.length ? [{
            label: 'Durchschnitt', data: avgData, type: 'line',
            borderColor: GREEN2, backgroundColor: 'transparent',
            pointRadius: 0, pointHoverRadius: 4, borderWidth: 2, tension: 0.5,
            fill: false, order: 1,
        }] : []),
    ];

    mainChart = new Chart(ctx, {
        type: isBar ? 'bar' : 'line',
        data: { labels, datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#141810', borderColor: '#232b1a', borderWidth: 1,
                    titleColor: HONEY, bodyColor: '#d4cfc0',
                },
            },
            scales: {
                x: { grid: { color: '#1a1f13' }, ticks: { maxTicksLimit: 12, maxRotation: isBar ? 45 : 0 } },
                y: { grid: { color: '#1a1f13' }, beginAtZero: true },
            },
        },
    });
}

async function loadMainChart() {
    const isBar = activeRange === '30d' || activeRange === '365d';
    document.getElementById('chartTitle').textContent = RANGE_LABELS[activeRange];
    try {
        const res = await fetch(`/api/aggregated?period=${activeRange}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const rows = await res.json();
        const labels = rows.map(r => formatChartLabel(r.label, isBar));
        const countData = rows.map(r => r.count);
        const avgData = rows.map(r => r.avg);
        const smoothAvg = isBar ? avgData : (activeRange === '1h' ? [] : movingAvg(avgData, 5));
        buildChart(labels, countData, smoothAvg, isBar);
    } catch (e) { console.error('Chart load failed', e); }
}

document.querySelectorAll('#rangeGroup .sw-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('#rangeGroup .sw-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeRange = btn.dataset.range;
        loadMainChart();
    });
});

// ── Data fetching ─────────────────────────────────────────────────
async function fetchStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        document.getElementById('statNow').textContent = data.latest?.count ?? '—';
        document.getElementById('statPeak').textContent = data.peak ?? '—';
        document.getElementById('statAvg').textContent = data.avg ?? '—';
        const ts = data.latest?.timestamp;
        if (ts) {
            const age = (Date.now() - new Date(ts).getTime()) / 1000;
            document.getElementById('statusDot').className = age < 180 ? 'status-dot' : 'status-dot offline';
        }
        document.getElementById('lastUpdated').textContent =
            'aktualisiert ' + new Date().toLocaleTimeString('de-CH');
    } catch (e) {
        document.getElementById('statusDot').className = 'status-dot offline';
    }
}

async function fetchSnapshot() {
    try {
        const res = await fetch('/api/snapshot');
        const data = await res.json();
        if (!data.url) return;
        const wrap = document.getElementById('snapshotWrap');
        const img = wrap.querySelector('img');
        const src = `${data.url}?t=${Date.now()}`;
        if (img) {
            img.src = src;
        } else {
            wrap.innerHTML = `<img src="${src}" alt="Snapshot"><div class="snapshot-timestamp">${formatSnapshotTimestamp(data.timestamp)}</div>`;
        }
    } catch (e) { }
}

async function fetchAlarms() {
    try {
        const res = await fetch('/api/alarms');
        const rows = await res.json();
        const list = document.getElementById('alarmList');
        list.innerHTML = rows.length
            ? rows.map(r => `
          <div class="alarm-entry">
            <span class="time">${formatDateTime(r.timestamp)}</span>
            <span class="count">${r.count} Bienen</span>
            <span class="thresh">Limit: ${r.threshold}</span>
          </div>`).join('')
            : '<div class="no-alarms">Keine Alarme</div>';
    } catch (e) { }
}

async function checkAlarm() {
    try {
        const res = await fetch('/api/alarms/check', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ threshold: currentThreshold }),
        });
        const data = await res.json();
        if (data.alarm) {
            document.getElementById('alarmBannerText').textContent =
                `⚠ Schwarmalarm! ${data.count} Bienen erkannt (Limit: ${currentThreshold}) um ${formatTime(data.timestamp)}`;
            const banner = document.getElementById('alarmBanner');
            banner.classList.add('visible');
            setTimeout(() => banner.classList.remove('visible'), 15000);
            await fetchAlarms();
        }
    } catch (e) { }
}

async function loadConfig() {
    try {
        const res = await fetch('/api/config');
        const data = await res.json();
        currentThreshold = data.threshold;
        document.getElementById('thresholdDisplay').textContent = currentThreshold;
        return data.refresh_seconds || 60;
    } catch (e) { return 60; }
}

async function refresh() {
    await Promise.all([fetchStats(), fetchSnapshot(), fetchAlarms(), loadMainChart()]);
    await checkAlarm();
}

loadConfig().then(refreshSeconds => {
    refresh();
    setInterval(refresh, refreshSeconds * 1000);
});