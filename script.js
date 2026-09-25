const CHART_COLORS = {
  spark: '#f5a623',
  pos: '#3ddc97',
  neg: '#ff6b6b',
  accent2: '#5b8def',
  grid: '#262c38',
  text: '#9aa3b4',
};

Chart.defaults.color = CHART_COLORS.text;
Chart.defaults.borderColor = CHART_COLORS.grid;
Chart.defaults.font.family = "'Inter', sans-serif";

async function loadResults() {
  try {
    const res = await fetch('/api/results');
    if (!res.ok) throw new Error('no results yet');
    const data = await res.json();
    render(data);
  } catch (e) {
    document.querySelector('main').innerHTML =
      '<div class="empty-note" style="padding:80px 0;text-align:center;">' +
      'No results yet — run <code>python spark_pipeline.py</code> first, then reload this page.' +
      '</div>';
  }
}

function render(data) {
  renderHeroStats(data);
  renderEDA(data.eda);
  renderModels(data.models, data.best_model);
  renderScaling(data.scaling_demo || []);
  renderInsights(data.business_insights);
  renderErrors(data.error_analysis?.misclassified_examples || []);
}

function renderHeroStats(data) {
  document.getElementById('stat-rows').textContent = data.eda.total_rows.toLocaleString();
  const best = data.models.find(m => m.name === data.best_model);
  document.getElementById('stat-best-f1').textContent = best ? best.f1.toFixed(3) : '—';
  document.getElementById('stat-partitions').textContent = data.dataset_info.num_partitions ?? '—';
}

function renderEDA(eda) {
  new Chart(document.getElementById('chart-class'), {
    type: 'doughnut',
    data: {
      labels: ['Positive', 'Negative'],
      datasets: [{
        data: [eda.class_distribution.positive || 0, eda.class_distribution.negative || 0],
        backgroundColor: [CHART_COLORS.pos, CHART_COLORS.neg],
        borderWidth: 0,
      }],
    },
    options: { plugins: { legend: { position: 'bottom' } } },
  });

  new Chart(document.getElementById('chart-length'), {
    type: 'bar',
    data: {
      labels: eda.review_length.histogram.map(h => h.bucket + '+'),
      datasets: [{
        label: 'Reviews',
        data: eda.review_length.histogram.map(h => h.count),
        backgroundColor: CHART_COLORS.accent2,
        borderRadius: 4,
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: 'Words per review' } } },
    },
  });

  new Chart(document.getElementById('chart-words'), {
    type: 'bar',
    data: {
      labels: eda.top_words.map(w => w.word),
      datasets: [{
        label: 'Frequency',
        data: eda.top_words.map(w => w.count),
        backgroundColor: CHART_COLORS.spark,
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: 'y',
      plugins: { legend: { display: false } },
    },
  });
}

function renderModels(models, bestName) {
  const tbody = document.querySelector('#model-table tbody');
  tbody.innerHTML = models.map(m => `
    <tr>
      <td>${m.name}${m.name === bestName ? '<span class="best-tag">best</span>' : ''}</td>
      <td>${m.accuracy}</td>
      <td>${m.precision}</td>
      <td>${m.recall}</td>
      <td>${m.f1}</td>
      <td>${m.roc_auc ?? '—'}</td>
      <td>${m.train_time_sec}</td>
    </tr>
  `).join('');

  new Chart(document.getElementById('chart-metrics'), {
    type: 'bar',
    data: {
      labels: ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
      datasets: models.map((m, i) => ({
        label: m.name,
        data: [m.accuracy, m.precision, m.recall, m.f1, m.roc_auc ?? 0],
        backgroundColor: [CHART_COLORS.spark, CHART_COLORS.accent2, CHART_COLORS.pos][i % 3],
        borderRadius: 4,
      })),
    },
    options: {
      scales: { y: { min: 0, max: 1 } },
      plugins: { legend: { position: 'bottom' } },
    },
  });

  const rocDatasets = models.filter(m => m.roc_curve && m.roc_curve.length).map((m, i) => ({
    label: m.name,
    data: m.roc_curve.map(p => ({ x: p.fpr, y: p.tpr })),
    borderColor: [CHART_COLORS.spark, CHART_COLORS.accent2, CHART_COLORS.pos][i % 3],
    backgroundColor: 'transparent',
    showLine: true,
    pointRadius: 0,
    borderWidth: 2,
  }));
  new Chart(document.getElementById('chart-roc'), {
    type: 'scatter',
    data: {
      datasets: [
        ...rocDatasets,
        {
          label: 'Random guess',
          data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
          borderColor: '#3a3f4c',
          borderDash: [4, 4],
          showLine: true,
          pointRadius: 0,
        },
      ],
    },
    options: {
      scales: {
        x: { title: { display: true, text: 'False Positive Rate' }, min: 0, max: 1 },
        y: { title: { display: true, text: 'True Positive Rate' }, min: 0, max: 1 },
      },
      plugins: { legend: { position: 'bottom' } },
    },
  });

  const cmGrid = document.getElementById('confusion-grid');
  cmGrid.innerHTML = models.map(m => `
    <div class="cm-card">
      <h4>${m.name} — Confusion Matrix</h4>
      <div class="cm-grid">
        <div class="cm-cell cm-tp"><div class="n">${m.confusion_matrix.tp}</div><div class="l">True Positive</div></div>
        <div class="cm-cell cm-fn"><div class="n">${m.confusion_matrix.fn}</div><div class="l">False Negative</div></div>
        <div class="cm-cell cm-fp"><div class="n">${m.confusion_matrix.fp}</div><div class="l">False Positive</div></div>
        <div class="cm-cell cm-tn"><div class="n">${m.confusion_matrix.tn}</div><div class="l">True Negative</div></div>
      </div>
    </div>
  `).join('');
}

function renderScaling(scalingDemo) {
  if (!scalingDemo.length) return;
  new Chart(document.getElementById('chart-scaling'), {
    type: 'line',
    data: {
      labels: scalingDemo.map(point => point.rows.toLocaleString()),
      datasets: [{
        label: 'Logistic Regression training time (seconds)',
        data: scalingDemo.map(point => point.train_time_sec),
        borderColor: CHART_COLORS.spark,
        backgroundColor: CHART_COLORS.spark,
        tension: 0.2,
        pointRadius: 5,
      }],
    },
    options: {
      scales: {
        x: { title: { display: true, text: 'Rows used for training' } },
        y: { title: { display: true, text: 'Wall-clock training time (seconds)' }, beginAtZero: true },
      },
      plugins: { legend: { position: 'bottom' } },
    },
  });
}

function renderInsights(insights) {
  const grid = document.getElementById('insights-grid');
  const rows = [
    [insights.overall_sentiment.total_reviews_analyzed.toLocaleString(), 'Reviews analyzed'],
    [insights.overall_sentiment.percent_negative + '%', 'Negative review rate'],
    [insights.model_summary.best_model_name, 'Best performing model'],
    [insights.review_characteristics.avg_words_negative_review ?? '—', 'Avg. words / negative review'],
    [insights.review_characteristics.avg_words_positive_review ?? '—', 'Avg. words / positive review'],
    [insights.model_summary.best_model_f1, 'Best model F1 score'],
  ];
  grid.innerHTML = rows.map(([n, l]) => `
    <div class="insight-card"><div class="n">${n}</div><div class="l">${l}</div></div>
  `).join('');
}

function renderErrors(errors) {
  const list = document.getElementById('errors-list');
  if (!errors || !errors.length) {
    list.innerHTML = '<div class="empty-note">No misclassified examples captured.</div>';
    return;
  }
  list.innerHTML = errors.map(e => `
    <div class="error-item">
      "${e.review}"
      <div class="tags">True: <b>${e.true_sentiment}</b> · Predicted: <b>${e.predicted_sentiment}</b> · ${e.length_words} words</div>
    </div>
  `).join('');
}

// ---------- Live analyzer ----------
const analyzeBtn = document.getElementById('analyze-btn');
const reviewInput = document.getElementById('review-input');
const resultBox = document.getElementById('result-box');

document.querySelectorAll('.example-chip').forEach(chip => {
  chip.addEventListener('click', () => { reviewInput.value = chip.dataset.text; });
});

analyzeBtn.addEventListener('click', async () => {
  const text = reviewInput.value.trim();
  if (!text) return;
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing...';
  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ review: text }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    resultBox.classList.remove('hidden');
    const sentimentEl = document.getElementById('result-sentiment');
    sentimentEl.textContent = data.sentiment;
    sentimentEl.className = 'result-sentiment ' + (data.sentiment === 'POSITIVE' ? 'pos' : 'neg');

    document.getElementById('bar-neg').style.width = data.prob_negative + '%';
    document.getElementById('bar-pos').style.width = data.prob_positive + '%';
    document.getElementById('prob-neg-label').textContent = data.prob_negative + '%';
    document.getElementById('prob-pos-label').textContent = data.prob_positive + '%';
    document.getElementById('result-meta').textContent =
      `Model: ${data.model_used} · Confidence: ${data.confidence}%`;
  } catch (e) {
    alert('Prediction failed: ' + e.message + '\n\nMake sure api.py is running and spark_pipeline.py has completed.');
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze Review';
  }
});

loadResults();
