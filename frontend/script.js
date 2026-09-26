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
  if (data.absa) renderABSA(data.absa);
  if (data.complaint_mining) renderComplaints(data.complaint_mining);
  if (data.lda_topics) renderTopics(data.lda_topics);
  renderScaling(data.scaling_demo || []);
  renderInsights(data.business_insights);
  renderErrors(data.error_analysis?.misclassified_examples || []);
}

function renderHeroStats(data) {
  const totalReviews = data.eda.total_rows;
  const positiveReviews = data.eda.class_distribution.positive || 0;
  const negativeReviews = data.eda.class_distribution.negative || 0;
  const best = data.models.find(m => m.name === data.best_model);
  document.getElementById('stat-reviews').textContent = totalReviews.toLocaleString();
  document.getElementById('stat-positive').textContent = totalReviews
    ? `${(positiveReviews / totalReviews * 100).toFixed(1)}%`
    : '—';
  document.getElementById('stat-negative').textContent = totalReviews
    ? `${(negativeReviews / totalReviews * 100).toFixed(1)}%`
    : '—';
  document.getElementById('stat-best-model').textContent = data.best_model;
  document.getElementById('stat-best-accuracy').textContent = best
    ? `${(best.accuracy * 100).toFixed(1)}%`
    : '—';
  document.getElementById('stat-aspect-mentions').textContent = data.absa
    ? data.absa.total_aspect_mentions.toLocaleString()
    : '—';
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

function renderABSA(absa) {
  if (!absa || !absa.aspects) return;

  // 1. Aspect Progress Overview
  const progressList = document.getElementById('aspect-progress-list');
  if (progressList) {
    progressList.innerHTML = absa.aspects.map(a => `
      <div class="aspect-progress-item">
        <div class="aspect-progress-header">
          <span class="aspect-name">${a.aspect} ${a.is_pain_point ? '<span class="badge" style="color:var(--neg); font-size:11px; margin-left:6px;">⚠️ Pain Point</span>' : ''}</span>
          <span class="aspect-meta">${a.mentions.toLocaleString()} mentions · <b style="color:var(--neg);">${a.negative_pct}% Neg</b> / <b style="color:var(--pos);">${a.positive_pct}% Pos</b></span>
        </div>
        <div class="aspect-bar-track">
          <div class="aspect-bar-fill-neg" style="width:${a.negative_pct}%;"></div>
          <div class="aspect-bar-fill-pos" style="width:${a.positive_pct}%;"></div>
        </div>
      </div>
    `).join('');
  }

  // 2. Aspect Sentiment Distribution (Stacked Bar)
  const chartSentimentEl = document.getElementById('chart-aspect-sentiment');
  if (chartSentimentEl) {
    new Chart(chartSentimentEl, {
      type: 'bar',
      data: {
        labels: absa.aspects.map(a => a.aspect),
        datasets: [
          {
            label: 'Positive %',
            data: absa.aspects.map(a => a.positive_pct),
            backgroundColor: CHART_COLORS.pos,
            borderRadius: 4,
          },
          {
            label: 'Negative %',
            data: absa.aspects.map(a => a.negative_pct),
            backgroundColor: CHART_COLORS.neg,
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        scales: {
          x: { stacked: true },
          y: { stacked: true, max: 100, title: { display: true, text: 'Percentage (%)' } },
        },
        plugins: { legend: { position: 'bottom' } },
      },
    });
  }

  // 3. Aspect Negative Rate (%)
  const chartNegativeEl = document.getElementById('chart-aspect-negative');
  if (chartNegativeEl) {
    new Chart(chartNegativeEl, {
      type: 'bar',
      data: {
        labels: absa.aspects.map(a => a.aspect),
        datasets: [{
          label: 'Negative Feedback Rate (%)',
          data: absa.aspects.map(a => a.negative_pct),
          backgroundColor: absa.aspects.map(a => a.is_pain_point ? CHART_COLORS.neg : CHART_COLORS.spark),
          borderRadius: 4,
        }],
      },
      options: {
        indexAxis: 'y',
        scales: {
          x: { min: 0, max: 100, title: { display: true, text: 'Negative Sentiment %' } },
        },
        plugins: { legend: { display: false } },
      },
    });
  }

  // 4. Mention Volume per Aspect
  const chartMentionsEl = document.getElementById('chart-aspect-mentions');
  if (chartMentionsEl) {
    new Chart(chartMentionsEl, {
      type: 'bar',
      data: {
        labels: absa.aspects.map(a => a.aspect),
        datasets: [{
          label: 'Mentions Count',
          data: absa.aspects.map(a => a.mentions),
          backgroundColor: CHART_COLORS.accent2,
          borderRadius: 4,
        }],
      },
      options: {
        scales: {
          y: { beginAtZero: true, title: { display: true, text: 'Reviews Discussing Aspect' } },
        },
        plugins: { legend: { display: false } },
      },
    });
  }

  // 5. Pain Points Grid
  const painGrid = document.getElementById('pain-points-grid');
  if (painGrid) {
    painGrid.innerHTML = (absa.pain_points || []).map(p => `
      <div class="pain-card">
        <div class="title">${p.aspect}</div>
        <div class="rate">${p.negative_pct}% Negative</div>
        <div class="desc">${p.mentions.toLocaleString()} customer mentions analyzed.</div>
        <div class="badge">High Negative Feedback</div>
      </div>
    `).join('');
  }
}

// ---------- Complaint Mining & LDA Topics ----------
function renderComplaints(cm) {
  // 1. Populate Top Complaints table
  const tbody = document.querySelector('#complaints-table tbody');
  if (tbody) {
    const rows = cm.top_complaints || [];
    if (rows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-dim)">No complaints mined — run spark_pipeline.py first.</td></tr>';
    } else {
      tbody.innerHTML = rows.map(c => `
        <tr>
          <td><span style="font-weight:600">${c.phrase}</span></td>
          <td><span class="badge-ngram">${c.n_gram_type}</span></td>
          <td>${c.frequency.toLocaleString()}</td>
          <td>${c.percentage}%</td>
          <td><span class="aspect-chip">${c.aspect}</span></td>
        </tr>
      `).join('');
    }
  }

  // 2. Top Complaints frequency bar chart
  const chartComplaintsEl = document.getElementById('chart-complaints');
  if (chartComplaintsEl && (cm.top_complaints || []).length) {
    const top = (cm.top_complaints || []).slice(0, 10);
    new Chart(chartComplaintsEl, {
      type: 'bar',
      data: {
        labels: top.map(c => c.phrase),
        datasets: [{
          label: 'Frequency',
          data: top.map(c => c.frequency),
          backgroundColor: top.map(c => {
            const aspectColors = {
              'Battery': '#f5a623',
              'Build Quality': '#ff6b6b',
              'Price / Value': '#5b8def',
              'Customer Support': '#3ddc97',
              'Delivery / Packaging': '#b97cf3',
              'General Dissatisfaction': '#9aa3b4',
            };
            return aspectColors[c.aspect] || '#9aa3b4';
          }),
          borderRadius: 4,
        }],
      },
      options: {
        indexAxis: 'y',
        plugins: { legend: { display: false } },
        scales: {
          x: { beginAtZero: true, title: { display: true, text: 'Occurrences' } },
        },
      },
    });
  }

  // 3. Top Bigrams vs Trigrams grouped bar chart
  const chartNgramsEl = document.getElementById('chart-ngrams');
  if (chartNgramsEl) {
    const topBi  = (cm.top_bigrams  || []).slice(0, 8);
    const topTri = (cm.top_trigrams || []).slice(0, 8);
    const allLabels = [...new Set([...topBi.map(c => c.phrase), ...topTri.map(c => c.phrase)])];
    const biMap  = Object.fromEntries(topBi.map(c => [c.phrase, c.frequency]));
    const triMap = Object.fromEntries(topTri.map(c => [c.phrase, c.frequency]));
    new Chart(chartNgramsEl, {
      type: 'bar',
      data: {
        labels: allLabels,
        datasets: [
          {
            label: 'Bigrams',
            data: allLabels.map(l => biMap[l] || 0),
            backgroundColor: '#e6550d',
            borderRadius: 3,
          },
          {
            label: 'Trigrams',
            data: allLabels.map(l => triMap[l] || 0),
            backgroundColor: '#756bb1',
            borderRadius: 3,
          },
        ],
      },
      options: {
        indexAxis: 'y',
        plugins: { legend: { position: 'bottom' } },
        scales: {
          x: { beginAtZero: true, title: { display: true, text: 'Frequency' } },
        },
      },
    });
  }

  // 4. Aspect-to-Complaint breakdown grid
  const aspectGrid = document.getElementById('aspect-complaints-grid');
  if (aspectGrid) {
    const ac = cm.aspect_complaints || {};
    const aspectColors = {
      'Battery': '#f5a623',
      'Build Quality': '#ff6b6b',
      'Price / Value': '#5b8def',
      'Customer Support': '#3ddc97',
      'Delivery / Packaging': '#b97cf3',
      'General Dissatisfaction': '#9aa3b4',
    };
    aspectGrid.innerHTML = Object.entries(ac).map(([aspect, complaints]) => `
      <div class="pain-card">
        <div class="title" style="color:${aspectColors[aspect] || 'var(--text)'}">${aspect}</div>
        <div class="desc" style="margin-top:8px;">
          ${complaints.length === 0
            ? '<span style="color:var(--text-dim);font-size:12px;">No complaint phrases linked to this aspect.</span>'
            : complaints.map(c => `
                <div style="font-size:12px; padding:3px 0; border-bottom:1px solid var(--border); display:flex; justify-content:space-between;">
                  <span>${c.phrase}</span>
                  <span style="color:var(--text-dim); margin-left:8px; white-space:nowrap;">${c.frequency}×</span>
                </div>
              `).join('')
          }
        </div>
      </div>
    `).join('');
  }
}

function renderTopics(topics) {
  if (!topics || !topics.length) return;

  // 1. Bar chart: Cumulative top-term weight per topic
  const chartTopicsEl = document.getElementById('chart-topics');
  if (chartTopicsEl) {
    const topWeights = topics.map(t =>
      (t.terms_with_weights || []).slice(0, 4).reduce((s, w) => s + (w.weight || 0), 0)
    );
    const labels = topics.map(t => `Topic ${t.topic_id}\n(${(t.top_words || []).slice(0, 2).join(', ')})`);
    new Chart(chartTopicsEl, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Cumulative Top-Term Weight',
          data: topWeights,
          backgroundColor: ['#3182bd', '#e6550d', '#31a354', '#756bb1', '#f5a623'],
          borderRadius: 5,
        }],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, title: { display: true, text: 'Weight' } },
        },
      },
    });
  }

  // 2. Topic detail cards
  const topicsGrid = document.getElementById('topics-grid');
  if (topicsGrid) {
    const cardColors = ['#3182bd', '#e6550d', '#31a354', '#756bb1', '#f5a623'];
    topicsGrid.innerHTML = topics.map((t, i) => `
      <div class="pain-card">
        <div class="title" style="color:${cardColors[i % cardColors.length]}">${t.label}</div>
        <div class="rate" style="font-size:12px; color:var(--text-dim); margin-bottom:8px;">${t.statistical_interpretation}</div>
        <div class="desc">
          ${(t.terms_with_weights || []).slice(0, 6).map(tw => `
            <div style="display:flex; justify-content:space-between; font-size:12px; padding:2px 0; border-bottom:1px solid var(--border);">
              <span style="font-weight:500">${tw.term}</span>
              <span style="color:var(--text-dim)">${(tw.weight * 100).toFixed(2)}%</span>
            </div>
          `).join('')}
        </div>
        <div style="margin-top:8px; font-size:10.5px; color:var(--text-dim); font-style:italic;">${t.note}</div>
      </div>
    `).join('');
  }
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

    // Render Aspect Breakdown
    const aspectContainer = document.getElementById('aspect-results-container');
    const aspectList = document.getElementById('aspect-tags-list');
    if (aspectContainer && aspectList) {
      if (data.aspects && data.aspects.length > 0) {
        aspectContainer.classList.remove('hidden');
        aspectList.innerHTML = data.aspects.map(asp => `
          <div class="aspect-badge-item ${asp.sentiment === 'POSITIVE' ? 'pos' : 'neg'}">
            <div>
              <div class="aspect-badge-name">${asp.aspect}</div>
              <div class="aspect-badge-context">"${asp.context}"</div>
            </div>
            <span class="aspect-badge-sent ${asp.sentiment === 'POSITIVE' ? 'pos' : 'neg'}">
              ${asp.sentiment} (${asp.confidence}%)
            </span>
          </div>
        `).join('');
      } else {
        aspectContainer.classList.remove('hidden');
        aspectList.innerHTML = '<div style="font-size:12.5px; color:var(--text-dim); padding:4px 0;">No predefined product aspects (Battery, Build Quality, Price, Support, Delivery) detected in this review.</div>';
      }
    }

    // Render Complaint Patterns
    const complaintsContainer = document.getElementById('complaints-results-container');
    const complaintsList = document.getElementById('complaints-tags-list');
    if (complaintsContainer && complaintsList) {
      complaintsContainer.classList.remove('hidden');
      if (data.complaints && data.complaints.length > 0) {
        complaintsList.innerHTML = data.complaints.map(c => `
          <div class="aspect-badge-item neg" style="flex-direction:column; align-items:flex-start; gap:2px;">
            <div style="display:flex; justify-content:space-between; width:100%; align-items:center;">
              <span class="aspect-badge-name">⚠️ ${c.phrase}</span>
              <span class="aspect-badge-sent neg" style="font-size:10.5px;">${c.n_gram_type}</span>
            </div>
            <span style="font-size:11.5px; color:var(--text-dim);">Aspect: ${c.aspect}</span>
          </div>
        `).join('');
      } else {
        complaintsList.innerHTML = '<div style="font-size:12.5px; color:var(--text-dim); padding:4px 0;">No recurring complaint patterns detected in this review.</div>';
      }
    }
  } catch (e) {
    alert('Prediction failed: ' + e.message + '\n\nMake sure api.py is running and spark_pipeline.py has completed.');
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze Review';
  }
});

loadResults();
