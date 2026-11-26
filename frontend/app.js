/**
 * Dashboard operativo del IDS para el equipo de TI.
 * Consume exclusivamente la API REST local (/ready, /metrics, /predict, /alerts)
 * sin lógica de preprocesamiento duplicada ni entrenamiento.
 */
(() => {
  "use strict";

  const REFRESH_MS = 5000;

  // Descripción breve de cada categoría de UNSW-NB15 para el operador.
  const ATTACK_INFO = {
    Analysis: "Escaneo de puertos, spam o intrusiones vía HTML.",
    Backdoor: "Acceso remoto oculto que evade la autenticación.",
    DoS: "Intento de saturar un servicio para dejarlo inaccesible.",
    Exploits: "Explotación de una vulnerabilidad conocida.",
    Fuzzers: "Envío masivo de datos aleatorios para provocar fallos.",
    Generic: "Ataque genérico contra cifrado o protocolos.",
    Reconnaissance: "Reconocimiento de la red para recopilar información.",
    Shellcode: "Inyección de código para tomar control de un equipo.",
    Worms: "Malware que se replica y propaga por la red.",
  };

  const state = { exampleInput: [] };

  const escapeHtml = (str) =>
    String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");

  const fmtDate = (iso) => {
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleString("es-ES");
  };

  const fmtBytes = (n) => {
    const v = Number(n) || 0;
    if (v < 1024) return `${v} B`;
    if (v < 1024 ** 2) return `${(v / 1024).toFixed(1)} KB`;
    return `${(v / 1024 ** 2).toFixed(1)} MB`;
  };

  const fmtDur = (s) => {
    const v = Number(s);
    if (!Number.isFinite(v)) return "—";
    return v < 1 ? `${(v * 1000).toFixed(0)} ms` : `${v.toFixed(2)} s`;
  };

  function initTabs() {
    const buttons = document.querySelectorAll(".tab-btn");
    const panels = document.querySelectorAll(".tab-panel");
    buttons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = btn.getAttribute("data-tab");
        buttons.forEach((b) => {
          b.classList.toggle("active", b === btn);
          b.setAttribute("aria-selected", b === btn ? "true" : "false");
        });
        panels.forEach((p) => p.classList.toggle("active", p.id === target));
      });
    });
  }

  // Estado del sistema (/ready)
  async function loadReadiness() {
    const dot = document.getElementById("status-dot");
    const text = document.getElementById("status-text");
    try {
      const resp = await fetch("/ready");
      const data = await resp.json();
      if (resp.ok && data.ready) {
        dot.className = "status-dot online";
        text.textContent = "Sistema operativo";
      } else {
        dot.className = "status-dot offline";
        text.textContent = "Modelo no disponible";
      }
    } catch (_) {
      dot.className = "status-dot offline";
      text.textContent = "Sin conexión con el servidor";
    }
  }

  // Ejemplo real para la pestaña de análisis (/metrics expone example_input)
  async function loadExample() {
    try {
      const resp = await fetch("/metrics");
      if (!resp.ok) return;
      const data = await resp.json();
      if (Array.isArray(data.example_input)) state.exampleInput = data.example_input;
    } catch (_) {
      // El ejemplo es opcional
    }
  }

  // Análisis de flujos (POST /predict)
  async function runPrediction() {
    const errorBox = document.getElementById("predict-error-box");
    const summaryBar = document.getElementById("predict-summary-bar");
    const rawBody = document.getElementById("predict-json-input").value.trim();
    errorBox.classList.add("hidden");

    if (!rawBody) {
      showError("Introduzca o cargue al menos un flujo para analizar.");
      return;
    }

    try {
      const resp = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: rawBody,
      });
      const data = await resp.json();
      if (!resp.ok) {
        summaryBar.classList.add("hidden");
        showError(`No se pudo analizar: ${data.detail || data.error || `HTTP ${resp.status}`}`);
        return;
      }
      summaryBar.classList.remove("hidden");
      document.getElementById("res-count").textContent = String(data.count);
      document.getElementById("res-attacks").textContent = String(data.attacks_detected);
      renderResults(data.predictions || []);
      await loadAlerts();
    } catch (err) {
      showError(`Error de conexión con el servidor: ${err}`);
    }
  }

  function showError(msg) {
    const errorBox = document.getElementById("predict-error-box");
    errorBox.textContent = msg;
    errorBox.classList.remove("hidden");
  }

  function renderResults(predictions) {
    const container = document.getElementById("predict-results-list");
    if (!predictions.length) {
      container.innerHTML = `<p class="empty-state">Sin resultados.</p>`;
      return;
    }
    container.innerHTML = predictions
      .map((p, idx) => {
        const isAttack = Number(p.label) === 1;
        const title = isAttack
          ? `Ataque: ${escapeHtml(p.attack_cat)}`
          : "Tráfico normal";
        const desc = isAttack ? escapeHtml(ATTACK_INFO[p.attack_cat] || "") : "Sin actividad sospechosa.";
        const warn =
          Array.isArray(p.unknown_categories) && p.unknown_categories.length > 0
            ? `<span class="badge badge-warn">Valores no reconocidos: ${escapeHtml(p.unknown_categories.join(", "))} — resultado menos fiable</span>`
            : "";
        return `
          <article class="pred-item ${isAttack ? "is-attack" : ""}">
            <div class="pred-header">
              <strong>Flujo #${idx + 1}</strong>
              <span class="badge ${isAttack ? "badge-attack" : "badge-ok"}">${title}</span>
            </div>
            <div class="pred-meta"><span>${desc}</span>${warn}</div>
          </article>
        `;
      })
      .join("");
  }

  // Alertas (GET /alerts)
  async function loadAlerts() {
    const catSelect = document.getElementById("filter-attack-cat");
    const limitSelect = document.getElementById("filter-limit");
    const tbody = document.getElementById("alerts-tbody");
    const chipsBar = document.getElementById("alerts-category-chips");

    const params = new URLSearchParams();
    params.set("limit", limitSelect.value);
    if (catSelect.value) params.set("attack_cat", catSelect.value);

    try {
      const resp = await fetch(`/alerts?${params.toString()}`);
      if (!resp.ok) return;
      const data = await resp.json();

      document.getElementById("kpi-total-alerts").textContent = String(data.total_alerts ?? 0);
      const entries = Object.entries(data.by_category || {});
      document.getElementById("kpi-top-category").textContent =
        entries.length > 0 ? `${entries[0][0]} (${entries[0][1]})` : "—";
      chipsBar.innerHTML = entries
        .map(([cat, cnt]) => `<span class="badge badge-attack">${escapeHtml(cat)}: ${cnt}</span>`)
        .join("");

      const alerts = data.alerts || [];
      if (!catSelect.value) {
        document.getElementById("kpi-last-alert").textContent =
          alerts.length > 0 ? fmtDate(alerts[0].created_utc) : "—";
      }

      document.getElementById("alerts-updated").textContent =
        `Última actualización: ${new Date().toLocaleTimeString("es-ES")} · se actualiza cada 5 s.`;

      if (!alerts.length) {
        tbody.innerHTML = `<tr><td colspan="7" class="empty-state">No hay alertas registradas.</td></tr>`;
        return;
      }

      tbody.innerHTML = alerts
        .map((a) => {
          const fs = a.flow_summary || {};
          const service = fs.service && fs.service !== "-" ? fs.service : "sin servicio";
          return `
            <tr>
              <td>#${a.id}</td>
              <td>${escapeHtml(fmtDate(a.created_utc))}</td>
              <td><span class="badge badge-attack">${escapeHtml(a.attack_cat)}</span></td>
              <td>${escapeHtml(ATTACK_INFO[a.attack_cat] || "")}</td>
              <td>${escapeHtml(String(fs.proto ?? "—").toUpperCase())} / ${escapeHtml(service)}</td>
              <td>${fmtDur(fs.dur)}</td>
              <td>${fmtBytes(fs.sbytes)} / ${fmtBytes(fs.dbytes)}</td>
            </tr>
          `;
        })
        .join("");
    } catch (_) {
      // Errores transitorios: se reintenta en el siguiente ciclo
    }
  }

  function bindEvents() {
    document.getElementById("btn-run-predict").addEventListener("click", runPrediction);

    document.getElementById("btn-load-example").addEventListener("click", () => {
      if (state.exampleInput.length > 0) {
        document.getElementById("predict-json-input").value = JSON.stringify(state.exampleInput, null, 2);
      }
    });

    document.getElementById("file-input").addEventListener("change", async (ev) => {
      const file = ev.target.files && ev.target.files[0];
      if (!file) return;
      document.getElementById("predict-json-input").value = await file.text();
      ev.target.value = "";
    });

    document.getElementById("btn-refresh-alerts").addEventListener("click", loadAlerts);
    document.getElementById("filter-attack-cat").addEventListener("change", loadAlerts);
    document.getElementById("filter-limit").addEventListener("change", loadAlerts);
  }

  document.addEventListener("DOMContentLoaded", async () => {
    initTabs();
    bindEvents();
    await Promise.all([loadReadiness(), loadExample(), loadAlerts()]);
    setInterval(() => {
      loadReadiness();
      loadAlerts();
    }, REFRESH_MS);
  });
})();
