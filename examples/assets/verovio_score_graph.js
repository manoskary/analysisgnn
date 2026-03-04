(function () {
  const payload = window.__AGN_PAYLOAD__ || {};
  const pagesEl = document.getElementById("agn-pages");
  const statusEl = document.getElementById("agn-status");
  const panelEl = document.getElementById("agn-note-panel");
  const visibleTypes = new Set((payload.meta && payload.meta.visible_edge_types) || []);
  const edgeColor = {
    onset: "#2563eb",
    consecutive: "#dc2626",
    during: "#16a34a",
    rest: "#f59e0b",
  };

  function setStatus(msg) {
    if (statusEl) statusEl.textContent = msg;
  }

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function iterNoteElements(root) {
    return Array.from(root.querySelectorAll("g.note[id]"));
  }

  function resolveNoteGlyph(noteGroup) {
    if (!noteGroup) return null;
    const use = noteGroup.querySelector("use");
    if (use) return use;
    return noteGroup;
  }

  function glyphXY(noteGroup) {
    const glyph = resolveNoteGlyph(noteGroup);
    if (!glyph) return null;
    if (glyph.x && glyph.x.animVal && glyph.y && glyph.y.animVal) {
      const x = Number(glyph.x.animVal.value) + (glyph.width && glyph.width.animVal ? Number(glyph.width.animVal.value) / 5 : 0);
      const y = Number(glyph.y.animVal.value);
      if (Number.isFinite(x) && Number.isFinite(y)) return { x, y };
    }
    if (noteGroup.getBBox) {
      const box = noteGroup.getBBox();
      const x = Number(box.x + box.width * 0.5);
      const y = Number(box.y + box.height * 0.5);
      if (Number.isFinite(x) && Number.isFinite(y)) return { x, y };
    }
    return null;
  }

  function renderPanel(note) {
    if (!panelEl) return;
    if (!note) {
      panelEl.innerHTML = "No note selected.";
      return;
    }
    const taskEntries = Object.entries(note.tasks || {});
    const confEntries = Object.entries(note.confidence || {});
    const cards = [];
    cards.push(`<div class="agn-card"><div class="agn-label">Row</div><div class="agn-value">${escapeHtml(note.row)}</div></div>`);
    cards.push(`<div class="agn-card"><div class="agn-label">Note ID</div><div class="agn-value">${escapeHtml(note.note_id || "")}</div></div>`);
    cards.push(`<div class="agn-card"><div class="agn-label">Pitch</div><div class="agn-value">${escapeHtml(note.pitch_spelling || "")} (${escapeHtml(note.pitch_midi || "")})</div></div>`);
    cards.push(`<div class="agn-card"><div class="agn-label">Timing</div><div class="agn-value">m${escapeHtml(note.measure || "")} @ ${escapeHtml(note.onset_beat || "")}</div></div>`);
    cards.push(`<div class="agn-card"><div class="agn-label">Complete RN</div><div class="agn-value">${escapeHtml(note.romanNumeral_full || "")}</div></div>`);
    for (const [task, value] of taskEntries) {
      const conf = confEntries.find(([k]) => k === task);
      const confTxt = conf ? ` (${Number(conf[1]).toFixed(3)})` : "";
      cards.push(
        `<div class="agn-card"><div class="agn-label">${escapeHtml(task)}</div><div class="agn-value">${escapeHtml(value)}${escapeHtml(confTxt)}</div></div>`
      );
    }
    panelEl.innerHTML = `<h3>Note Analysis</h3><div class="agn-grid">${cards.join("")}</div>`;
  }

  function ensureVerovioReady() {
    return new Promise((resolve, reject) => {
      const timeoutMs = 30000;
      const startTs = Date.now();
      let lastInstantiateError = null;

      function failWith(reason) {
        const detail = lastInstantiateError && lastInstantiateError.message
          ? ` Last instantiate error: ${lastInstantiateError.message}`
          : "";
        reject(new Error(`${reason}${detail}`));
      }

      function tick() {
        if (!window.verovio || typeof window.verovio.toolkit !== "function") {
          if (Date.now() - startTs > timeoutMs) {
            failWith(
              "Verovio script/runtime missing. Check network/CSP for https://www.verovio.org/."
            );
            return;
          }
          window.setTimeout(tick, 120);
          return;
        }

        const mod = window.verovio.module;
        const runtimeReady = !mod || mod.calledRun || mod.runtimeInitialized;
        if (!runtimeReady) {
          if (Date.now() - startTs > timeoutMs) {
            failWith(
              "Verovio runtime did not become ready. Check browser console for wasm loading errors."
            );
            return;
          }
          window.setTimeout(tick, 120);
          return;
        }

        try {
          const tk = new window.verovio.toolkit();
          resolve(tk);
          return;
        } catch (err) {
          lastInstantiateError = err;
          if (Date.now() - startTs > timeoutMs) {
            failWith(
              "Verovio toolkit did not initialize. Check browser console for network/CSP errors."
            );
            return;
          }
          window.setTimeout(tick, 180);
        }
      }

      tick();
    });
  }

  async function render() {
    try {
      if (!payload || !payload.score_xml) {
        setStatus("No score payload available. Run inference first.");
        return;
      }
      const tk = await ensureVerovioReady();
      tk.setOptions({
        breaks: "none",
        footer: "none",
        header: "none",
        adjustPageHeight: true,
        adjustPageWidth: true,
        pageMarginBottom: 0,
        pageMarginTop: 0,
      });
      tk.loadData(payload.score_xml);
      const pageCount = Math.max(1, Number(tk.getPageCount() || 1));
      pagesEl.innerHTML = "";
      const noteMap = new Map();
      const notes = Array.isArray(payload.notes) ? payload.notes : [];

      for (let page = 1; page <= pageCount; page += 1) {
        const svgString = tk.renderToSVG(page, {});
        const svg = new DOMParser().parseFromString(svgString, "image/svg+xml").documentElement;
        const pageWrap = document.createElement("div");
        pageWrap.className = "agn-page";
        pageWrap.dataset.page = String(page);
        pageWrap.appendChild(svg);
        pagesEl.appendChild(pageWrap);
      }

      const allRenderedNotes = iterNoteElements(pagesEl);
      const mappedRows = new Set();
      // First pass: map by note id.
      for (const note of notes) {
        const noteId = note && note.note_id ? String(note.note_id) : "";
        if (!noteId) continue;
        const element = document.getElementById(noteId);
        if (!element) continue;
        const noteGroup = element.classList && element.classList.contains("note")
          ? element
          : element.closest("g.note");
        if (!noteGroup) continue;
        const pageMargin = noteGroup.closest(".page-margin");
        if (!pageMargin) continue;
        const xy = glyphXY(noteGroup);
        if (!xy) continue;
        noteMap.set(Number(note.row), {
          note,
          noteGroup,
          pageMargin,
          x: xy.x,
          y: xy.y,
        });
        mappedRows.add(Number(note.row));
      }
      // Fallback: map remaining rows by sequence order.
      let renderIdx = 0;
      for (const note of notes) {
        const row = Number(note.row);
        if (mappedRows.has(row)) continue;
        if (renderIdx >= allRenderedNotes.length) break;
        const noteGroup = allRenderedNotes[renderIdx];
        renderIdx += 1;
        const pageMargin = noteGroup.closest(".page-margin");
        if (!pageMargin) continue;
        const xy = glyphXY(noteGroup);
        if (!xy) continue;
        noteMap.set(row, {
          note,
          noteGroup,
          pageMargin,
          x: xy.x,
          y: xy.y,
        });
      }

      const edgeEls = [];
      const edges = payload.edges || {};
      const edgeTypes = Object.keys(edgeColor);
      for (const edgeType of edgeTypes) {
        const pair = edges[edgeType];
        if (!Array.isArray(pair) || pair.length < 2) continue;
        const src = pair[0] || [];
        const dst = pair[1] || [];
        const n = Math.min(src.length, dst.length);
        for (let i = 0; i < n; i += 1) {
          const s = Number(src[i]);
          const d = Number(dst[i]);
          const a = noteMap.get(s);
          const b = noteMap.get(d);
          if (!a || !b) continue;
          if (a.pageMargin !== b.pageMargin) continue;
          const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
          path.setAttribute("d", `M ${a.x} ${a.y} L ${b.x} ${b.y}`);
          path.setAttribute("class", `agn-edge agn-edge-${edgeType} agn-src-${s} agn-dst-${d}`);
          path.setAttribute("stroke", edgeColor[edgeType]);
          path.setAttribute("stroke-width", "16");
          path.dataset.edgeType = edgeType;
          path.dataset.src = String(s);
          path.dataset.dst = String(d);
          path.style.display = visibleTypes.has(edgeType) ? "block" : "none";
          a.pageMargin.appendChild(path);
          edgeEls.push(path);
        }
      }

      function setActiveRow(row) {
        for (const item of noteMap.values()) {
          item.noteGroup.classList.remove("agn-active");
        }
        const item = noteMap.get(Number(row));
        if (!item) return;
        item.noteGroup.classList.add("agn-active");
      }

      function highlightIncident(row) {
        const r = String(row);
        for (const edge of edgeEls) {
          const isVisibleType = visibleTypes.has(edge.dataset.edgeType);
          const incident = edge.dataset.src === r || edge.dataset.dst === r;
          if (!isVisibleType) {
            edge.style.display = "none";
            continue;
          }
          edge.style.display = "block";
          edge.style.strokeOpacity = incident ? "0.9" : "0.18";
        }
      }

      for (const [row, item] of noteMap.entries()) {
        item.noteGroup.addEventListener("click", () => {
          setActiveRow(row);
          highlightIncident(row);
          renderPanel(item.note);
        });
      }

      setStatus(
        `Rendered ${notes.length} notes in horizontal continuous view (${pageCount} svg page fragment(s)). ` +
        `Visible edges: ${Array.from(visibleTypes).join(", ") || "none"}`
      );
      renderPanel(null);
    } catch (error) {
      console.error(error);
      setStatus(`Rendering error: ${String(error && error.message ? error.message : error)}`);
    }
  }

  render();
})();
