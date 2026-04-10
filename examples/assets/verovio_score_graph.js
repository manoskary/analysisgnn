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
    cards.push(`<div class="agn-card"><div class="agn-label">Score Note ID</div><div class="agn-value">${escapeHtml(note.note_id || "")}</div></div>`);
    if (note.table_note_id != null && String(note.table_note_id).trim() !== "") {
      cards.push(`<div class="agn-card"><div class="agn-label">Table Note ID</div><div class="agn-value">${escapeHtml(note.table_note_id)}</div></div>`);
    }
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

  function findNoteElementById(noteId) {
    if (!noteId) return null;
    const raw = String(noteId);
    const direct = document.getElementById(raw);
    if (direct) return direct;
    const noteEls = iterNoteElements(pagesEl);
    for (const noteEl of noteEls) {
      const nid = String(noteEl.id || "");
      if (!nid) continue;
      if (nid === raw || nid.endsWith(raw) || nid.includes(raw)) {
        return noteEl;
      }
    }
    return null;
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
      const mappedIdx = new Set();
      // First pass: map by note id.
      for (let i = 0; i < notes.length; i += 1) {
        const note = notes[i];
        const noteIndex = Number(note && note.index != null ? note.index : i);
        const noteId = note && note.note_id ? String(note.note_id) : "";
        if (!noteId) continue;
        const element = findNoteElementById(noteId);
        if (!element) continue;
        const noteGroup = element.classList && element.classList.contains("note")
          ? element
          : element.closest("g.note");
        if (!noteGroup) continue;
        const pageMargin = noteGroup.closest(".page-margin");
        if (!pageMargin) continue;
        const xy = glyphXY(noteGroup);
        if (!xy) continue;
        noteMap.set(noteIndex, {
          note,
          noteGroup,
          pageMargin,
          x: xy.x,
          y: xy.y,
        });
        mappedIdx.add(noteIndex);
      }
      // Fallback: map remaining rows by sequence order.
      let renderIdx = 0;
      for (let i = 0; i < notes.length; i += 1) {
        const note = notes[i];
        const noteIndex = Number(note && note.index != null ? note.index : i);
        if (mappedIdx.has(noteIndex)) continue;
        if (renderIdx >= allRenderedNotes.length) break;
        const noteGroup = allRenderedNotes[renderIdx];
        renderIdx += 1;
        const pageMargin = noteGroup.closest(".page-margin");
        if (!pageMargin) continue;
        const xy = glyphXY(noteGroup);
        if (!xy) continue;
        noteMap.set(noteIndex, {
          note,
          noteGroup,
          pageMargin,
          x: xy.x,
          y: xy.y,
        });
      }

      // Apply note colors from payload (e.g., NCT coloring).
      // Uses CSS custom property --agn-note-color + class .agn-colored so that
      // the active-note highlight (.agn-active with !important) still overrides.
      const noteColors = payload.note_colors || {};
      for (const [noteIndexStr, color] of Object.entries(noteColors)) {
        const item = noteMap.get(Number(noteIndexStr));
        if (!item || !item.noteGroup || !color) continue;
        item.noteGroup.classList.add("agn-colored");
        item.noteGroup.style.setProperty("--agn-note-color", color);
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

      function renderRomanNumeralOverlay() {
        const spans = (((payload.meta || {}).roman_spans) || []);
        if (!Array.isArray(spans) || spans.length === 0) return 0;
        const onsetAnchors = new Map();
        for (const item of noteMap.values()) {
          const onset = Number(item.note && item.note.onset_div);
          if (!Number.isFinite(onset)) continue;
          const prev = onsetAnchors.get(onset);
          if (!prev || item.x < prev.x) {
            onsetAnchors.set(onset, item);
          }
        }
        const pageLabelY = new Map();
        let drawn = 0;
        for (const span of spans) {
          const onset = Number(span.start_onset_div);
          const label = String((span.label == null ? "" : span.label)).trim();
          if (!Number.isFinite(onset) || !label) continue;
          const anchor = onsetAnchors.get(onset);
          if (!anchor) continue;
          let y = pageLabelY.get(anchor.pageMargin);
          if (y == null) {
            try {
              const box = anchor.pageMargin.getBBox();
              y = Number(box.y + box.height + 22);
            } catch (_err) {
              y = Number(anchor.y + 28);
            }
            pageLabelY.set(anchor.pageMargin, y);
          }
          const textEl = document.createElementNS("http://www.w3.org/2000/svg", "text");
          textEl.setAttribute("x", String(anchor.x));
          textEl.setAttribute("y", String(y));
          textEl.setAttribute("class", "agn-rn-label");
          textEl.textContent = label;
          anchor.pageMargin.appendChild(textEl);
          drawn += 1;
        }
        return drawn;
      }

      function setActiveNote(noteIndex) {
        for (const item of noteMap.values()) {
          item.noteGroup.classList.remove("agn-active");
        }
        const item = noteMap.get(Number(noteIndex));
        if (!item) return;
        item.noteGroup.classList.add("agn-active");
      }

      function highlightIncident(noteIndex) {
        const r = String(noteIndex);
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

      for (const [noteIndex, item] of noteMap.entries()) {
        item.noteGroup.addEventListener("click", () => {
          setActiveNote(noteIndex);
          highlightIncident(noteIndex);
          renderPanel(item.note);
        });
      }
      const rnLabelCount = renderRomanNumeralOverlay();

      setStatus(
        `Rendered ${notes.length} notes in horizontal continuous view (${pageCount} svg page fragment(s)). ` +
        `Visible edges: ${Array.from(visibleTypes).join(", ") || "none"}. ` +
        `RN labels: ${rnLabelCount}`
      );
      renderPanel(null);
    } catch (error) {
      console.error(error);
      setStatus(`Rendering error: ${String(error && error.message ? error.message : error)}`);
    }
  }

  render();
})();
