/* AB Reader - lettore EPUB con voce Kokoro.
 *
 * La riproduzione usa la Web Audio API invece di <audio>: ogni frase e' un
 * buffer schedulato a un istante assoluto del contesto audio, quindi sapere
 * quale parola sta suonando e' una semplice sottrazione, non una stima.
 */

'use strict';

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- utilità

async function api(path, options = {}) {
  const opts = { ...options };
  if (opts.body && !(opts.body instanceof FormData)) {
    opts.headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
    opts.body = JSON.stringify(opts.body);
  }
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* risposta non json */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

let toastTimer = null;
function toast(msg, ms = 2600) {
  const el = $('toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, ms);
}

function debounce(fn, ms) {
  let t = null;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// ---------------------------------------------------------------- stato

const state = {
  settings: {},
  voices: [],
  book: null,          // { id, title, chapters: [...], language, ... }
  chapter: null,       // { index, title, blocks, sentences }
  sentEls: [],
  wordEls: [],
  currentSentence: -1,
  currentWord: -1,
  noteAnchor: null,    // { sentence_idx, text }
};

const LOOKAHEAD = 3;       // frasi schedulate in anticipo
const PREFETCH_AHEAD = 4;  // frasi messe in coda di sintesi sul server

// ---------------------------------------------------------------- audio

const player = {
  ctx: null,
  gain: null,
  playing: false,
  scheduled: [],     // { idx, source, startAt, endAt, offset, words }
  cursor: 0,         // prossima frase da schedulare
  nextStartAt: 0,
  pendingOffset: 0,  // secondi da saltare all'inizio della prima frase
  gen: 0,            // invalidato a ogni seek: le pump() vecchie si fermano
  pumping: false,
  finishing: false,  // cambio capitolo in corso
  gap: 0.05,         // respiro fra una frase e l'altra
};

let sentCache = new Map();  // indice frase -> Promise<{ buffer, words, duration }>

function ensureCtx() {
  if (!player.ctx) {
    player.ctx = new (window.AudioContext || window.webkitAudioContext)();
    player.gain = player.ctx.createGain();
    player.gain.connect(player.ctx.destination);
    requestAnimationFrame(tick);
  }
  return player.ctx;
}

function ttsBody(idx) {
  return {
    book_id: state.book.id,
    chapter_idx: state.chapter.index,
    sentence_idx: idx,
    voice: currentVoice(),
    speed: Number(state.settings.speed),
  };
}

function fetchSentence(idx) {
  if (sentCache.has(idx)) return sentCache.get(idx);
  const promise = (async () => {
    const meta = await api('/api/tts', { method: 'POST', body: ttsBody(idx) });
    const res = await fetch(meta.audio_url);
    const buffer = await player.ctx.decodeAudioData(await res.arrayBuffer());
    return { buffer, words: meta.words, duration: meta.duration, exact: meta.exact };
  })();
  promise.catch(() => sentCache.delete(idx));  // un errore non deve restare in cache
  sentCache.set(idx, promise);
  return promise;
}

const prefetchAhead = debounce((from) => {
  const idx = [];
  for (let i = from + 1; i <= from + PREFETCH_AHEAD; i++) {
    if (i < state.chapter.sentences.length && !sentCache.has(i)) idx.push(i);
  }
  if (idx.length) {
    api('/api/tts/prefetch', {
      method: 'POST',
      body: { ...ttsBody(0), sentence_idx: idx },
    }).catch(() => { /* il prefetch e' opzionale */ });
  }
}, 120);

async function pump() {
  if (player.pumping || !player.playing) return;
  player.pumping = true;
  const gen = player.gen;
  try {
    while (player.playing && gen === player.gen &&
           player.scheduled.length < LOOKAHEAD &&
           player.cursor < state.chapter.sentences.length) {
      const idx = player.cursor;
      setStatus(sentCache.has(idx) ? '' : 'sintesi in corso…');
      let item;
      try {
        item = await fetchSentence(idx);
      } catch (err) {
        setStatus('');
        toast('Sintesi non riuscita: ' + err.message);
        pause();
        return;
      }
      if (gen !== player.gen || !player.playing) return;
      setStatus('');

      const offset = player.pendingOffset;
      player.pendingOffset = 0;
      const startAt = Math.max(player.ctx.currentTime + 0.08, player.nextStartAt);
      const src = player.ctx.createBufferSource();
      src.buffer = item.buffer;
      src.connect(player.gain);
      src.start(startAt, Math.min(offset, Math.max(0, item.buffer.duration - 0.05)));

      const endAt = startAt + item.buffer.duration - offset;
      player.scheduled.push({ idx, source: src, startAt, endAt, offset, words: item.words });
      player.nextStartAt = endAt + player.gap;
      player.cursor = idx + 1;
      prefetchAhead(idx);
    }
  } finally {
    player.pumping = false;
  }
}

function stopAll() {
  player.gen++;
  for (const s of player.scheduled) {
    try { s.source.onended = null; s.source.stop(); } catch (_) { /* già finita */ }
  }
  player.scheduled = [];
  player.nextStartAt = 0;
}

function findActive(now) {
  for (const s of player.scheduled) {
    if (now >= s.startAt && now < s.endAt) return s;
  }
  // Prima che parta la frase successiva, resta illuminata quella in arrivo.
  return player.scheduled.length && now < player.scheduled[0].startAt
    ? player.scheduled[0] : null;
}

function wordAt(words, t) {
  let lo = 0, hi = words.length - 1, best = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (words[mid][0] <= t) { best = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  return best;
}

function tick() {
  requestAnimationFrame(tick);
  if (!player.ctx || !state.chapter) return;

  const now = player.ctx.currentTime;
  // Le frasi già finite escono dalla coda.
  while (player.scheduled.length && player.scheduled[0].endAt <= now - 0.01) {
    player.scheduled.shift();
  }

  if (player.playing) {
    if (!player.scheduled.length && player.cursor >= state.chapter.sentences.length) {
      onChapterFinished();
      return;
    }
    if (player.scheduled.length < LOOKAHEAD) pump();
  }

  const active = findActive(now);
  if (!active) return;

  if (active.idx !== state.currentSentence) setActiveSentence(active.idx);

  const local = Math.max(0, now - active.startAt + active.offset);
  const w = wordAt(active.words, local);
  if (w !== state.currentWord) setActiveWord(w);
}

// ---------------------------------------------------------------- comandi

async function play() {
  if (!state.chapter) return;
  ensureCtx();
  if (player.ctx.state === 'suspended') await player.ctx.resume();
  player.playing = true;
  if (!player.scheduled.length) {
    // Nulla in coda: si riparte dal cursore, riavvolgendo se il capitolo è finito.
    player.nextStartAt = 0;
    if (player.cursor >= state.chapter.sentences.length) player.cursor = 0;
  }
  pump();
  updatePlayButton();
}

async function pause() {
  player.playing = false;
  if (player.ctx && player.ctx.state === 'running') await player.ctx.suspend();
  updatePlayButton();
  setStatus('');
}

function togglePlay() { player.playing ? pause() : play(); }

/** Riparte da una frase (opzionalmente da una parola precisa al suo interno). */
async function seek(sentenceIdx, wordIdx = 0, autoplay = null) {
  if (!state.chapter) return;
  const n = state.chapter.sentences.length;
  if (sentenceIdx < 0 || sentenceIdx >= n) return;

  const wasPlaying = autoplay === null ? player.playing : autoplay;
  ensureCtx();
  stopAll();
  player.cursor = sentenceIdx;
  player.pendingOffset = 0;

  setActiveSentence(sentenceIdx);
  setActiveWord(wordIdx > 0 ? wordIdx : -1);

  if (wordIdx > 0) {
    // Serve il tempo della parola: si prende dai metadati, già in cache o no.
    try {
      const item = await fetchSentence(sentenceIdx);
      if (item.words[wordIdx]) player.pendingOffset = Math.max(0, item.words[wordIdx][0]);
    } catch (_) { /* si riparte comunque dall'inizio della frase */ }
  }

  saveProgress(sentenceIdx, Math.max(0, wordIdx));
  if (wasPlaying) await play(); else updatePlayButton();
  prefetchAhead(sentenceIdx - 1);
}

function nextSentence() { seek(state.currentSentence + 1); }
function prevSentence() {
  // Se siamo a metà frase, il primo "indietro" torna al suo inizio.
  const active = player.scheduled.length ? player.scheduled[0] : null;
  const into = active && player.ctx ? player.ctx.currentTime - active.startAt : 0;
  seek(into > 1.2 ? state.currentSentence : state.currentSentence - 1);
}

async function onChapterFinished() {
  // tick() gira a ogni frame: senza guardia il capitolo successivo verrebbe
  // aperto decine di volte mentre la richiesta è ancora in volo.
  if (player.finishing) return;
  player.finishing = true;
  try {
    const next = state.chapter.index + 1;
    if (next >= state.book.chapters.length) {
      await pause();
      toast('Fine del libro.');
      return;
    }
    const wasPlaying = player.playing;
    await openChapter(next, 0, false);
    if (wasPlaying) await seek(0, 0, true);
  } finally {
    player.finishing = false;
  }
}

// ---------------------------------------------------------------- rendering

/** HTML del capitolo. Funzione pura: dipende solo dai dati, così è collaudabile. */
function chapterHtml(chapter) {
  const parts = [];
  const blocks = chapter.blocks;
  const sentencesOf = (b) => b.s.map((sid) => chapter.sentences[sid].t).join(' ');

  // Quasi sempre il primo blocco è il titolo stesso del capitolo. Va reso una
  // volta sola, ma restando fatto di frasi: anche il titolo viene letto.
  const first = blocks[0];
  const titleIsFirstBlock = first && /^h[1-6]$/.test(first.tag) &&
    sentencesOf(first).trim() === chapter.title.trim();
  if (titleIsFirstBlock) {
    parts.push(`<h1 class="chapter-title">${
      first.s.map((sid) => renderSentence(sid, chapter.sentences[sid])).join(' ')}</h1>`);
  } else {
    parts.push(`<h1 class="chapter-title">${escapeHtml(chapter.title)}</h1>`);
  }

  for (const block of blocks.slice(titleIsFirstBlock ? 1 : 0)) {
    const tag = /^h[1-6]$/.test(block.tag) ? block.tag
      : (block.tag === 'blockquote' ? 'blockquote' : 'p');
    const inner = block.s.map((sid) => renderSentence(sid, chapter.sentences[sid])).join(' ');
    parts.push(`<${tag}>${inner}</${tag}>`);
  }
  return parts.join('\n');
}

function renderChapter(chapter) {
  state.chapter = chapter;
  state.sentEls = new Array(chapter.sentences.length);
  state.wordEls = new Array(chapter.sentences.length);
  state.currentSentence = -1;
  state.currentWord = -1;

  const reader = $('reader');
  reader.innerHTML = chapterHtml(chapter);

  // Riferimenti presi una volta sola: l'evidenziazione gira a ogni frame e non
  // può permettersi una query nel DOM.
  for (const el of reader.querySelectorAll('.s')) state.sentEls[+el.dataset.s] = el;
  for (let i = 0; i < chapter.sentences.length; i++) {
    const host = state.sentEls[i];
    state.wordEls[i] = host ? Array.from(host.querySelectorAll('.w')) : [];
  }
}

function renderSentence(sid, sentence) {
  const text = sentence.t;
  let html = '', cursor = 0;
  sentence.w.forEach(([start, end], wi) => {
    if (start > cursor) html += escapeHtml(text.slice(cursor, start));
    html += `<span class="w" data-s="${sid}" data-w="${wi}">${escapeHtml(text.slice(start, end))}</span>`;
    cursor = end;
  });
  if (cursor < text.length) html += escapeHtml(text.slice(cursor));
  return `<span class="s" data-s="${sid}">${html}</span>`;
}

function setActiveSentence(idx) {
  const prev = state.sentEls[state.currentSentence];
  if (prev) prev.classList.remove('active');
  setActiveWord(-1);
  state.currentSentence = idx;
  const el = state.sentEls[idx];
  if (el) {
    el.classList.add('active');
    if (state.settings.auto_scroll) scrollIntoViewIfNeeded(el);
  }
  updatePosition();
  saveProgress(idx, 0);
}

function setActiveWord(wi) {
  const words = state.wordEls[state.currentSentence];
  if (words) {
    const prev = words[state.currentWord];
    if (prev) prev.classList.remove('active');
    const el = words[wi];
    if (el) el.classList.add('active');
  }
  state.currentWord = wi;
}

function scrollIntoViewIfNeeded(el) {
  const pane = $('reader-pane');
  const r = el.getBoundingClientRect();
  const p = pane.getBoundingClientRect();
  const margin = p.height * 0.22;
  if (r.top < p.top + margin || r.bottom > p.bottom - margin) {
    pane.scrollTop += r.top - p.top - p.height * 0.35;
  }
}

function updatePosition() {
  const n = state.chapter ? state.chapter.sentences.length : 0;
  const i = state.currentSentence;
  $('pos-label').textContent = n
    ? `Frase ${Math.max(0, i) + 1} di ${n} — ${state.chapter.title}`
    : '—';
  $('progress-fill').style.width = n ? `${((i + 1) / n) * 100}%` : '0';
}

function updatePlayButton() {
  $('btn-play').textContent = player.playing ? '❚❚' : '▶';
}

function setStatus(text) { $('status-label').textContent = text; }

// ---------------------------------------------------------------- progresso

const saveProgressNow = (sentenceIdx, wordIdx) => {
  if (!state.book || !state.chapter) return;
  api(`/api/books/${state.book.id}/progress`, {
    method: 'POST',
    body: {
      chapter_idx: state.chapter.index,
      sentence_idx: sentenceIdx,
      word_idx: wordIdx,
    },
  }).catch(() => { /* la posizione si risalva alla frase dopo */ });
};
const saveProgress = debounce(saveProgressNow, 900);

// ---------------------------------------------------------------- libreria

/** Megabyte in forma leggibile: "<1 MB", "340 MB", "1,2 GB". */
function formatMB(mb) {
  if (mb < 1) return '<1 MB';
  if (mb < 1000) return `${Math.round(mb)} MB`;
  return `${(mb / 1000).toFixed(1).replace('.', ',')} GB`;
}

async function loadLibrary() {
  const [books, cache] = await Promise.all([api('/api/books'), api('/api/cache')]);
  const all = $('btn-cache-all');
  all.hidden = !(cache.cache_mb > 0);
  all.textContent = `Svuota tutto l'audio (${formatMB(cache.cache_mb)})`;
  const list = $('book-list');
  $('lib-empty').hidden = books.length > 0;
  list.innerHTML = books.map((b) => {
    const pos = b.chapter_idx != null
      ? `ripresa: cap. ${b.chapter_idx + 1}, frase ${b.sentence_idx + 1}`
      : 'mai aperto';
    return `<li class="book-item" data-id="${b.id}">
      <div class="bi-main">
        <div class="bi-title">${escapeHtml(b.title)}</div>
        <div class="bi-sub">${escapeHtml(b.author || 'autore ignoto')} · ${b.n_chapters} capitoli · ${pos}</div>
      </div>
      <span class="lang-tag">${escapeHtml(b.language || '?')}</span>
      ${b.cache_mb > 0 ? `<button class="btn ghost small bi-cache" data-id="${b.id}"
        title="Cancella l'audio già sintetizzato di questo libro: verrà rifatto alla prossima lettura">
        Svuota audio (${formatMB(b.cache_mb)})</button>` : ''}
      <button class="btn ghost small bi-del" data-id="${b.id}" title="Rimuovi dalla libreria">Rimuovi</button>
    </li>`;
  }).join('');
}

async function uploadEpub(file) {
  const fd = new FormData();
  fd.append('file', file);
  $('lib-status').textContent = `Analizzo «${file.name}»…`;
  try {
    const book = await api('/api/books', { method: 'POST', body: fd });
    $('lib-status').textContent = '';
    await loadLibrary();
    toast(`«${book.title}» aggiunto (${book.n_chapters} capitoli, ${book.language}).`);
  } catch (err) {
    $('lib-status').textContent = '';
    toast('Import fallito: ' + err.message, 4000);
  }
}

// ---------------------------------------------------------------- apertura

async function openBook(bookId, resume = true) {
  const book = await api(`/api/books/${bookId}`);
  state.book = book;
  $('book-title').textContent = book.title;
  $('chapter-select').innerHTML = book.chapters.map((c) =>
    `<option value="${c.index}">${c.index + 1}. ${escapeHtml(c.title)}</option>`).join('');

  fillVoiceSelect();
  $('library').hidden = true;
  $('reader-view').hidden = false;

  const p = resume ? book.progress : { chapter_idx: 0, sentence_idx: 0, word_idx: 0 };
  await openChapter(p.chapter_idx || 0, p.sentence_idx || 0, false);
}

async function openChapter(idx, sentenceIdx = 0, autoplay = false) {
  stopAll();
  player.playing = false;
  sentCache = new Map();
  updatePlayButton();

  const chapter = await api(`/api/books/${state.book.id}/chapters/${idx}`);
  renderChapter(chapter);
  $('chapter-select').value = String(idx);
  $('reader-pane').scrollTop = 0;

  player.cursor = Math.min(sentenceIdx, chapter.sentences.length - 1);
  setActiveSentence(player.cursor);
  updatePosition();
  await loadNotes();
  prefetchAhead(player.cursor - 1);
  if (autoplay) await play();
}

function backToLibrary() {
  pause();
  stopAll();
  state.book = null;
  state.chapter = null;
  $('reader-view').hidden = true;
  $('library').hidden = false;
  loadLibrary();
}

// ---------------------------------------------------------------- voci

function currentVoice() {
  const sel = $('voice-select').value;
  return sel || (state.book && state.book.language === 'it'
    ? state.settings.voice_it : state.settings.voice_en);
}

function fillVoiceSelect() {
  const lang = state.book ? state.book.language : 'it';
  const preferred = lang === 'it' ? state.settings.voice_it : state.settings.voice_en;
  // Le voci della lingua del libro vengono prima, ma restano tutte selezionabili.
  const sorted = [...state.voices].sort((a, b) =>
    (a.lang === lang ? 0 : 1) - (b.lang === lang ? 0 : 1) || a.id.localeCompare(b.id));
  $('voice-select').innerHTML = sorted.map((v) =>
    `<option value="${v.id}">${escapeHtml(v.label)} · ${escapeHtml(v.lang_label)} (${v.gender})</option>`
  ).join('');
  $('voice-select').value = preferred;
}

async function onVoiceChanged() {
  const voice = $('voice-select').value;
  const key = state.book && state.book.language === 'it' ? 'voice_it' : 'voice_en';
  state.settings[key] = voice;
  await api('/api/settings', { method: 'POST', body: { [key]: voice } });
  await restartAfterAudioChange();
}

async function onSpeedChanged() {
  const v = Number($('speed').value);
  state.settings.speed = v;
  $('speed-label').textContent = v.toFixed(2) + '×';
  saveSpeed(v);
}
const saveSpeed = debounce(async (v) => {
  await api('/api/settings', { method: 'POST', body: { speed: v } });
  await restartAfterAudioChange();
}, 400);

/** Voce o velocità cambiate: l'audio in cache non vale più. */
async function restartAfterAudioChange() {
  const wasPlaying = player.playing;
  const idx = Math.max(0, state.currentSentence);
  stopAll();
  sentCache = new Map();
  player.cursor = idx;
  player.pendingOffset = 0;
  if (wasPlaying) { player.playing = true; await play(); }
  prefetchAhead(idx - 1);
}

// ---------------------------------------------------------------- appunti

async function loadNotes() {
  if (!state.book) return;
  const scope = $('notes-scope').value;
  const q = scope === 'chapter' ? `?chapter=${state.chapter.index}` : '';
  const notes = await api(`/api/books/${state.book.id}/notes${q}`);
  renderNotes(notes);
}

function renderNotes(notes) {
  const list = $('note-list');
  if (!notes.length) {
    list.innerHTML = '<li class="muted" style="font-size:13px">Nessun appunto qui.</li>';
    return;
  }
  list.innerHTML = notes.map((n) => {
    const when = new Date(n.created_at * 1000).toLocaleString('it-IT',
      { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
    const anchor = n.anchor
      ? `<div class="n-anchor" data-goto="${n.chapter_idx}:${n.sentence_idx ?? 0}"
             title="Vai a questo punto">${escapeHtml(n.anchor)}</div>`
      : '';
    return `<li class="note-item" data-id="${n.id}">
      ${anchor}
      <div class="n-body">${escapeHtml(n.body)}</div>
      <div class="n-foot">
        <span>${when}${n.chapter_idx !== state.chapter.index ? ` · cap. ${n.chapter_idx + 1}` : ''}</span>
        <span class="spacer"></span>
        <button data-act="edit">modifica</button>
        <button data-act="del">elimina</button>
      </div>
    </li>`;
  }).join('');
}

function setNoteAnchorFromCurrent() {
  if (!state.chapter) return;
  const idx = Math.max(0, state.currentSentence);
  const sentence = state.chapter.sentences[idx];
  if (!sentence) return;
  state.noteAnchor = { sentence_idx: idx, text: sentence.t.slice(0, 220) };
  showNoteAnchor();
}

function showNoteAnchor() {
  const el = $('note-anchor');
  if (!state.noteAnchor) { el.hidden = true; return; }
  el.textContent = state.noteAnchor.text;
  el.hidden = false;
}

async function saveNote() {
  const body = $('note-input').value.trim();
  if (!body) { toast('Scrivi qualcosa prima di salvare.'); return; }
  if (!state.noteAnchor) setNoteAnchorFromCurrent();
  await api(`/api/books/${state.book.id}/notes`, {
    method: 'POST',
    body: {
      chapter_idx: state.chapter.index,
      sentence_idx: state.noteAnchor ? state.noteAnchor.sentence_idx : state.currentSentence,
      anchor: state.noteAnchor ? state.noteAnchor.text : '',
      body,
    },
  });
  $('note-input').value = '';
  state.noteAnchor = null;
  showNoteAnchor();
  await loadNotes();
  toast('Appunto salvato.');
}

// ---------------------------------------------------------------- eventi

function bindEvents() {
  // libreria
  $('btn-add').onclick = () => $('file-input').click();
  $('file-input').onchange = (e) => {
    if (e.target.files[0]) uploadEpub(e.target.files[0]);
    e.target.value = '';
  };
  $('btn-cache-all').onclick = async () => {
    if (!confirm("Cancellare l'audio già sintetizzato di tutti i libri? Verrà rifatto alla prossima lettura.")) return;
    const res = await api('/api/cache/clear', { method: 'POST' });
    toast(`Cache audio svuotata: liberati ${formatMB(res.freed_mb)}.`);
    await loadLibrary();
  };
  $('book-list').onclick = async (e) => {
    const clear = e.target.closest('.bi-cache');
    if (clear) {
      e.stopPropagation();
      if (!confirm("Cancellare l'audio già sintetizzato di questo libro? Verrà rifatto alla prossima lettura.")) return;
      const res = await api(`/api/books/${clear.dataset.id}/cache/clear`, { method: 'POST' });
      toast(`Audio del libro cancellato: liberati ${formatMB(res.freed_mb)}.`);
      await loadLibrary();
      return;
    }
    const del = e.target.closest('.bi-del');
    if (del) {
      e.stopPropagation();
      if (!confirm('Rimuovere questo libro dalla libreria? Gli appunti verranno cancellati.')) return;
      await api(`/api/books/${del.dataset.id}`, { method: 'DELETE' });
      await loadLibrary();
      return;
    }
    const item = e.target.closest('.book-item');
    if (item) openBook(item.dataset.id).catch((err) => toast('Apertura fallita: ' + err.message));
  };
  $('btn-library').onclick = backToLibrary;

  // riproduzione
  $('btn-play').onclick = togglePlay;
  $('btn-next').onclick = nextSentence;
  $('btn-prev').onclick = prevSentence;
  $('btn-resume').onclick = async () => {
    const book = await api(`/api/books/${state.book.id}`);
    const p = book.progress;
    if (p.chapter_idx !== state.chapter.index) await openChapter(p.chapter_idx, p.sentence_idx, false);
    await seek(p.sentence_idx, p.word_idx || 0, false);
    toast(`Ripreso da capitolo ${p.chapter_idx + 1}, frase ${p.sentence_idx + 1}.`);
  };
  $('progress-track').onclick = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const n = state.chapter.sentences.length;
    const ratio = (e.clientX - r.left) / r.width;
    seek(Math.min(n - 1, Math.max(0, Math.floor(ratio * n))));
  };

  // clic su una parola: riparte da lì
  $('reader').onclick = (e) => {
    const w = e.target.closest('.w');
    if (!w) return;
    seek(+w.dataset.s, +w.dataset.w);
  };

  // selezione di testo: diventa l'ancora dell'appunto
  $('reader').onmouseup = () => {
    const sel = window.getSelection();
    const text = sel ? sel.toString().trim() : '';
    if (text.length > 2) {
      const node = sel.anchorNode;
      const host = node && (node.nodeType === 1 ? node : node.parentElement);
      const sentEl = host ? host.closest('.s') : null;
      state.noteAnchor = {
        sentence_idx: sentEl ? +sentEl.dataset.s : Math.max(0, state.currentSentence),
        text: text.slice(0, 220),
      };
      showNoteAnchor();
    }
  };

  // navigazione
  $('chapter-select').onchange = (e) => openChapter(+e.target.value, 0, false);
  $('voice-select').onchange = onVoiceChanged;
  $('speed').oninput = onSpeedChanged;

  // aspetto
  $('theme-select').onchange = (e) => applySetting('theme', e.target.value);
  $('btn-font-up').onclick = () => applySetting('font_size', Math.min(30, state.settings.font_size + 1));
  $('btn-font-down').onclick = () => applySetting('font_size', Math.max(13, state.settings.font_size - 1));
  $('btn-notes-toggle').onclick = () => {
    $('notes-pane').classList.toggle('collapsed');
    $('splitter').hidden = $('notes-pane').classList.contains('collapsed');
  };

  // appunti
  $('btn-note-save').onclick = saveNote;
  $('btn-note-clear').onclick = () => {
    $('note-input').value = '';
    state.noteAnchor = null;
    showNoteAnchor();
  };
  $('note-input').onfocus = () => { if (!state.noteAnchor) setNoteAnchorFromCurrent(); };
  $('note-input').onkeydown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); saveNote(); }
  };
  $('notes-scope').onchange = loadNotes;
  $('note-list').onclick = onNoteListClick;

  // scorciatoie da tastiera
  document.onkeydown = (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
    if (typing || $('reader-view').hidden) return;
    if (e.key === ' ') { e.preventDefault(); togglePlay(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); nextSentence(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); prevSentence(); }
    else if (e.key === 'n') { e.preventDefault(); setNoteAnchorFromCurrent(); $('note-input').focus(); }
  };

  bindSplitter();
  bindDropZone();
}

function onNoteListClick(e) {
  const item = e.target.closest('.note-item');
  if (!item) return;
  const id = +item.dataset.id;

  const goto = e.target.closest('[data-goto]');
  if (goto) {
    const [ch, s] = goto.dataset.goto.split(':').map(Number);
    (async () => {
      if (ch !== state.chapter.index) await openChapter(ch, s, false);
      await seek(s, 0, false);
    })();
    return;
  }

  const act = e.target.dataset.act;
  if (act === 'del') {
    if (!confirm('Eliminare questo appunto?')) return;
    api(`/api/notes/${id}`, { method: 'DELETE' }).then(loadNotes);
  } else if (act === 'edit') {
    const bodyEl = item.querySelector('.n-body');
    const original = bodyEl.textContent;
    const ta = document.createElement('textarea');
    ta.value = original;
    ta.rows = Math.min(10, original.split('\n').length + 2);
    bodyEl.replaceWith(ta);
    ta.focus();
    const commit = async () => {
      const text = ta.value.trim();
      if (text && text !== original) await api(`/api/notes/${id}`, { method: 'PUT', body: { body: text } });
      await loadNotes();
    };
    ta.onblur = commit;
    ta.onkeydown = (ev) => {
      if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) ta.blur();
      if (ev.key === 'Escape') { ta.onblur = null; loadNotes(); }
    };
  }
}

function bindSplitter() {
  const splitter = $('splitter');
  let dragging = false;
  splitter.onmousedown = (e) => { dragging = true; e.preventDefault(); };
  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const w = Math.min(640, Math.max(220, window.innerWidth - e.clientX));
    document.documentElement.style.setProperty('--notes-w', w + 'px');
  });
  document.addEventListener('mouseup', () => { dragging = false; });
}

function bindDropZone() {
  const overlay = $('drop-overlay');
  let depth = 0;
  document.addEventListener('dragenter', (e) => {
    e.preventDefault();
    if (++depth === 1) overlay.hidden = false;
  });
  document.addEventListener('dragover', (e) => e.preventDefault());
  document.addEventListener('dragleave', () => { if (--depth <= 0) { depth = 0; overlay.hidden = true; } });
  document.addEventListener('drop', (e) => {
    e.preventDefault();
    depth = 0;
    overlay.hidden = true;
    const file = e.dataTransfer.files[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.epub')) { toast('Serve un file .epub'); return; }
    if (!$('library').hidden) uploadEpub(file);
    else { backToLibrary(); uploadEpub(file); }
  });
}

// ---------------------------------------------------------------- impostazioni

function applySettings() {
  document.body.dataset.theme = state.settings.theme;
  document.documentElement.style.setProperty('--reader-font', state.settings.font_size + 'px');
  $('theme-select').value = state.settings.theme;
  $('speed').value = state.settings.speed;
  $('speed-label').textContent = Number(state.settings.speed).toFixed(2) + '×';
}

async function applySetting(key, value) {
  state.settings[key] = value;
  applySettings();
  await api('/api/settings', { method: 'POST', body: { [key]: value } });
}

// ---------------------------------------------------------------- avvio

async function main() {
  bindEvents();
  try {
    const st = await api('/api/state');
    state.settings = st.settings;
    state.voices = st.voices;
    applySettings();
    if (!st.models_ready) {
      toast('Modelli vocali non ancora scaricati: esegui setup.bat.', 8000);
    }
  } catch (err) {
    toast('Server non raggiungibile: ' + err.message, 8000);
    return;
  }
  await loadLibrary();
}

main();
