// The page is a view and nothing more. Every number it draws is read from a
// snapshot returned by the Go library (window.goneuralDemo, installed by
// main.wasm); nothing here computes an activation, a loss or a gradient.
(() => {
  "use strict";

  const SLOW_MS = 700; // one stage per tick
  const FAST_MS = 20; // one whole training step per 20ms, about 50 a second
  const MAX_CATCH_UP = 25; // steps run in one tick when the timer fires late
  const SAMPLE_MS = 250; // how often "Model writes" refreshes in fast mode
  const MAX_POINTS = 400; // loss history is thinned beyond this

  const SVG_NS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);

  const ui = {
    word: $("word"),
    stages: [...$("stages").children],
    net: $("net"),
    caption: $("caption"),
    step: $("step"),
    play: $("play"),
    speed: $("speed"),
    speedValue: $("speed-value"),
    reset: $("reset"),
    sample: $("sample"),
    average: $("average"),
    best: $("best"),
    updates: $("updates"),
    chart: $("chart"),
    form: $("word-form"),
    input: $("word-input"),
    train: $("train"),
    error: $("word-error"),
    spec: $("spec"),
  };

  let demo = null; // the Go API
  let snap = null; // latest snapshot
  let view = null; // SVG elements of the network, rebuilt per word
  let playing = false;
  let fast = false;
  let timer = null;
  let history = []; // [{ updates, loss }]
  let stride = 1;
  let lastSampleAt = 0;
  let fastClock = 0; // time up to which fast mode's steps have been run

  // ---- helpers -----------------------------------------------------------

  function svg(name, attrs = {}, text) {
    const node = document.createElementNS(SVG_NS, name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // Calls into Go. Failures come back as {error}, never as exceptions.
  function call(name, ...args) {
    const result = demo[name](...args);
    if (result && typeof result === "object" && result.error) throw new Error(result.error);
    return result;
  }

  const num = (x, digits = 2) => x.toFixed(digits).replace("-", "−");

  function percent(p) {
    const rounded = Math.round(p * 100);
    if (rounded === 0 && p > 0) return "<1%";
    return rounded + "%";
  }

  // How a vocabulary entry is named in a sentence. Entry 0 is the marker.
  function letterName(id, end) {
    if (id === 0) return end ? "the end marker" : "the start marker";
    return `“${snap.vocab[id]}”`;
  }

  function randomSeed() {
    return crypto.getRandomValues(new Uint32Array(1))[0];
  }

  // ---- building the views for a word -------------------------------------

  function buildWord() {
    ui.word.replaceChildren();
    const chars = [snap.vocab[0], ...snap.word, snap.vocab[0]];
    chars.forEach((c, i) => {
      const box = document.createElement("span");
      box.textContent = c;
      if (i === 0 || i === chars.length - 1) box.className = "marker";
      ui.word.append(box);
    });
  }

  function buildNetwork() {
    const vocab = snap.vocab.length;
    const hidden = snap.weights.inputHidden.length;

    const top = 24;
    const rowHeight = 32;
    const rows = Math.max(vocab, hidden);
    const height = top + rows * rowHeight + 4;
    const yOf = (i, n) => top + (rows * rowHeight * (i + 0.5)) / n;

    const X = { input: 28, hidden: 150, output: 260, bar: 282, barWidth: 50, percent: 374 };
    const R = { letter: 12, hidden: 17 };

    ui.net.setAttribute("viewBox", `0 0 380 ${height}`);
    ui.net.replaceChildren();

    const wires = svg("g");
    const units = svg("g");
    ui.net.append(
      svg("text", { class: "heading", x: X.input - R.letter, y: 11 }, "LETTER IN"),
      svg("text", { class: "heading", x: X.hidden, y: 11, "text-anchor": "middle" }, "HIDDEN"),
      svg("text", { class: "heading", x: X.output - R.letter, y: 11 }, "NEXT LETTER"),
      wires,
      units,
    );

    view = { wires, inputHidden: [], hiddenOutput: [], inputs: [], hidden: [], outputs: [], barWidth: X.barWidth };

    for (let h = 0; h < hidden; h++) {
      view.inputHidden.push([]);
      for (let v = 0; v < vocab; v++) {
        const line = svg("line", {
          class: "wire",
          x1: X.input + R.letter, y1: yOf(v, vocab),
          x2: X.hidden - R.hidden, y2: yOf(h, hidden),
        });
        wires.append(line);
        view.inputHidden[h].push(line);
      }
    }
    for (let v = 0; v < vocab; v++) {
      view.hiddenOutput.push([]);
      for (let h = 0; h < hidden; h++) {
        const line = svg("line", {
          class: "wire",
          x1: X.hidden + R.hidden, y1: yOf(h, hidden),
          x2: X.output - R.letter, y2: yOf(v, vocab),
        });
        wires.append(line);
        view.hiddenOutput[v].push(line);
      }
    }

    const glyph = (v, x, y) =>
      svg("text", { class: v === 0 ? "glyph marker" : "glyph", x, y }, snap.vocab[v]);

    for (let v = 0; v < vocab; v++) {
      const y = yOf(v, vocab);

      const input = svg("circle", { class: "unit", cx: X.input, cy: y, r: R.letter });
      units.append(input, glyph(v, X.input, y));
      view.inputs.push(input);

      const output = svg("circle", { class: "unit", cx: X.output, cy: y, r: R.letter });
      const track = svg("rect", { class: "bar-track", x: X.bar, y: y - 5, width: X.barWidth, height: 10, rx: 2 });
      const bar = svg("rect", { class: "bar", x: X.bar, y: y - 5, width: 0, height: 10, rx: 2 });
      const label = svg("text", { class: "percent", x: X.percent, y });
      units.append(output, glyph(v, X.output, y), track, bar, label);
      view.outputs.push({ unit: output, bar, label });
    }

    for (let h = 0; h < hidden; h++) {
      const y = yOf(h, hidden);
      const unit = svg("circle", { class: "unit", cx: X.hidden, cy: y, r: R.hidden });
      const label = svg("text", { class: "activation", x: X.hidden, y });
      units.append(unit, label);
      view.hidden.push({ unit, label });
    }
  }

  // ---- drawing a snapshot ------------------------------------------------

  const weightWidth = (w) => 0.6 + Math.min(Math.abs(w), 4) * 1.3;
  const gradientWidth = (g) => 0.8 + Math.sqrt(Math.min(Math.abs(g), 1)) * 5.5;

  function drawWire(line, kind, width, opacity) {
    line.setAttribute("class", kind ? `wire ${kind}` : "wire");
    line.setAttribute("stroke-width", width.toFixed(2));
    if (opacity === undefined) line.removeAttribute("stroke-opacity");
    else line.setAttribute("stroke-opacity", opacity.toFixed(2));
    // Coloured wires are moved to the end of the group so they paint on top.
    if (kind) view.wires.append(line);
  }

  function drawNetwork() {
    const stage = snap.stage;
    const signal = stage === "forward" || stage === "loss";
    const blame = stage === "backward";
    const change = stage === "update" ? snap.lastChange : null;

    const { weights, gradients } = snap;

    weights.inputHidden.forEach((row, h) => {
      row.forEach((w, v) => {
        const line = view.inputHidden[h][v];
        const g = gradients.inputHidden[h][v];
        if (signal && v === snap.input) drawWire(line, "signal", weightWidth(w));
        else if (blame && g !== 0) drawWire(line, "blame", gradientWidth(g));
        else if (change && change.layer === "inputHidden" && change.from === v && change.to === h)
          drawWire(line, "changed", weightWidth(w));
        else drawWire(line, "", weightWidth(w));
      });
    });

    weights.hiddenOutput.forEach((row, v) => {
      row.forEach((w, h) => {
        const line = view.hiddenOutput[v][h];
        const g = gradients.hiddenOutput[v][h];
        // A hidden unit near zero sends little down its wires, however
        // heavy they are, so the signal colour fades with its activation.
        if (signal) drawWire(line, "signal", weightWidth(w), 0.25 + 0.75 * Math.abs(snap.hidden[h]));
        else if (blame && g !== 0) drawWire(line, "blame", gradientWidth(g));
        else if (change && change.layer === "hiddenOutput" && change.from === h && change.to === v)
          drawWire(line, "changed", weightWidth(w));
        else drawWire(line, "", weightWidth(w));
      });
    });

    view.inputs.forEach((unit, v) => {
      unit.setAttribute("class", v === snap.input ? "unit active" : "unit");
    });

    view.hidden.forEach(({ unit, label }, h) => {
      unit.setAttribute("class", signal ? "unit lit" : "unit");
      label.textContent = snap.hidden ? num(snap.hidden[h]) : "";
    });

    const targetKnown = stage === "loss" || stage === "backward" || stage === "update";
    view.outputs.forEach(({ unit, bar, label }, v) => {
      unit.setAttribute("class", targetKnown && v === snap.target ? "unit want" : "unit");
      const p = snap.probabilities ? snap.probabilities[v] : null;
      bar.setAttribute("width", p === null ? 0 : (p * view.barWidth).toFixed(2));
      bar.setAttribute("class", signal ? "bar live" : "bar");
      label.textContent = p === null ? "" : percent(p);
    });
  }

  function drawWord() {
    [...ui.word.children].forEach((box, i) => {
      box.classList.toggle("in", i === snap.pair);
      box.classList.toggle("want", i === snap.pair + 1);
    });
  }

  function drawStages() {
    for (const chip of ui.stages) {
      if (chip.dataset.stage === snap.stage) chip.setAttribute("aria-current", "step");
      else chip.removeAttribute("aria-current");
    }
  }

  function setCaption(stage, title, text) {
    ui.caption.dataset.stage = stage;
    ui.caption.replaceChildren();
    if (title) {
      const b = document.createElement("b");
      b.textContent = title + " ";
      ui.caption.append(b);
    }
    ui.caption.append(text);
  }

  function drawCaption() {
    if (playing && fast) {
      setCaption("update", "Running fast:", `${snap.updates} steps so far. Pause to look at one step up close.`);
      return;
    }

    switch (snap.stage) {
      case "ready":
        setCaption("ready", "Ready.", `The weights are random, so the model writes nonsense. Press Step to feed in ${letterName(snap.input, false)}.`);
        break;
      case "forward":
        setCaption("forward", "Forward:", `fed in ${letterName(snap.input, false)}. The bars are its guesses for the next letter.`);
        break;
      case "loss":
        setCaption("loss", "Loss:", `wanted ${letterName(snap.target, true)}, gave it ${percent(snap.probabilities[snap.target])}. Loss ${num(snap.loss)} (0 is perfect).`);
        break;
      case "backward":
        setCaption("backward", "Backward:", "coral lines carry the blame back. Thicker means a bigger fix.");
        break;
      case "update": {
        const c = snap.lastChange;
        const from = c.layer === "inputHidden" ? letterName(c.from, false) : `hidden ${c.from + 1}`;
        const to = c.layer === "inputHidden" ? `hidden ${c.to + 1}` : letterName(c.to, true);
        // Late in training even the biggest change is tiny, so show as many
        // digits as it takes for the two values to read differently.
        let digits = 2;
        while (digits < 5 && num(c.old, digits) === num(c.new, digits)) digits++;
        setCaption("update", "Update:", `biggest change: ${from} → ${to} went from ${num(c.old, digits)} to ${num(c.new, digits)}.`);
        break;
      }
    }
  }

  function drawSample(force) {
    const now = performance.now();
    if (!force && now - lastSampleAt < SAMPLE_MS) return;
    lastSampleAt = now;

    const text = call("sample");
    ui.sample.textContent = text || "(nothing)";
    ui.sample.classList.toggle("empty", !text);
  }

  function drawNumbers() {
    ui.average.textContent = num(snap.averageLoss, 3);
    ui.best.textContent = num(snap.bestLoss, 3);
    ui.updates.textContent = snap.updates;
  }

  function drawChart() {
    const box = { left: 6, right: 374, top: 14, bottom: 96 };
    const maxX = Math.max(history[history.length - 1].updates, 30);
    const maxY = Math.max(...history.map((p) => p.loss), snap.bestLoss, 0.1) * 1.05;
    const x = (u) => box.left + ((box.right - box.left) * u) / maxX;
    const y = (l) => box.bottom - ((box.bottom - box.top) * l) / maxY;

    const floorY = y(snap.bestLoss);
    const points = history.map((p) => `${x(p.updates).toFixed(1)},${y(p.loss).toFixed(1)}`).join(" ");

    ui.chart.replaceChildren(
      svg("line", { class: "base", x1: box.left, y1: box.bottom, x2: box.right, y2: box.bottom }),
      svg("line", { class: "floor", x1: box.left, y1: floorY, x2: box.right, y2: floorY }),
      svg("polyline", { class: "trace", points }),
      svg("text", { class: "axis", x: box.left, y: 9 }, `LOSS, 0 TO ${num(maxY, 1)}`),
      svg("line", { class: "floor", x1: box.right - 108, y1: 6, x2: box.right - 88, y2: 6 }),
      svg("text", { class: "axis", x: box.right, y: 9, "text-anchor": "end" }, "BEST POSSIBLE"),
      svg("text", { class: "axis", x: box.left, y: 108 }, "0"),
      svg("text", { class: "axis", x: box.right, y: 108, "text-anchor": "end" }, `${maxX} STEPS`),
    );
  }

  function record() {
    const last = history[history.length - 1];
    if (last && snap.updates - last.updates < stride) return;

    history.push({ updates: snap.updates, loss: snap.averageLoss });
    if (history.length > MAX_POINTS) {
      history = history.filter((_, i) => i % 2 === 0);
      stride *= 2;
    }
  }

  function draw() {
    drawWord();
    drawStages();
    drawNetwork();
    drawCaption();
    drawNumbers();

    if (snap.stage === "update" || snap.stage === "ready") {
      record();
      drawChart();
      drawSample(!(playing && fast));
    }
  }

  // ---- control -----------------------------------------------------------

  function advance() {
    try {
      if (playing && fast) {
        // Browsers deliver timers late (and rarely, in a background tab),
        // so fast mode runs however many steps the clock says are due
        // rather than one per tick.
        const now = performance.now();
        const due = Math.min(Math.floor((now - fastClock) / FAST_MS), MAX_CATCH_UP);
        if (due < 1) return;
        fastClock = due === MAX_CATCH_UP ? now : fastClock + due * FAST_MS;
        snap = call("stepPair", due);
      } else {
        snap = call("step");
      }
      draw();
    } catch (err) {
      fail(err);
    }
  }

  function schedule() {
    clearInterval(timer);
    timer = null;
    fastClock = performance.now() - FAST_MS;
    if (playing) timer = setInterval(advance, fast ? FAST_MS : SLOW_MS);
  }

  function setPlaying(on) {
    playing = on;
    ui.play.textContent = on ? "Pause" : "Play";
    document.body.classList.toggle("fast", on && fast);
    schedule();
    if (on) advance();
    else if (snap) {
      // Leaving fast mode: replace the running caption with the stage's own.
      drawCaption();
      drawSample(true);
    }
  }

  function setFast(on) {
    fast = on;
    ui.speedValue.textContent = on ? "Fast" : "Slow";
    ui.speed.setAttribute("aria-pressed", String(on));
    ui.speed.setAttribute("aria-label", on ? "Speed: fast. Switch to slow." : "Speed: slow. Switch to fast.");
    document.body.classList.toggle("fast", playing && on);
    schedule();
    if (snap) drawCaption();
  }

  // Starts over on a word with new random weights. Returns an error message
  // for the form to show, or null.
  function start(word, seed) {
    let next;
    try {
      next = call("reset", word, seed);
    } catch (err) {
      return err.message;
    }

    setPlaying(false);
    snap = next;
    history = [];
    stride = 1;
    buildWord();
    buildNetwork();
    draw();
    return null;
  }

  function showError(message) {
    ui.error.textContent = message || "";
    ui.error.hidden = !message;
    if (message) ui.input.setAttribute("aria-invalid", "true");
    else ui.input.removeAttribute("aria-invalid");
  }

  function fail(err) {
    setPlaying(false);
    for (const b of [ui.step, ui.play, ui.speed, ui.reset, ui.train]) b.disabled = true;
    setCaption("error", "Something went wrong.", String(err.message || err));
  }

  ui.step.addEventListener("click", () => {
    if (playing) setPlaying(false);
    advance();
  });
  ui.play.addEventListener("click", () => setPlaying(!playing));
  ui.speed.addEventListener("click", () => setFast(!fast));
  ui.reset.addEventListener("click", () => start(snap.word, randomSeed()));

  ui.form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!demo) return;
    const word = ui.input.value.trim().toLowerCase();
    const message = start(word, randomSeed());
    showError(message);
    if (message) ui.input.focus();
    else ui.input.value = word;
  });
  ui.input.addEventListener("input", () => showError(""));

  // ---- loading the library -----------------------------------------------

  async function load() {
    const go = new Go();
    const ready = new Promise((resolve) => (window.onGoneuralReady = resolve));

    let result;
    try {
      result = await WebAssembly.instantiateStreaming(fetch("main.wasm"), go.importObject);
    } catch (streamingError) {
      // Some static hosts serve .wasm with the wrong content type, which
      // only the streaming path objects to.
      const response = await fetch("main.wasm");
      if (!response.ok) throw new Error(`main.wasm: HTTP ${response.status}`);
      result = await WebAssembly.instantiate(await response.arrayBuffer(), go.importObject);
    }

    go.run(result.instance); // never resolves: the Go program stays alive
    await ready;
    demo = window.goneuralDemo;

    const config = demo.config;
    ui.spec.textContent =
      `One-hot letter in, ${config.hiddenUnits} tanh hidden units, softmax out. ` +
      `Cross-entropy loss, plain SGD at learning rate ${config.learningRate}, one letter pair per step.`;
    ui.input.maxLength = config.maxWordLength;

    // ?word= and ?seed= make a particular run linkable and reproducible.
    const query = new URLSearchParams(location.search);
    const seed = /^\d{1,9}$/.test(query.get("seed") || "") ? Number(query.get("seed")) : randomSeed();
    if (start((query.get("word") || "hello").toLowerCase(), seed) !== null) start("hello", seed);

    for (const b of [ui.step, ui.play, ui.speed, ui.reset, ui.train]) b.disabled = false;
    document.body.classList.remove("loading");
  }

  load().catch((err) => {
    const hint = location.protocol === "file:"
      ? "This page has to be served over http, not opened as a file. Run: go run ./demo/serve"
      : String(err.message || err);
    setCaption("error", "Couldn’t load the Go library.", hint);
  });
})();
