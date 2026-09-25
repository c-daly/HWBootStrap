"use strict";
// Isolated, deterministic UI fixture. This is not the HexWars engine or network client.
(() => {
  const $ = (id) => document.getElementById(id),
    art = window.HexMachines;
  const icon = (id) => `<svg aria-hidden="true"><use href="#i-${id}"/></svg>`;
  const cells = [];
  for (let r = 0; r < 6; r++)
    for (let q = 0; q < 7; q++) {
      if (
        (r === 0 && (q < 2 || q > 4)) ||
        (r === 5 && (q === 0 || q === 6)) ||
        (r === 1 && q === 6) ||
        (r === 4 && q === 0)
      )
        continue;
      let type =
        (q === 3 && r < 3) || (q === 4 && r === 4)
          ? "water"
          : (q === 4 && r === 2) || (q === 5 && r === 1) || (q === 1 && r === 2)
            ? "forest"
            : (q + r) % 4 === 0
              ? "stone"
              : "plain";
      let height = type === "water" ? 0 : type === "stone" ? 2 : 1;
      if (q === 2 && r === 3) {
        height = 2;
        type = "stone";
      }
      cells.push({
        id: `${q},${r}`,
        q,
        r,
        type,
        height,
        label: `${String.fromCharCode(65 + q)}${r + 1}`,
      });
    }
  const byCell = new Map(cells.map((c) => [c.id, c]));
  const point = (c) => ({
    x: 112 + c.q * 92 + (c.r % 2) * 46,
    y: 181 + c.r * 49 - c.height * 13,
  });
  const top = "0,-31 45,-15.5 45,15.5 0,31 -45,15.5 -45,-15.5";
  const baseUnits = [
    {
      id: "longshot",
      name: "Longshot",
      role: "Sniper",
      form: "lance",
      team: "friendly",
      cell: "2,3",
      hp: 4,
      maxHp: 4,
      damage: 5,
      defense: 0,
      move: 2,
      climb: 1,
      range: 5,
      rangeArc: 1,
      vision: 3,
    },
    {
      id: "anchor",
      name: "Anchor",
      role: "Bulwark",
      form: "bastion",
      team: "friendly",
      cell: "1,4",
      hp: 6,
      maxHp: 6,
      damage: 2,
      defense: 7,
      move: 1,
      climb: 1,
      range: 2,
      rangeArc: 0,
      vision: 2,
    },
    {
      id: "scout",
      name: "Scout",
      role: "Runner",
      form: "glide",
      team: "friendly",
      cell: "3,5",
      hp: 3,
      maxHp: 3,
      damage: 1,
      defense: 0,
      move: 5,
      climb: 1,
      range: 2,
      rangeArc: 0,
      vision: 3,
    },
    {
      id: "bulwark",
      name: "Bulwark",
      role: "Brute",
      form: "atlas",
      team: "enemy",
      cell: "4,2",
      hp: 7,
      maxHp: 7,
      damage: 2,
      defense: 2,
      move: 1,
      climb: 1,
      range: 2,
      rangeArc: 0,
      vision: 2,
    },
    {
      id: "sentry",
      name: "Sentry",
      role: "Sniper",
      form: "edge",
      team: "enemy",
      cell: "5,1",
      hp: 4,
      maxHp: 4,
      damage: 3,
      defense: 1,
      move: 2,
      climb: 1,
      range: 4,
      rangeArc: 0,
      vision: 2,
    },
  ];
  let units,
    selected,
    mode,
    target,
    destination,
    round,
    draftForm,
    hoverCell = null;
  const unit = () => units.find((u) => u.id === selected);
  const enemy = () => units.find((u) => u.id === target && u.hp > 0);
  const alive = (team) => units.filter((u) => u.team === team && u.hp > 0);
  const canAct = (u) => u.remaining > 0 || !u.fired;
  function announce(text) {
    $("announcement").textContent = text;
  }
  function reset() {
    units = baseUnits.map((u) => ({ ...u, remaining: u.move, fired: false }));
    selected = "longshot";
    mode = "attack";
    target = "bulwark";
    destination = null;
    round = 4;
    hoverCell = null;
    render();
    announce("Sample encounter reset. Longshot selected.");
  }
  const axial = (c) => ({ q: c.q - Math.floor(c.r / 2), r: c.r });
  function distance(a, b) {
    const x = axial(a),
      y = axial(b);
    return (
      (Math.abs(x.q - y.q) +
        Math.abs(x.r - y.r) +
        Math.abs(x.q + x.r - y.q - y.r)) /
      2
    );
  }
  function paths(u) {
    const occupied = new Set(
      units.filter((x) => x.hp > 0 && x.id !== u.id).map((x) => x.cell),
    );
    const found = new Map([[u.cell, { cost: 0, path: [u.cell] }]]),
      queue = [u.cell];
    while (queue.length) {
      queue.sort((a, b) => found.get(a).cost - found.get(b).cost);
      const id = queue.shift(),
        current = byCell.get(id),
        entry = found.get(id);
      for (const next of cells) {
        if (
          distance(current, next) !== 1 ||
          next.type === "water" ||
          occupied.has(next.id) ||
          next.height - current.height > u.climb
        )
          continue;
        const cost =
          entry.cost +
          (next.type === "forest" ? 2 : 1) +
          Math.max(0, next.height - current.height);
        if (
          cost > u.remaining ||
          (found.has(next.id) && found.get(next.id).cost <= cost)
        )
          continue;
        found.set(next.id, { cost, path: [...entry.path, next.id] });
        queue.push(next.id);
      }
    }
    found.delete(u.cell);
    return found;
  }
  function targets(u) {
    return u.fired
      ? []
      : alive("enemy").filter(
          (e) =>
            distance(byCell.get(u.cell), byCell.get(e.cell)) <=
            u.range +
              Math.max(
                0,
                byCell.get(u.cell).height - byCell.get(e.cell).height,
              ),
        );
  }
  function forecast(u, e) {
    const height = Math.max(
        0,
        byCell.get(u.cell).height - byCell.get(e.cell).height,
      ),
      cover = byCell.get(e.cell).type === "forest" ? 1 : 0;
    const damage =
      u.damage <= 0 ? 0 : Math.max(0, u.damage + height - e.defense - cover);
    return { height, cover, damage, left: Math.max(0, e.hp - damage) };
  }
  function terrain() {
    const reach = mode === "move" ? paths(unit()) : new Map();
    $("terrain").innerHTML = [...cells]
      .sort((a, b) => a.r - b.r || a.q - b.q)
      .map((c) => {
        const p = point(c),
          d = 15 + c.height * 13,
          reachable = reach.has(c.id),
          picked = c.id === destination;
        let texture = "";
        if (c.type === "water")
          texture =
            '<path d="M-26 0q10-5 20 0t20 0M-15 10q10-5 20 0" fill="none" stroke="#77a9b4" stroke-opacity=".4"/>';
        if (c.type === "stone")
          texture =
            '<path d="M-31-6-19-13-1-9 3 1 20 4M3 1l-7 12M20 4l11-5" fill="none" stroke="#b1b7ae" stroke-opacity=".22"/>';
        if (c.type === "forest")
          texture = [
            [-21, -4],
            [19, 3],
            [3, -11],
          ]
            .map(
              ([x, y]) =>
                `<g transform="translate(${x} ${y})"><path d="M-7 0 0-20 7 0 0 5Z" fill="#233f3a"/><path d="M0-20 7 0 0 5Z" fill="#496659"/><path d="M0 4v5" stroke="#849183"/></g>`,
            )
            .join("");
        const steps = Array.from(
          { length: c.height },
          (_, i) =>
            `<path d="M-45 ${21 + i * 11} 0 ${36.5 + i * 11} 45 ${21 + i * 11}" fill="none" stroke="#83949a" stroke-opacity=".15"/>`,
        ).join("");
        const label = `${c.label}, ${c.type === "plain" ? "open ground" : c.type}, elevation ${c.height}${reachable ? `, reachable for ${reach.get(c.id).cost} movement` : ""}`;
        return `<g class="tile ${reachable ? "reachable" : ""}" transform="translate(${p.x} ${p.y})" data-cell="${c.id}" role="button" tabindex="${reachable ? "0" : "-1"}" aria-label="${label}"><path d="M-45-15.5 0 0 45-15.5V${15.5 + d}L0 ${31 + d}-45 ${15.5 + d}Z" fill="#233540"/><path d="M0 31 45 15.5V${15.5 + d}L0 ${31 + d}Z" fill="#192d38"/>${steps}<polygon class="top-face" points="${top}" fill="url(#${c.type}-top)"/><g class="tile-decoration">${texture}</g>${reachable ? `<polygon class="reachable-shade" points="${top}"/><text class="tile-coord" x="0" y="12" text-anchor="middle">${reach.get(c.id).cost}</text>` : ""}${picked ? `<polygon class="destination-ring" points="${top}" transform="scale(.86)"/>` : ""}</g>`;
      })
      .join("");
  }
  function machines() {
    const legal = new Set(
      mode === "attack" ? targets(unit()).map((u) => u.id) : [],
    );
    $("machines").innerHTML = units
      .filter((u) => u.hp > 0)
      .sort((a, b) => point(byCell.get(a.cell)).y - point(byCell.get(b.cell)).y)
      .map((u) => {
        const p = point(byCell.get(u.cell)),
          isSelected = u.id === selected,
          marked = u.id === target && mode === "attack",
          color = u.team === "friendly" ? "#a4dfca" : "#e7aa87";
        const body = art
          .render(u.form, { team: u.team, decorative: true })
          .replace("<svg ", '<svg x="-56" y="-89" width="112" height="112" ');
        return `<g class="unit-on-board" data-unit="${u.id}" transform="translate(${p.x} ${p.y})" role="button" tabindex="0" aria-label="${u.name}, ${u.team === "friendly" ? "your unit" : "opponent"}, ${u.hp} of ${u.maxHp} health${isSelected ? ", selected" : ""}${legal.has(u.id) ? ", attack available" : ""}"><ellipse class="unit-shadow" cx="0" cy="11" rx="35" ry="17" fill="#000" fill-opacity=".25"/><ellipse class="focus-ring" cx="0" cy="0" rx="45" ry="30"/>${body}<rect x="-18" y="-66" width="36" height="3" fill="#10222b"/><rect x="-18" y="-66" width="${(36 * u.hp) / u.maxHp}" height="3" fill="${color}"/>${isSelected ? '<path class="selection-bracket" d="M-41-4v14l14 7M41-4v14l-14 7M-22-19l-13 5M22-19l13 5"/>' : ""}${legal.has(u.id) ? `<path class="target-bracket" d="M-36-19v-9h12M36-19v-9H24M-36 8v9h12M36 8v9H24" opacity="${marked ? 1 : 0.5}"/>` : ""}${isSelected || marked ? `<text x="0" y="40" text-anchor="middle" class="unit-board-name">${u.name}</text>` : ""}</g>`;
      })
      .join("");
  }
  function route() {
    let html = "";
    const u = unit(),
      start = point(byCell.get(u.cell));
    if (mode === "move") {
      const dest = destination || hoverCell,
        entry = dest ? paths(u).get(dest) : null;
      if (entry) {
        const points = entry.path.map((id) => point(byCell.get(id)));
        html =
          `<path class="route-line" d="${points.map((p, i) => `${i ? "L" : "M"}${p.x} ${p.y}`).join(" ")}"/>` +
          points
            .slice(1)
            .map(
              (p) =>
                `<circle class="route-point" cx="${p.x}" cy="${p.y}" r="3.5"/>`,
            )
            .join("");
      }
    } else if (enemy() && targets(u).some((e) => e.id === target)) {
      const end = point(byCell.get(enemy().cell));
      html = `<path class="shot-line" d="M${start.x + 15} ${start.y - 38}Q${(start.x + end.x) / 2} ${Math.min(start.y, end.y) - 92} ${end.x} ${end.y - 35}"/><circle cx="${end.x}" cy="${end.y - 35}" r="4" fill="none" stroke="#edac8a"/>`;
    }
    $("routes").innerHTML = html;
  }
  function select(id) {
    const u = units.find((x) => x.id === id && x.hp > 0);
    if (!u) return;
    if (u.team === "enemy") {
      if (mode === "attack" && targets(unit()).some((e) => e.id === id)) {
        target = id;
        destination = null;
        render();
        announce(
          `${u.name} targeted. ${forecast(unit(), u).damage} damage previewed.`,
        );
      } else announce("That opponent is outside this unit’s available attack.");
      return;
    }
    selected = id;
    destination = null;
    hoverCell = null;
    target = null;
    if (u.fired && u.remaining > 0) mode = "move";
    if (mode === "attack") target = targets(u)[0]?.id || null;
    render();
    announce(
      `${u.name} selected. ${u.remaining} movement left. Attack ${u.fired ? "used" : "ready"}.`,
    );
  }
  function setMode(next) {
    const u = unit();
    if ((next === "move" && u.remaining <= 0) || (next === "attack" && u.fired))
      return;
    mode = next;
    destination = null;
    hoverCell = null;
    target = next === "attack" ? targets(u)[0]?.id || null : null;
    render();
    announce(
      next === "move"
        ? "Choose a highlighted hex or use the destination selector."
        : "Choose an outlined target to preview damage.",
    );
  }
  function squad() {
    $("squad-list").innerHTML = alive("friendly")
      .map(
        (u) =>
          `<button class="squad-card" data-select="${u.id}" aria-pressed="${u.id === selected}" aria-label="Select ${u.name}, ${u.role}, ${u.remaining} movement left, attack ${u.fired ? "used" : "ready"}"><span class="squad-art">${art.render(u.form, { decorative: true })}</span><span class="squad-copy"><strong>${u.name}</strong><small>${u.role}</small><span class="action-pips"><span class="${u.remaining ? "" : "spent"}">${icon("move")}</span><span class="${u.fired ? "spent" : ""}">${icon("attack")}</span></span></span><span class="squad-hp">${u.hp}/${u.maxHp}</span></button>`,
      )
      .join("");
    const count = alive("friendly").filter(canAct).length;
    $("ready-count").textContent = `${count} can act`;
    $("remaining-summary").textContent =
      `${count} ${count === 1 ? "unit can" : "units can"} still act`;
    $("friendly-count").textContent = alive("friendly").length;
    $("enemy-count").textContent = alive("enemy").length;
  }
  function inspector() {
    const u = unit(),
      form = art.catalog.find((a) => a.id === u.form),
      source = byCell.get(u.cell);
    $("unit-role").textContent =
      `${u.role.toUpperCase()} / 0${baseUnits.findIndex((x) => x.id === u.id) + 1}`;
    $("unit-name").textContent = u.name;
    $("unit-portrait").innerHTML = art.render(u.form);
    $("form-caption").textContent =
      `${form.name} · ${form.detail.toLowerCase()}`;
    $("unit-health").textContent = `${u.hp} / ${u.maxHp}`;
    $("unit-health-bar").innerHTML = Array.from(
      { length: u.maxHp },
      (_, i) => `<span class="${i >= u.hp ? "missing" : ""}"></span>`,
    ).join("");
    $("unit-stats").innerHTML = [
      ["attack", "Damage", u.damage],
      ["shield", "Defense", u.defense],
      ["attack", "Range", u.range],
      ["eye", "Vision", u.vision],
    ]
      .map(
        ([i, label, value]) =>
          `<div><dt>${icon(i)}${label}</dt><dd>${value}</dd></div>`,
      )
      .join("");
    $("availability").textContent = u.fired
      ? u.remaining
        ? "Move ready · attack used"
        : "Actions used"
      : u.remaining
        ? "Move + attack ready"
        : "Attack ready · move used";
    $("move-remaining").textContent = u.remaining
      ? `${u.remaining} ${u.remaining === 1 ? "step" : "steps"} left`
      : "Used";
    $("attack-remaining").textContent = u.fired ? "Used" : "Ready";
    $("move-mode").disabled = u.remaining <= 0;
    $("attack-mode").disabled = u.fired;
    $("move-mode").setAttribute("aria-pressed", mode === "move");
    $("attack-mode").setAttribute("aria-pressed", mode === "attack");
    $("commit").classList.toggle("move-commit", mode === "move");
    $("commit").disabled = true;
    $("commit-note").textContent = "Preview only until you confirm.";
    $("context-icon").innerHTML = icon(mode === "move" ? "move" : "attack");
    $("cell-info").textContent =
      `${source.label} / ${source.type === "stone" ? "ROCK" : source.type === "plain" ? "OPEN GROUND" : source.type.toUpperCase()} / ELEVATION ${source.height}`;
    if (mode === "attack") {
      const list = targets(u),
        e = list.find((x) => x.id === target);
      if (e) {
        const f = forecast(u, e);
        $("decision").innerHTML =
          `<p class="eyebrow">ATTACK PREVIEW</p><div class="target-title"><h3>${e.name}</h3><span>${byCell.get(e.cell).label} / OPPONENT</span></div><div class="forecast"><strong>${f.damage}</strong><div>damage<small class="${f.left === 0 ? "kill-label" : ""}">${f.left === 0 ? "Unit destroyed" : `Health ${e.hp} → ${f.left} / ${e.maxHp}`}</small></div></div><div class="damage-track" aria-label="${f.left} of ${e.maxHp} health after this attack">${Array.from({ length: e.maxHp }, (_, i) => `<span class="${i >= e.hp ? "empty" : i >= f.left ? "lost" : ""}"></span>`).join("")}</div><dl class="math"><div><dt>Weapon damage</dt><dd>${u.damage}</dd></div><div class="bonus"><dt>High ground · ${f.height} ${f.height === 1 ? "level" : "levels"}</dt><dd>+${f.height}</dd></div><div><dt>Armor ${e.defense} + cover ${f.cover}</dt><dd>−${e.defense + f.cover}</dd></div></dl>${list.length > 1 ? `<div class="target-list" aria-label="Available targets">${list.map((x) => `<button data-target="${x.id}" aria-pressed="${x.id === target}">${x.name}</button>`).join("")}</div>` : ""}`;
        $("commit").innerHTML =
          `Fire · ${f.damage} damage <span aria-hidden="true">↗</span>`;
        $("commit").disabled = false;
        $("context-hint").textContent =
          `${e.name} will ${f.left === 0 ? "be destroyed" : `have ${f.left} health left`}. Confirm the shot when you’re ready.`;
      } else {
        const reason = u.fired
          ? "Attack used this turn."
          : !alive("enemy").length
            ? "Field clear."
            : "No target in range.";
        $("decision").innerHTML =
          `<p class="eyebrow">ATTACK</p><p class="empty-preview">${reason}<small>${u.fired && u.remaining ? "You can still move this unit." : !alive("enemy").length ? "Reset the study to try another approach." : "Select Move to find a better position, or choose another unit."}</small></p>`;
        $("commit").textContent = u.fired ? "Attack used" : "No available shot";
        $("context-hint").textContent = reason;
      }
    } else {
      const reachable = paths(u),
        entry = reachable.get(destination),
        cell = byCell.get(destination);
      const picker = `<label class="sr-only" for="move-destination">Move destination</label><select id="move-destination" class="move-destination"><option value="">Choose a highlighted hex…</option>${[
        ...reachable,
      ]
        .sort((a, b) => a[1].cost - b[1].cost)
        .map(
          ([id, p]) =>
            `<option value="${id}" ${id === destination ? "selected" : ""}>${byCell.get(id).label} · ${p.cost} movement · elevation ${byCell.get(id).height}</option>`,
        )
        .join("")}</select>`;
      if (entry) {
        const uphill = Math.max(0, cell.height - source.height);
        $("decision").innerHTML =
          `<p class="eyebrow">MOVE PREVIEW</p><div class="target-title"><h3>${source.label} → ${cell.label}</h3><span>${cell.type.toUpperCase()}</span></div><div class="forecast move-forecast"><strong>${entry.cost}</strong><div>movement cost<small>${u.remaining - entry.cost} of ${u.move} remaining after move</small></div></div><dl class="math"><div><dt>Destination elevation</dt><dd>${cell.height}${uphill ? ` / +${uphill}` : ""}</dd></div><div><dt>Attack after moving</dt><dd>${u.fired ? "Already used" : "Still ready"}</dd></div></dl>${picker}`;
        $("commit").innerHTML =
          `Move to ${cell.label} <span aria-hidden="true">→</span>`;
        $("commit").disabled = false;
        $("context-hint").textContent =
          `Follow the marked route to ${cell.label}. Spend ${entry.cost} movement; ${u.fired ? "your attack is already used" : "keep your attack"}.`;
      } else {
        $("decision").innerHTML =
          `<p class="eyebrow">REPOSITION</p><p class="empty-preview">${reachable.size ? "Find a better angle." : "No movement available."}<small>${reachable.size ? "Choose a mint-outlined cell. The number shows its movement cost." : "Choose another unit or end your preview turn."}</small></p>${reachable.size ? picker : ""}`;
        $("commit").textContent = "Choose a destination";
        $("context-hint").textContent = reachable.size
          ? "Outlined cells are reachable. Choose one to preview the route."
          : "This unit cannot move farther this turn.";
      }
    }
    $("clear-preview").disabled = !destination && !target;
  }
  function render() {
    terrain();
    machines();
    route();
    squad();
    inspector();
    $("round").textContent = String(round).padStart(2, "0");
  }
  function chooseCell(id) {
    if (mode !== "move" || !paths(unit()).has(id)) {
      announce("Choose one of the highlighted reachable cells.");
      return;
    }
    destination = id;
    hoverCell = null;
    render();
    announce(
      `Move to ${byCell.get(id).label} previewed. Press the Move to ${byCell.get(id).label} button to confirm.`,
    );
  }
  $("board").addEventListener("click", (event) => {
    const u = event.target.closest("[data-unit]");
    if (u) {
      select(u.dataset.unit);
      return;
    }
    const c = event.target.closest("[data-cell]");
    if (c) chooseCell(c.dataset.cell);
  });
  $("board").addEventListener("keydown", (event) => {
    if (
      (event.key === "Enter" || event.key === " ") &&
      event.target.matches("[role=button]")
    ) {
      event.preventDefault();
      const key = event.target.dataset.unit || event.target.dataset.cell,
        attr = event.target.dataset.unit ? "data-unit" : "data-cell";
      event.target.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      $("board").querySelector(`[${attr}="${key}"]`)?.focus();
    }
  });
  $("board").addEventListener("pointerover", (event) => {
    const c = event.target.closest("[data-cell]");
    if (c && mode === "move" && !destination && hoverCell !== c.dataset.cell) {
      hoverCell = c.dataset.cell;
      route();
    }
  });
  $("board").addEventListener("pointerleave", () => {
    hoverCell = null;
    route();
  });
  $("squad-list").addEventListener("click", (event) => {
    const b = event.target.closest("[data-select]");
    if (b) {
      const id = b.dataset.select;
      select(id);
      $("squad-list").querySelector(`[data-select="${id}"]`).focus();
    }
  });
  $("decision").addEventListener("click", (event) => {
    const b = event.target.closest("[data-target]");
    if (b) {
      const id = b.dataset.target;
      select(id);
      $("decision").querySelector(`[data-target="${id}"]`)?.focus();
    }
  });
  $("decision").addEventListener("change", (event) => {
    if (event.target.id === "move-destination") {
      const id = event.target.value;
      if (id) chooseCell(id);
      else {
        destination = null;
        render();
      }
      $("move-destination")?.focus();
    }
  });
  $("move-mode").addEventListener("click", () => setMode("move"));
  $("attack-mode").addEventListener("click", () => setMode("attack"));
  function clear() {
    destination = null;
    target = null;
    hoverCell = null;
    render();
    announce("Action preview cleared.");
  }
  $("clear-preview").addEventListener("click", clear);
  $("commit").addEventListener("click", () => {
    const u = unit();
    if (mode === "move") {
      const p = paths(u).get(destination);
      if (!p) return;
      u.cell = destination;
      u.remaining -= p.cost;
      const label = byCell.get(destination).label;
      destination = null;
      hoverCell = null;
      if (!u.fired) {
        mode = "attack";
        target = targets(u)[0]?.id || null;
      }
      render();
      announce(
        `${u.name} moved to ${label}. ${u.remaining} movement left. Attack ${u.fired ? "used" : "ready"}.`,
      );
    } else {
      const e = targets(u).find((x) => x.id === target);
      if (!e) return;
      const f = forecast(u, e);
      e.hp = f.left;
      u.fired = true;
      target = null;
      render();
      $("board").classList.remove("shot-effect");
      requestAnimationFrame(() => $("board").classList.add("shot-effect"));
      setTimeout(() => $("board").classList.remove("shot-effect"), 350);
      announce(
        `${u.name} dealt ${f.damage} damage to ${e.name}. ${f.left ? `${f.left} health remaining.` : "Target destroyed."}`,
      );
    }
  });
  $("reset").addEventListener("click", reset);
  $("end-turn").addEventListener("click", () => {
    const n = alive("friendly").filter(canAct).length;
    $("turn-warning").textContent =
      `${n} ${n === 1 ? "unit still has" : "units still have"} actions available.`;
    $("turn-dialog").showModal();
  });
  $("keep-playing").addEventListener("click", () => $("turn-dialog").close());
  $("confirm-end").addEventListener("click", () => {
    round++;
    alive("friendly").forEach((u) => {
      u.remaining = u.move;
      u.fired = false;
    });
    destination = null;
    mode = "attack";
    target = targets(unit())[0]?.id || null;
    $("turn-dialog").close();
    render();
    announce(
      `Round ${round}. Your actions are refreshed. No opponent was simulated in this study.`,
    );
  });
  $("help").addEventListener("click", () => $("guide-dialog").showModal());
  let reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  function motion() {
    document.body.classList.toggle("reduced-motion", reduceMotion);
    $("reduce-motion").setAttribute("aria-pressed", reduceMotion);
    $("reduce-motion").textContent = reduceMotion
      ? "Motion reduced"
      : "Reduce motion";
  }
  $("reduce-motion").addEventListener("click", () => {
    reduceMotion = !reduceMotion;
    motion();
  });
  motion();
  function picker() {
    $("machine-picker").innerHTML = art.catalog
      .map(
        (a) =>
          `<button class="machine-choice" data-form="${a.id}" aria-pressed="${draftForm === a.id}">${art.render(a.id, { decorative: true })}<strong>${a.name}</strong><small>${a.detail}</small></button>`,
      )
      .join("");
    $("art-cue").textContent = art.catalog.find((a) => a.id === draftForm).cue;
  }
  $("open-workshop").addEventListener("click", () => {
    draftForm = unit().form;
    $("editing-unit").textContent = unit().name;
    picker();
    $("workshop-dialog").showModal();
  });
  $("machine-picker").addEventListener("click", (event) => {
    const b = event.target.closest("button[data-form]");
    if (b) {
      draftForm = b.dataset.form;
      picker();
      $("machine-picker").querySelector(`[data-form="${draftForm}"]`).focus();
    }
  });
  $("apply-art").addEventListener("click", () => {
    unit().form = draftForm;
    $("workshop-dialog").close();
    render();
    announce(
      `Appearance set to ${art.catalog.find((a) => a.id === draftForm).name}. Stats and abilities unchanged.`,
    );
  });
  document.addEventListener("keydown", (event) => {
    if (
      event.repeat ||
      event.altKey ||
      event.ctrlKey ||
      event.metaKey ||
      document.querySelector("dialog[open]") ||
      event.target.matches("input,select,textarea")
    )
      return;
    if (event.key.toLowerCase() === "m") {
      event.preventDefault();
      setMode("move");
      $("move-mode").focus();
    }
    if (event.key.toLowerCase() === "a") {
      event.preventDefault();
      setMode("attack");
      $("attack-mode").focus();
    }
    if (event.key === "Escape") clear();
  });
  reset();
})();
