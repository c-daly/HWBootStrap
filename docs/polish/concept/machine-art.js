"use strict";
// Original SVG machine studies. Appearance is cosmetic and independent of unit stats.
window.HexMachines = (() => {
  const catalog = [
    {
      id: "relay",
      name: "Relay",
      detail: "Utility rover",
      cue: "Six wheels, a compact cab and a tool mount.",
    },
    {
      id: "atlas",
      name: "Atlas",
      detail: "Armored crawler",
      cue: "Wide tracks and a low, heavily plated hull.",
    },
    {
      id: "edge",
      name: "Edge",
      detail: "Assault vehicle",
      cue: "A forward turret with paired short barrels.",
    },
    {
      id: "bastion",
      name: "Bastion",
      detail: "Shield carrier",
      cue: "A planted chassis behind a broad armored shield.",
    },
    {
      id: "glide",
      name: "Glide",
      detail: "Recon buggy",
      cue: "Exposed wheels, a light cab and a whip antenna.",
    },
    {
      id: "crux",
      name: "Crux",
      detail: "Articulated walker",
      cue: "Four jointed legs with broad climbing feet.",
    },
    {
      id: "lance",
      name: "Lance",
      detail: "Long-range gun",
      cue: "A long barrel and visible recoil housing on a tracked chassis.",
    },
    {
      id: "halo",
      name: "Halo",
      detail: "Sensor rover",
      cue: "A radar dish, mast and equipment housing.",
    },
  ];
  let sequence = 0;
  const escape = (value) =>
    String(value).replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
  function render(
    id,
    { team = "friendly", label = "", decorative = false } = {},
  ) {
    const art = catalog.find((x) => x.id === id) || catalog[0],
      key = `machine-${++sequence}`;
    const color = team === "enemy" ? "#e6a17b" : "#9cdac7";
    const path = (d, fill = "hull", extra = "") =>
      `<path d="${d}" fill="${fill[0] === "#" ? fill : `url(#${key}-${fill})`}" ${extra}/>`;
    const line = (d, stroke = "#9fadb4", width = 1.3) =>
      `<path d="${d}" fill="none" stroke="${stroke}" stroke-width="${width}" stroke-linejoin="round" stroke-linecap="round"/>`;
    const lens = (x, y, r = 2) =>
      `<circle cx="${x}" cy="${y}" r="${r + 2}" fill="#182c32"/><circle cx="${x}" cy="${y}" r="${r}" fill="${color}"/>`;
    const wheel = (x, y) =>
      `<ellipse cx="${x}" cy="${y}" rx="10" ry="13" fill="#111b23" stroke="#52616b" stroke-width="2"/><ellipse cx="${x}" cy="${y}" rx="4.5" ry="6.5" fill="#65737d"/><path d="M${x - 7} ${y - 7}l4-4M${x + 5} ${y + 6}l-3 5" stroke="#a4adb0" stroke-width="1.5"/>`;
    const tread = (x, y) =>
      path(`M${x} ${y}l48-25 13 8v20l-48 26-13-8Z`, "#14212b") +
      path(`M${x + 2} ${y + 2}l11 7 44-23v13l-44 24-11-7Z`, "side") +
      Array.from({ length: 6 }, (_, i) =>
        line(`M${x + 14 + i * 7} ${y + 9 - i * 3.8}v12`, "#819098", 1.6),
      ).join("");
    const hull = () =>
      path("M45 117 101 85 153 107 153 123 98 153 45 134Z", "side") +
      path("M45 117 101 85 153 107 98 138Z", "hull") +
      line("M48 118 98 138 150 109") +
      line("M56 131 94 147", color, 2.4) +
      path("M68 114 100 96 125 107 94 126Z", "top") +
      line("M107 99 130 109M111 96 136 107", "#253844", 2);
    const turret = () =>
      path("M76 103 99 87 120 96 121 112 99 124 77 114Z", "side") +
      path("M76 103 99 87 120 96 99 110Z", "top") +
      lens(83, 111);
    const barrel = (dx = 0, dy = 0, long = true) =>
      `<g transform="translate(${dx} ${dy})">` +
      path(
        long
          ? "M105 96 159 58 166 61 167 70 112 108Z"
          : "M105 96 135 76 143 80 143 87 112 108Z",
        "hull",
      ) +
      path(
        long
          ? "M105 96 159 58 166 61 112 101Z"
          : "M105 96 135 76 143 80 112 101Z",
        "top",
      ) +
      path(
        long
          ? "M159 58 166 61 167 70 160 67Z"
          : "M135 76 143 80 143 87 136 84Z",
        "#13212a",
      ) +
      line(long ? "M127 82 131 86v8" : "M122 84 127 89v7", "#b5c1c6", 2) +
      "</g>";
    let chassis = "";
    if (art.id === "crux") {
      chassis =
        path("M65 106 47 123 40 151 51 156 59 131 82 115Z", "hull") +
        path("M128 96 146 111 162 131 153 140 134 118 116 112Z", "hull") +
        path("M88 127 80 151 80 171 94 171 97 148 111 131Z", "side") +
        path("M135 118 145 143 138 163 151 168 158 142 149 117Z", "side") +
        path("M33 150 47 146 58 152 46 162 33 158Z", "top") +
        path("M72 169 85 162 99 169 86 178 72 174Z", "top") +
        path("M133 164 147 159 160 166 147 175 134 170Z", "top") +
        hull() +
        turret() +
        line("M48 126 52 123M85 151h8M149 139l5 2", color, 3);
    } else {
      const wheeled = ["relay", "glide", "halo"].includes(art.id);
      chassis = wheeled
        ? wheel(122, 99) + wheel(146, 112) + wheel(98, 111)
        : tread(67, 111);
      chassis += hull();
      if (wheeled) chassis += wheel(49, 137) + wheel(75, 149) + wheel(111, 145);
      else chassis += tread(47, 135);
      if (art.id === "lance")
        chassis += turret() + barrel() + line("M98 101 110 108", color, 2.4);
      if (art.id === "edge")
        chassis += turret() + barrel(-9, -2, false) + barrel(5, 6, false);
      if (art.id === "atlas")
        chassis +=
          path("M57 109 90 76 130 81 141 108 98 129Z", "hull") +
          path("M57 109 90 76 130 81 99 103Z", "top") +
          path("M99 103 130 81 141 108 98 129Z", "side") +
          line("M69 108 96 118", color, 3) +
          line("M92 82 101 101M110 89 127 93", "#b3bdc2") +
          lens(128, 106, 3);
      if (art.id === "bastion")
        chassis +=
          turret() +
          path("M51 73 91 49 116 61 120 119 82 142 53 126Z", "side") +
          path("M51 73 91 49 111 60 75 84 79 132 53 120Z", "hull") +
          path("M51 73 91 49 111 60 75 84Z", "top") +
          line("M59 82 62 112 72 119", color, 2.5) +
          line("M83 92 86 126M97 82 100 117", "#0f222d", 3) +
          lens(66, 84, 2);
      if (art.id === "glide")
        chassis +=
          path("M72 107 87 79 111 74 126 92 113 116 92 126Z", "hull") +
          path("M87 79 111 74 126 92 102 103Z", "top") +
          path("M79 103 88 86 98 108 94 117Z", "#142c38") +
          path("M104 106 126 97 119 110 104 117Z", "#192f3b") +
          line("M79 122 91 127", color, 2) +
          line("M118 80 122 45", "#a9b7bd", 1.5) +
          lens(122, 45, 1.5) +
          (art.id === "relay"
            ? path("M123 96 139 87 146 91 132 106Z", "top")
            : "");
      if (art.id === "relay")
        chassis +=
          path("M60 104 63 79 92 62 114 72 114 100 86 119Z", "hull") +
          path("M63 79 92 62 114 72 86 89Z", "top") +
          path("M66 83 80 91 80 105 65 98Z", "#1b303b") +
          path("M91 88 110 77 110 90 91 102Z", "#18313c") +
          line("M66 109 82 117", color, 2) +
          path("M119 107 127 96 149 69 156 73 134 108Z", "hull") +
          line("M131 92 149 70", "#bec9cd", 2) +
          path(
            "M146 68 153 62 166 71 166 84 162 84 162 74 153 70 150 75Z",
            "top",
          ) +
          path("M108 114 122 106 134 111 120 120Z", "top") +
          lens(70, 105, 2);
      if (art.id === "halo")
        chassis +=
          path("M79 114 80 91 111 73 129 84 129 103 97 122Z", "hull") +
          path("M80 91 111 73 129 84 97 103Z", "top") +
          line("M104 91 104 62", "#a1b0b8", 5) +
          path(
            "M76 47Q74 87 103 87Q125 81 134 53L123 42Q107 65 76 47Z",
            "side",
          ) +
          path("M76 47Q95 78 123 42Q98 34 76 47Z", "top") +
          line("M81 48 102 63 119 43M102 63 111 38", "#364e5a", 1.7) +
          lens(111, 38, 2) +
          line("M137 99 140 65", "#b3bdc2") +
          lens(140, 65, 1.5);
    }
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" class="machine-art" ${decorative ? 'aria-hidden="true"' : `role="img" aria-label="${escape(label || `${art.name}: ${art.detail}`)}"`} data-form="${art.id}"><defs>
      <linearGradient id="${key}-hull" x1="0" y1="0" x2=".75" y2="1"><stop stop-color="#899aa4"/><stop offset=".38" stop-color="#536875"/><stop offset="1" stop-color="#293b48"/></linearGradient>
      <linearGradient id="${key}-top" x2="1" y2="1"><stop stop-color="#bdc7cb"/><stop offset="1" stop-color="#647c8a"/></linearGradient>
      <linearGradient id="${key}-side" x2="1" y2=".8"><stop stop-color="#465b69"/><stop offset="1" stop-color="#172936"/></linearGradient>
      <radialGradient id="${key}-shadow"><stop stop-color="#02090e" stop-opacity=".8"/><stop offset="1" stop-color="#02090e" stop-opacity="0"/></radialGradient>
      </defs><ellipse cx="106" cy="163" rx="82" ry="26" fill="url(#${key}-shadow)"/>${chassis}
      <path d="M58 165 97 179 145 156" fill="none" stroke="${color}" stroke-width="2" ${team === "enemy" ? 'stroke-dasharray="6 5"' : ""}/></svg>`;
  }
  return { catalog, render };
})();
