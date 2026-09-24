'use strict';
// Original vector form studies. IDs are stable presentation metadata, never gameplay classes.
window.HexUnitArt = (() => {
  const catalog = [
    {id:'relay-01',name:'Relay',role:'Generalist',cue:'A balanced, faceted core.',shape:'relay'},
    {id:'atlas-01',name:'Atlas',role:'Brute',cue:'Broad shoulders. A low center of gravity.',shape:'atlas'},
    {id:'edge-01',name:'Edge',role:'Striker',cue:'A single forward-cutting plane.',shape:'edge'},
    {id:'bastion-01',name:'Bastion',role:'Bulwark',cue:'Two sheltering plates. One narrow seam.',shape:'bastion'},
    {id:'glide-01',name:'Glide',role:'Runner',cue:'A low hull, split into sweeping runners.',shape:'glide'},
    {id:'crux-01',name:'Crux',role:'Climber',cue:'A tall arch on three planted feet.',shape:'crux'},
    {id:'lance-01',name:'Lance',role:'Sniper',cue:'A quiet body with one unmistakable reach.',shape:'lance'},
    {id:'halo-01',name:'Halo',role:'Spotter',cue:'An open ring, lifted clear of its base.',shape:'halo'}
  ];
  const byId = new Map(catalog.map(art => [art.id, art]));
  let serial=0;
  function dominant(stats) {
    // Mirrors engine/HexWars.Engine/Roles.cs for this isolated interface study.
    const ranked=[['Striker',stats.damage],['Sniper',stats.range+stats.fireUp],
      ['Spotter',stats.vision+stats.seeUp],['Runner',stats.movement],['Climber',stats.climb],
      ['Bulwark',stats.defense],['Brute',stats.health]];
    const max=Math.max(0,...ranked.map(([,value])=>value));
    const leaders=ranked.filter(([,value])=>value===max);
    return leaders.length===1?leaders[0][0]:'Generalist';
  }
  const forRole = role => catalog.find(art=>art.role===role) || catalog[0];
  function render(id,{finish='porcelain',team='friendly',selected=false,label='',silhouette=false}={}) {
    const art=byId.get(id)||catalog[0];
    const key=`unit-${++serial}`;
    const graphite=finish==='graphite';
    const hi=graphite?'#b3c4ce':'#fffdf2',mid=graphite?'#6d8492':'#d8dfd9';
    const low=graphite?'#344852':'#8fa6ab',dark=graphite?'#1b2c35':'#647e88';
    const teamColor=team==='enemy'?'#f0a57e':'#80d9c4';
    const safeLabel=String(label||`${art.name}, ${art.role} form`).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const p=(d,fill='body',extra='')=>`<path d="${d}" fill="${fill.startsWith('#')?fill:`url(#${key}-${fill})`}" ${extra}/>`;
    const seam=(d)=>`<path d="${d}" fill="none" stroke="${teamColor}" stroke-width="2.7" stroke-linecap="round"/>`;
    const edge=(d)=>`<path d="${d}" fill="none" stroke="${hi}" stroke-opacity=".66" stroke-width="1.1" stroke-linecap="round"/>`;
    let body='';
    switch(art.shape) {
      case 'relay':
        body=p('M100 39 135 66 131 122 100 143 69 122 65 66Z')+
          p('M100 39 135 66 100 83 65 66Z','top')+p('M100 83 135 66 131 122 100 143Z','side')+
          p('M100 83 100 143 69 122 65 66Z','body')+seam('M100 88v37')+edge('M66 66 100 83 134 66');break;
      case 'atlas':
        body=p('M50 88Q49 68 68 61L87 53Q100 49 113 53L132 61Q151 68 150 88L144 129Q141 141 124 146L105 151Q100 153 95 151L76 146Q59 141 56 129Z')+
          p('M50 88Q49 68 68 61L87 53Q100 49 113 53L132 61Q151 68 150 88L120 101Q100 108 80 101Z','top')+
          p('M104 106 150 88 144 129Q141 141 124 146L105 151Z','side')+
          p('M66 113 82 119 82 133 68 128Z','side')+p('M119 119 136 113 134 128 119 134Z','dark')+
          seam('M85 100q15 5 30 0')+edge('M57 89 83 101Q100 107 117 101L144 90');break;
      case 'edge':
        body=p('M66 132 79 73 139 39 128 100 106 144Z')+
          p('M79 73 139 39 103 105 66 132Z','top')+
          p('M103 105 139 39 128 100 106 144Z','side')+
          p('M66 132 103 105 106 144 91 151Z','body')+
          seam('M106 107 129 64')+edge('M79 74 138 40');break;
      case 'bastion':
        body=p('M51 68 94 47 94 145 57 129Z')+
          p('M51 68 67 76 94 62 94 47Z','top')+
          p('M67 76 94 62 94 145 70 136Z','side')+
          p('M106 47 149 68 143 129 106 145Z','body')+
          p('M106 47 149 68 133 76 106 62Z','top')+
          p('M106 62 133 76 130 136 106 145Z','side')+
          seam('M100 81v50')+edge('M52 68 66 76M134 76 148 68');break;
      case 'glide':
        body=p('M41 127Q56 94 137 69L115 93Q77 106 69 133L58 143Z')+
          p('M76 148Q91 112 157 88L145 111Q112 129 105 156Z','side')+
          p('M55 127Q65 103 137 69L115 93Q82 110 69 133Z','top')+
          p('M88 143Q102 117 157 88L145 111Q117 130 105 149Z','top')+
          p('M73 106 103 90 126 103 98 120Z','body')+seam('M85 107 113 94')+
          edge('M58 123Q78 95 137 69M92 140Q112 114 157 88');break;
      case 'crux':
        body=p('M95 49 108 46 123 58 143 131 129 143 114 98 109 142 95 151 88 96 67 140 52 132 77 64Z')+
          p('M77 64 95 49 108 46 123 58 102 75Z','top')+
          p('M102 75 123 58 143 131 129 143 111 83 109 142 95 151Z','side')+
          p('M77 64 102 75 88 96 67 140 52 132Z','body')+
          seam('M82 70 98 78')+edge('M78 64 102 75 122 58');break;
      case 'lance':
        body=p('M56 114 83 83 122 94 138 121 103 148 66 137Z')+
          p('M56 114 83 83 122 94 103 121Z','top')+
          p('M103 121 122 94 138 121 103 148Z','side')+
          p('M83 94 81 86 146 39 158 40 158 50 98 104Z','body')+
          p('M81 86 146 39 158 40 94 92Z','top')+
          p('M94 92 158 40 158 50 98 104Z','side')+
          p('M146 39 158 40 158 50 147 48Z','dark')+
          seam('M70 116 93 123')+edge('M83 85 146 39');break;
      case 'halo':
        body=p('M80 111 96 99 113 104 124 140 102 151 76 140Z','body')+
          p('M96 99 113 104 124 140 102 151Z','side')+
          // The ring is a real open silhouette, not a painted circle.
          p('M147 63C147 30 123 24 96 41C69 58 51 88 54 111C57 137 79 140 107 121C131 105 148 83 147 63ZM128 69C128 83 116 99 101 109C84 120 73 117 73 103C73 89 85 69 101 58C118 47 129 53 128 69Z','side','fill-rule="evenodd"')+
          p('M141 59C141 27 117 23 91 40C65 57 47 86 50 109C53 133 76 136 101 119C126 103 142 80 141 59ZM122 66C122 80 111 97 95 107C79 118 69 113 69 101C69 86 80 68 96 57C111 46 123 51 122 66Z','body','fill-rule="evenodd"')+
          seam('M83 128 99 120')+edge('M58 88C66 67 80 50 96 41');break;
    }
    return `<svg xmlns="http://www.w3.org/2000/svg" class="unit-art${silhouette?' silhouette':''}" viewBox="0 0 200 200" role="img" aria-label="${safeLabel}" data-art-id="${art.id}"><defs>
      <linearGradient id="${key}-body" x1="0" y1="0" x2=".9" y2="1" gradientUnits="objectBoundingBox"><stop stop-color="${hi}"/><stop offset=".46" stop-color="${mid}"/><stop offset="1" stop-color="${low}"/></linearGradient>
      <linearGradient id="${key}-top" x1="0" y1="0" x2="1" y2="1"><stop stop-color="${hi}"/><stop offset="1" stop-color="${mid}"/></linearGradient>
      <linearGradient id="${key}-side" x1="0" y1="0" x2="1" y2=".7"><stop stop-color="${low}"/><stop offset="1" stop-color="${dark}"/></linearGradient>
      <linearGradient id="${key}-dark"><stop stop-color="${dark}"/><stop offset="1" stop-color="#13242d"/></linearGradient>
      <radialGradient id="${key}-shadow"><stop stop-color="#000" stop-opacity=".5"/><stop offset="1" stop-color="#000" stop-opacity="0"/></radialGradient>
    </defs><ellipse class="cast-shadow" cx="104" cy="159" rx="75" ry="28" fill="url(#${key}-shadow)"/>
    <g class="footing"><path d="M52 137 100 119 148 137V150L100 168 52 150Z" fill="#182831" stroke="#40545d" stroke-width=".7"/><path d="M52 137 100 119 148 137 100 155Z" fill="#263940"/>
    <path d="M57 147 100 163 143 147" fill="none" stroke="${teamColor}" stroke-width="${selected?3:2}" ${team==='enemy'?'stroke-dasharray="8 7"':''}/></g>
    <g class="sculpture">${body}</g>${selected?`<path class="selection" d="M37 140 37 153 57 161M163 140v13l-20 8" fill="none" stroke="${teamColor}" stroke-width="2"/>`:''}</svg>`;
  }
  return {catalog,byId,dominant,forRole,render};
})();
