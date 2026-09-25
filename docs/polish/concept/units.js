'use strict';
// A local review prototype. This does not submit commands or assets to HexWars/Steam.
const art = window.HexUnitArt;
const el = id => document.getElementById(id);
const storageKey = 'hexwars.polish.unit-designs.v1';
const statDefs = [
  ['health','Health',1],['damage','Damage',0],['defense','Defense',0],
  ['movement','Movement',0],['climb','Climb',0],['range','Range',0],
  ['fireUp','Firing ascent',0],['vision','Vision',0],['seeUp','Sight ascent',0]
];
const examples = {
  sniper:{name:'Longshot',stats:{health:3,damage:4,defense:0,movement:2,climb:1,range:5,fireUp:1,vision:2,seeUp:1}},
  spotter:{name:'Lookout',stats:{health:2,damage:0,defense:0,movement:3,climb:2,range:0,fireUp:0,vision:5,seeUp:2}},
  bulwark:{name:'Holdfast',stats:{health:5,damage:2,defense:7,movement:1,climb:0,range:1,fireUp:0,vision:2,seeUp:0}},
  runner:{name:'Courier',stats:{health:3,damage:1,defense:0,movement:6,climb:2,range:1,fireUp:0,vision:2,seeUp:1}},
  generalist:{name:'Allrounder',stats:{health:3,damage:3,defense:2,movement:3,climb:1,range:2,fireUp:1,vision:2,seeUp:1}}
};
let stats={...examples.sniper.stats}, artChoice='auto', finish='porcelain', team='friendly', silhouette=false;
let editingId=null, saved=[], storageAvailable=true;
const sanitizeName=value=>String(value||'').trim().replace(/[^A-Za-z0-9 '\-]/g,'').slice(0,20).trim();
const validStats=value=>!!value && statDefs.every(([key,,min])=>Number.isInteger(value[key]) && value[key]>=min && value[key]<=12);
const resolveArt=()=>artChoice==='auto'?art.forRole(art.dominant(stats)):art.byId.get(artChoice)||art.catalog[0];
const totalCost=()=>Object.values(stats).reduce((sum,value)=>sum+value,0);
const status=message=>{el('save-status').textContent=message;};
function roleReason(role) {
  const scores=[['damage',stats.damage],['range + firing ascent',stats.range+stats.fireUp],
    ['vision + sight ascent',stats.vision+stats.seeUp],['movement',stats.movement],
    ['climb',stats.climb],['defense',stats.defense],['health',stats.health]];
  const top=Math.max(...scores.map(([,value])=>value));
  return role==='Generalist'?`The highest stat scores tie at ${top}. No single role dominates.`:
    `${scores.find(([,value])=>value===top)[0]} leads at ${top}. Appearance does not add abilities.`;
}
for(const [key,label,min] of statDefs) {
  const cell=document.createElement('div');cell.className='stat-cell';
  const text=document.createElement('label');text.htmlFor=`unit-stat-${key}`;text.textContent=label;
  const input=document.createElement('input');Object.assign(input,{id:`unit-stat-${key}`,type:'number',min:String(min),max:'12',step:'1',value:String(stats[key])});
  input.addEventListener('change',()=>{
    const value=Number(input.value);
    stats[key]=Number.isFinite(value)?Math.max(min,Math.min(12,Math.floor(value))):min;
    input.value=stats[key];render();
  });cell.append(text,input);el('stat-grid').append(cell);
}
for(const [index,item] of art.catalog.entries()) {
  const button=document.createElement('button');button.className='art-card';button.dataset.artId=item.id;
  button.setAttribute('aria-label',`Choose ${item.name} appearance`);button.setAttribute('aria-pressed','false');
  button.innerHTML=`<span class="card-top"><span class="art-index">${String(index+1).padStart(2,'0')} / ${item.name.toUpperCase()}</span><span class="suggested"></span></span><div class="card-art"></div><h3>${item.name}</h3><span class="card-bottom"><span class="card-role">${item.role} form</span><span class="selection-tick" aria-hidden="true">Selected ✓</span></span>`;
  button.addEventListener('click',()=>{artChoice=item.id;render();status(`${item.name} selected. This appearance will stay selected as you edit stats.`);});
  el('art-collection').append(button);
}
function render() {
  const chosen=resolveArt(),role=art.dominant(stats),suggested=art.forRole(role);
  const options={finish,team,silhouette};
  el('form-number').textContent=String(art.catalog.indexOf(chosen)+1).padStart(2,'0');
  el('form-name').textContent=chosen.name;el('form-cue').textContent=chosen.cue;
  el('choice-mode').textContent=artChoice==='auto'?'Automatic':'Your choice';
  el('hero-art').innerHTML=art.render(chosen.id,{...options,selected:true});
  for(const size of [32,48,72])el(`scale-${size}`).innerHTML=art.render(chosen.id,options);
  el('point-cost').textContent=totalCost();el('derived-role').textContent=role;
  el('role-reason').textContent=roleReason(role);
  el('appearance-value').textContent=`${chosen.name} · ${finish==='porcelain'?'Porcelain':'Graphite'}`;
  el('auto-recommendation').textContent=`${suggested.name} is suggested for ${role}.`;
  el('auto-art').setAttribute('aria-pressed',String(artChoice==='auto'));
  el('art-explanation').textContent=artChoice==='auto'?'Automatic art follows your stats. Choose a form below to keep a specific appearance.':
    `You chose ${chosen.name}. ${chosen.role!==role?`Your stats still describe a ${role}. `:''}Stat edits will not replace this form. Use “Match art to role” to return to automatic.`;
  document.querySelector('.coordinate.right').textContent=finish==='porcelain'?'PORCELAIN / SATIN':'GRAPHITE / SATIN';
  for(const button of document.querySelectorAll('.art-card')) {
    const id=button.dataset.artId;
    button.setAttribute('aria-pressed',String(id===chosen.id));
    button.querySelector('.suggested').textContent=id===suggested.id?'Suggested':'';
    button.querySelector('.card-art').innerHTML=art.render(id,{...options,label:`${art.byId.get(id).name}, ${art.byId.get(id).role} form`});
  }
  el('board-chosen').innerHTML=art.render(chosen.id,{...options,selected:true});
  el('neighbor-a').innerHTML=art.render('halo-01',{...options,team:'friendly'});
  el('neighbor-b').innerHTML=art.render('bastion-01',{...options,team:'enemy'});
  el('board-name').textContent=sanitizeName(el('design-name').value)||'Unnamed design';
  el('board-role').textContent=role;
}
function syncInputs() {for(const [key] of statDefs)el(`unit-stat-${key}`).value=stats[key];}
el('auto-art').addEventListener('click',()=>{artChoice='auto';render();status('Automatic matching enabled. The form follows the dominant role; tied scores use Relay.');});
el('design-name').addEventListener('input',()=>{el('board-name').textContent=sanitizeName(el('design-name').value)||'Unnamed design';});
el('design-name').addEventListener('change',()=>{el('design-name').value=sanitizeName(el('design-name').value);render();});
el('preset').addEventListener('change',()=>{
  const example=examples[el('preset').value];stats={...example.stats};el('design-name').value=example.name;
  // An example is a new draft; manual appearance is intentionally retained across stat replacement.
  editingId=null;el('save-design').innerHTML='Save design <span aria-hidden="true">↗</span>';
  syncInputs();render();status(artChoice==='auto'?'New example loaded. Automatic art follows its role.':'New example loaded. Your chosen appearance is retained.');
});
for(const button of document.querySelectorAll('[data-finish]'))button.addEventListener('click',()=>{
  finish=button.dataset.finish;syncFinish();render();
});
function syncFinish(){document.querySelectorAll('[data-finish]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.finish===finish)));}
for(const button of document.querySelectorAll('[data-team]'))button.addEventListener('click',()=>{
  team=button.dataset.team;document.querySelectorAll('[data-team]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));render();
});
el('silhouette').addEventListener('change',()=>{silhouette=el('silhouette').checked;render();});
function record() {
  return {schemaVersion:1,id:editingId||`design-${Date.now()}-${Math.random().toString(36).slice(2,8)}`,
    name:sanitizeName(el('design-name').value)||'Unnamed design',stats:{...stats},
    appearance:{selection:artChoice,resolvedArtId:resolveArt().id,finish}};
}
function renderSaved() {
  const root=el('saved-designs');root.replaceChildren();
  if(!saved.length){const message=document.createElement('p');message.className='empty';message.textContent='Your first saved design will appear here.';root.append(message);return;}
  for(const item of saved) {
    const id=item.appearance.selection==='auto'?art.forRole(art.dominant(item.stats)).id:item.appearance.selection;
    const button=document.createElement('button');button.className='saved-card';button.dataset.designId=item.id;
    button.setAttribute('aria-label',`Open saved design ${item.name}`);
    button.innerHTML=art.render(id,{finish:item.appearance.finish});
    const copy=document.createElement('div'),name=document.createElement('strong'),caption=document.createElement('span');
    name.textContent=item.name;caption.textContent=`${art.byId.get(id).name} · ${item.appearance.finish} · ${Object.values(item.stats).reduce((a,b)=>a+b,0)} points`;
    copy.append(name,caption);button.append(copy);button.addEventListener('click',()=>{
      stats={...item.stats};artChoice=item.appearance.selection;finish=item.appearance.finish;editingId=item.id;
      el('design-name').value=item.name;syncInputs();syncFinish();render();
      el('save-design').innerHTML='Update design <span aria-hidden="true">↗</span>';
      status(`${item.name} opened with its saved form and finish.`);
      document.querySelector('.workbench').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});
    });root.append(button);
  }
}
el('save-design').addEventListener('click',()=>{
  const item=record(),index=saved.findIndex(existing=>existing.id===item.id);
  if(index<0)saved.push(item);else saved[index]=item;
  editingId=item.id;el('design-name').value=item.name;
  if(storageAvailable)try{localStorage.setItem(storageKey,JSON.stringify({schemaVersion:1,designs:saved}));}catch{storageAvailable=false;}
  renderSaved();render();el('save-design').innerHTML='Update design <span aria-hidden="true">↗</span>';
  status(storageAvailable?`${item.name} saved in this browser, including ${resolveArt().name} and ${finish}.`:
    `${item.name} kept for this session only. Browser storage is unavailable or contains unreadable data; Export JSON to keep it.`);
});
el('export-design').addEventListener('click',()=>{
  const item=record(),blob=new Blob([JSON.stringify(item,null,2)+'\n'],{type:'application/json'});
  const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=`hexwars-${item.name.toLowerCase().replace(/[^a-z0-9]+/g,'-')}.json`;
  document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  status('Exported a concept design with stats, appearance selection, resolved art ID and finish.');
});
try {
  const raw=localStorage.getItem(storageKey);
  if(raw) {
    const data=JSON.parse(raw);
    if(data.schemaVersion!==1||!Array.isArray(data.designs))throw Error('Unknown storage format');
    const ids=new Set();let fallback=false;
    for(const item of data.designs) {
      if(item.schemaVersion!==1||typeof item.id!=='string'||ids.has(item.id)||!validStats(item.stats))throw Error('Invalid saved design');
      ids.add(item.id);const appearance=item.appearance||{};
      let selection=appearance.selection;
      if(selection!=='auto'&&!art.byId.has(selection)){selection='auto';fallback=true;}
      saved.push({schemaVersion:1,id:item.id,name:sanitizeName(item.name)||'Unnamed design',
        stats:Object.fromEntries(statDefs.map(([key])=>[key,item.stats[key]])),
        appearance:{selection,resolvedArtId:selection==='auto'?art.forRole(art.dominant(item.stats)).id:selection,finish:appearance.finish==='graphite'?'graphite':'porcelain'}});
    }
    if(fallback)status('An unavailable saved form uses automatic role art. Stats and names were kept.');
  }
} catch {saved=[];storageAvailable=false;status('Saved designs could not be read. Existing browser data is untouched; use Export JSON for this session.');}
const ns='http://www.w3.org/2000/svg';
for(let row=0;row<4;row++)for(let col=0;col<6;col++){
  if((row===0&&col===0)||(row===3&&col===5))continue;
  const x=77+col*101+(row%2)*50,y=78+row*55;
  const group=document.createElementNS(ns,'g');group.setAttribute('transform',`translate(${x} ${y})`);
  const side=document.createElementNS(ns,'path');side.setAttribute('d','M-50 0-25 28h50L50 0v12L25 40h-50l-25-28Z');side.setAttribute('fill','#1a282e');side.setAttribute('stroke','#31444a');
  const top=document.createElementNS(ns,'path');top.setAttribute('d','M-50 0-25-28h50L50 0 25 28h-50Z');top.setAttribute('fill',(row+col)%5===0?'#354b4c':'#293b40');top.setAttribute('stroke','#506267');top.setAttribute('stroke-width','.7');
  group.append(side,top);el('board-hexes').append(group);
}
render();renderSaved();
