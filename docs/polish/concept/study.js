'use strict';
// Illustrative interface fixture only. Production forecasts must use the game engine.
const $ = id => document.getElementById(id);
const directionNotes = {
  workshop: 'Painted pieces, warm surfaces, precise mechanical details. Make a unit feel like something you built.',
  atlas: 'Contour marks, printed counters, a field-book palette. Make terrain and intent readable at a glance.',
  orbital: 'Modular machines, altitude marks, restrained industrial contrast. Extend the floating battlefield into an orbital foundry.'
};
document.querySelectorAll('button[data-direction]').forEach(button => button.addEventListener('click', () => {
  document.body.dataset.direction = button.dataset.direction;
  document.querySelectorAll('button[data-direction]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
  $('direction-note').textContent = directionNotes[button.dataset.direction];
}));
let currentScreen = 'battle';
document.querySelectorAll('button[data-screen]').forEach(button => button.addEventListener('click', () => {
  currentScreen = button.dataset.screen;
  document.querySelectorAll('button[data-screen]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
  for (const screen of ['battle','design','learn']) $(screen + '-panel').hidden = currentScreen !== screen;
  $('field-kicker').textContent = currentScreen === 'design' ? 'Your next invention' : currentScreen === 'learn' ? 'One decision at a time' : 'Selected / 01';
  $('field-heading').textContent = currentScreen === 'design' ? 'What does your army need?' : currentScreen === 'learn' ? 'Learn on the board.' : 'Longshot';
  $('field-copy').textContent = currentScreen === 'design' ? 'A scout? A wall? A very particular answer.' : currentScreen === 'learn' ? 'Use a real situation to teach one idea.' : 'A little height goes a long way.';
}));
function forecast() {
  const height = $('high-ground').checked ? 1 : 0;
  const damage = 5 + height - 2 - 1;
  $('damage').textContent = damage;
  $('remaining-hp').textContent = `Target: 5 → ${5-damage} HP`;
  $('height-bonus').textContent = height ? '+1' : '0';
  $('hp-remaining').style.width = `${(5-damage)*20}%`;
  $('hp-lost').style.width = `${damage*20}%`;
  document.querySelector('.hp-track').setAttribute('aria-label', `${5-damage} of 5 target HP remaining after the example hit`);
}
$('high-ground').addEventListener('change', forecast); forecast();
const statDefs = [
  ['health','Health',1],['damage','Damage',0],['movement','Movement',0],['range','Range',0],['vision','Vision',0],
  ['defense','Defense',0],['climb','Climb',0],['fireUp','Firing ascent',0],['seeUp','Sight ascent',0]
];
const templates = {
  longshot:{health:3,damage:5,movement:1,range:4,vision:2,defense:0,climb:0,fireUp:1,seeUp:1},
  spotter:{health:2,damage:0,movement:3,range:0,vision:5,defense:0,climb:2,fireUp:0,seeUp:2},
  bulwark:{health:7,damage:2,movement:0,range:1,vision:2,defense:5,climb:0,fireUp:0,seeUp:1}
};
let stats = {...templates.longshot};
function updateStats() {
  $('cost').textContent = Object.values(stats).reduce((a,b)=>a+b,0);
  for(const [key,,min] of statDefs) {
    $(`stat-${key}`).value = stats[key];
    $(`minus-${key}`).disabled = stats[key] <= min;
    $(`plus-${key}`).disabled = stats[key] >= 12;
  }
  const limits = [];
  if(stats.damage === 0) limits.push('Cannot deal damage, even from high ground.');
  if(stats.movement === 0) limits.push('Cannot reposition.');
  else if(stats.climb === 0) limits.push('Cannot climb to a higher hex.');
  if(stats.defense === 0) limits.push('No purchased defense; terrain can still help.');
  if(stats.health <= 2) limits.push('Little health to spare.');
  if(!limits.length) limits.push('More capability costs more points. Test this allocation against a specific problem.');
  $('tradeoff').textContent = limits.slice(0,2).join(' ');
}
for(const [index,[key,label,min]] of statDefs.entries()) {
  const row = document.createElement('div'); row.className='stat-row';
  const labelEl=document.createElement('label'); labelEl.htmlFor=`stat-${key}`; labelEl.textContent=label;
  const minus=document.createElement('button'); minus.id=`minus-${key}`;minus.textContent='−';minus.setAttribute('aria-label',`Decrease ${label}`);
  const input=document.createElement('input');input.id=`stat-${key}`;input.type='number';input.min=min;input.max=12;input.step=1;input.value=stats[key];
  const plus=document.createElement('button');plus.id=`plus-${key}`;plus.textContent='+';plus.setAttribute('aria-label',`Increase ${label}`);
  minus.addEventListener('click',()=>{stats[key]=Math.max(min,stats[key]-1);updateStats();});
  plus.addEventListener('click',()=>{stats[key]=Math.min(12,stats[key]+1);updateStats();});
  input.addEventListener('change',()=>{const value=Number(input.value);stats[key]=Number.isFinite(value)?Math.max(min,Math.min(12,Math.floor(value))):min;updateStats();});
  row.append(labelEl,minus,input,plus);$(index<5?'main-stats':'extra-stats').append(row);
}
$('template').addEventListener('change',()=>{stats={...templates[$('template').value]};updateStats();});updateStats();
const lessons = [
  ['Start with a useful move.','Select Longshot. Its movement and attack markers should tell you what you can do this turn.','Success: explain the difference between moving and attacking, without opening the rules.'],
  ['Predict the hit.','Compare a shot from level ground with the same shot from one level higher. Read the forecast before committing.','Success: predict the target’s remaining HP and explain the height bonus.'],
  ['Build the counter.','After earning points, change a template to answer one new problem. Keep the original so you can compare both attempts.','Success: name the capability you bought and the cost or weakness you accepted.']
];
let lesson=0;
$('next-lesson').addEventListener('click',()=>{
  lesson=(lesson+1)%lessons.length;
  $('lesson-count').textContent=`${lesson+1} of 3`;
  [$('lesson-title'),$('lesson-copy'),$('lesson-goal')].forEach((node,i)=>node.textContent=lessons[lesson][i]);
  document.querySelectorAll('.lesson-steps li').forEach((li,i)=>li.classList.toggle('active',i===lesson));
  $('next-lesson').textContent=lesson===2?'Return to the first moment ↺':'Preview next teaching moment →';
});
const ns='http://www.w3.org/2000/svg';
function svg(tag,attrs,parent){const el=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);parent.append(el);return el;}
const positions={};
for(let r=0;r<5;r++)for(let q=0;q<7;q++){
  if((r===0 && (q===0||q===6))||(r===4&&q===6))continue;
  const elevation=(q===2&&r===2)?2:((q===1&&r===3)||(q+r)%5===0?1:0);
  const x=103+q*99+(r%2)*49,y=181+r*67-elevation*21,depth=18+elevation*21;
  positions[`${q},${r}`]={x,y};
  const g=svg('g',{transform:`translate(${x} ${y})`},$('tiles'));
  svg('polygon',{points:`-51,0 -25.5,29 25.5,29 51,0 51,${depth} 25.5,${29+depth} -25.5,${29+depth} -51,${depth}`,class:'tile-side'},g);
  const terrain=(q===4&&r===2)||(q===5&&r===1)?'forest':(q===3&&r!==2)?'water':elevation?'rock':'';
  svg('polygon',{points:'-51,0 -25.5,-29 25.5,-29 51,0 25.5,29 -25.5,29',class:`tile-top ${terrain}`},g);
  if(elevation)svg('path',{d:`M-47 ${depth-3}l23 26h49l23-26`,class:'contour'},g);
  if(terrain==='forest')svg('path',{d:'M-20 5l7-13 7 13M-2 8l8-16 8 16M12-2l6-11 6 11',class:'contour'},g);
  if(terrain==='water')svg('path',{d:'M-23-4q10-5 20 0t20 0M-23 7q10-5 20 0t20 0',class:'contour'},g);
}
const attacker=positions['1,3'],target=positions['4,2'];
svg('path',{d:`M${attacker.x} ${attacker.y-12}Q${attacker.x+110} ${attacker.y-130} ${target.x} ${target.y-12}`,class:'route'},$('routes'));
function piece(q,r,team,kind,label){
  const p=positions[`${q},${r}`],g=svg('g',{transform:`translate(${p.x} ${p.y-5})`,class:`piece-${team}`},$('pieces'));
  svg('ellipse',{cx:4,cy:12,rx:31,ry:15,class:'piece-shadow'},g);
  if(label)svg('ellipse',{cx:0,cy:7,rx:37,ry:22,class:team==='friendly'?'selected-ring':'target-ring'},g);
  svg('path',{d:'M-24-3v12c0 17 48 17 48 0V-3',class:'piece-body'},g);
  svg('ellipse',{cx:0,cy:-3,rx:24,ry:14,class:'piece-body'},g);
  if(kind==='longshot'){
    svg('path',{d:'M-13-8v-15l12-7 16 8v16z',class:'piece-light'},g);
    svg('path',{d:'M0-22l30-16 4 7L7-14z',class:'piece-body'},g);
    svg('circle',{cx:1,cy:-9,r:3,fill:'currentColor'},g);
  }else if(kind==='bulwark'){
    svg('path',{d:'M-18-8v-23l18-7 18 7v23L0 2z',class:'piece-body'},g);
    svg('path',{d:'M-10-26v16L0-4l10-6v-16',class:'piece-mark'},g);
  }else{
    svg('path',{d:'M-12-7v-15l12-8 12 8v15L0 0z',class:'piece-body'},g);
    svg('path',{d:'M-5-19l7 5-7 5M3-19l7 5-7 5',class:'piece-mark'},g);
  }
  if(team==='enemy')svg('path',{d:'M-22 7l5 3m11 5h9m13-5 5-3',class:'piece-mark'},g);
  if(label){const t=svg('text',{x:0,y:42,'text-anchor':'middle',class:'piece-label'},g);t.textContent=label;}
}
piece(5,0,'enemy','longshot','');piece(0,2,'friendly','bulwark','');piece(4,2,'enemy','bulwark','BULWARK');piece(1,3,'friendly','longshot','LONGSHOT');piece(2,4,'friendly','runner','');
