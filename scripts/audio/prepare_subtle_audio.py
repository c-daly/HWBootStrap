"""Master the existing recordings and render a small tactile cue palette.

Requires NumPy. Original recordings are read-only. Outputs 44.1 kHz PCM WAVs,
a level/peak receipt, and a browser audition using the exact game clips.
Run from the repository root: python scripts/audio/prepare_subtle_audio.py
"""
from pathlib import Path
import hashlib, json, math, wave
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'Assets/HexWars/Resources/Audio'
BANK = SOURCE / 'Soft'
STUDY = ROOT / 'engine/HexWars.NetServer/wwwroot/sound-study'
RATE = 44100
RNG = np.random.default_rng(25419)

def read(path):
    with wave.open(str(path)) as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        x = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = ((x ^ (1 << 23)) - (1 << 23)).astype(float) / (1 << 23)
    else:
        assert width == 2
        x = np.frombuffer(raw, dtype='<i2').astype(float) / 32768
    x = x.reshape(-1, channels)
    if rate != RATE:
        # Original weapon recordings are 48 kHz. Band-limit before interpolating.
        n = 1 << (len(x) - 1).bit_length()
        f = np.fft.rfftfreq(n, 1 / rate)
        x = np.fft.irfft(np.fft.rfft(x, n, axis=0) * (1 / np.sqrt(1 + (f / 10000) ** 12))[:, None], n, axis=0)[:len(x)]
        t = np.arange(round(len(x) * RATE / rate)) / RATE
        x = np.stack([np.interp(t, np.arange(len(x)) / rate, x[:, c]) for c in range(channels)], axis=1)
    return x

def write(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x)
    assert np.isfinite(x).all() and abs(x).max() < .999, str(path)
    with wave.open(str(path), 'wb') as w:
        w.setparams((1 if x.ndim == 1 else x.shape[1], 2, RATE, 0, 'NONE', 'not compressed'))
        w.writeframes(np.round(x * 32767).astype('<i2').tobytes())

def fade(x, attack=.004, release=.065):
    x = x.copy()
    a, b = min(len(x), round(attack * RATE)), min(len(x), round(release * RATE))
    if a: x[:a] *= np.linspace(0, 1, a).reshape((-1,) + (1,) * (x.ndim-1)) ** 2
    if b: x[-b:] *= np.linspace(1, 0, b).reshape((-1,) + (1,) * (x.ndim-1)) ** 2
    return x

def band(x, low=110, high=4200):
    n = 1 << (len(x) * 2 - 1).bit_length()
    f = np.fft.rfftfreq(n, 1 / RATE)
    response = (f / np.sqrt(f*f + low*low)) ** 2 / np.sqrt(1 + (f / high) ** 8)
    return np.fft.irfft(np.fft.rfft(x, n) * response, n)[:len(x)]

def master(x, rms, peak=.20):
    x = x - np.mean(x)
    x = fade(x)
    scale = min(rms / max(1e-9, np.sqrt(np.mean(x*x))), peak / max(1e-9, abs(x).max()))
    return x * scale

def contact(seconds, body, roughness=.25):
    t = np.arange(round(seconds * RATE)) / RATE
    grain = band(RNG.normal(size=len(t)), 350, 3200) * np.exp(-t * 45)
    resonance = np.sin(2*np.pi*body*t)*np.exp(-t*36) + .32*np.sin(2*np.pi*body*2.37*t)*np.exp(-t*55)
    return resonance + roughness * grain

def notes(freqs, step=.13, tail=.28):
    x = np.zeros(round((step * (len(freqs)-1) + tail) * RATE))
    t = np.arange(round(tail*RATE))/RATE
    for i, f in enumerate(freqs):
        tone = (np.sin(2*np.pi*f*t) + .16*np.sin(2*np.pi*f*2*t)) * (1-np.exp(-t*95))*np.exp(-t*13)
        start=round(i*step*RATE);x[start:start+len(tone)] += tone[:len(x)-start]
    return x

BANK.mkdir(parents=True, exist_ok=True)
STUDY.mkdir(parents=True, exist_ok=True)
clips = {}
for family, duration, rms, peak, high in [('Light', .30, .027, .16, 4200), ('Mid', .43, .038, .19, 3600), ('Heavy', .60, .045, .22, 3000)]:
    for variant in range(4):
        name=f'Attack{family}_{variant}'
        x=read(SOURCE/(name+'.wav')).mean(axis=1)[:round(duration*RATE)]
        clips[name]=master(band(x, 100, high), rms, peak)
for name, original, duration, rms in [('Deploy','DeployDoor',.44,.027),('Design','CreateRacked',.25,.017)]:
    x=read(SOURCE/(original+'.wav')).mean(axis=1)[:round(duration*RATE)]
    clips[name]=master(band(x, 140, 3500),rms,.15)
clips['Select']=master(contact(.075,430,.2),.006,.032)
clips['Move']=master(contact(.18,175,.8),.017,.09)
clips['Place']=master(contact(.14,210,.35),.019,.095)
clips['Build']=master(contact(.24,150,.6),.024,.13)
clips['Claim']=master(notes([330,440],.09,.20),.017,.09)
clips['EndTurn']=master(notes([392,523.25],.11,.25),.018,.075)
clips['Win']=master(notes([261.63,329.63,392],.15,.42),.022,.095)
t=np.arange(round(.38*RATE))/RATE
clips['Death']=master(contact(.38,115,1.6)+band(RNG.normal(size=len(t)),160,1800)*np.exp(-t*17),.032,.18)
for name,x in clips.items(): write(BANK/(name+'.wav'),x)

# Exact previous procedural move/turn cues, for an honest level-matched-input A/B.
def previous_click(freq, seconds, volume):
    n=int(RATE*seconds);u=np.arange(n)/n
    phase=np.cumsum(freq*(1-.15*u)/RATE*2*np.pi)
    return np.sin(phase)*np.minimum(1,(1-u)*4)*(1-u)*volume

examples=[('Move','Move','A soft contact instead of a high beep.',previous_click(660,.07,.18)),
          ('AttackLight_0','Light shot','Shorter, rounded transients and room for the impact.',read(SOURCE/'AttackLight_0.wav')),
          ('AttackMid_0','Medium shot','Tamed peak and a shorter tail.',read(SOURCE/'AttackMid_0.wav')),
          ('AttackHeavy_0','Heavy shot','More weight, without a long low rumble.',read(SOURCE/'AttackHeavy_0.wav')),
          ('Deploy','Deployment','A brief mechanical latch.',read(SOURCE/'DeployDoor.wav')),
          ('Design','Design saved','A small confirmation, with no fanfare.',read(SOURCE/'CreateRacked.wav')),
          ('EndTurn','Turn change','A quiet two-note handover.',previous_click(330,.11,.22))]
rows=[]
for name,label,description,old in examples:
    write(STUDY/(name+'-before.wav'),old)
    write(STUDY/(name+'.wav'),clips[name])
    rows.append(dict(name=name,label=label,description=description,before=name+'-before.wav',after=name+'.wav'))
for name in ['Select','Place','Death','Win']:write(STUDY/(name+'.wav'),clips[name])
# An actual paced sequence, using the same mastered samples and default settings as the game.
sequence=[(0.4,'Select'),(1.2,'Move'),(3.3,'Select'),(4.1,'AttackLight_0'),(6.7,'AttackMid_0'),(7.6,'Death'),(10.8,'Deploy'),(13.6,'Design'),(16.8,'EndTurn')]
timeline=np.zeros(RATE*20)
for at,name in sequence:
    i=round(at*RATE);timeline[i:i+len(clips[name])] += clips[name]
write(STUDY/'sequence.wav',timeline)
# Short context excerpts. Original music remains the soundtrack; this pass changes its mix.
for name,gain in [('TitleMusic',.12),('AmbientBed',.05),('DesignerHum',.09)]:
    x=read(SOURCE/(name+'.wav'))[:RATE*12]
    if len(x)<RATE*12:x=np.tile(x,(math.ceil(RATE*12/len(x)),1))[:RATE*12]
    write(STUDY/(name+'.wav'),fade(x*gain,.8,.8))
report={'sample_rate':RATE,'seed':25419,'original_recordings_unchanged':True,'clips':{},'comparison':rows,
        'sequence':[{'seconds':at,'cue':name} for at,name in sequence],
        'default_loop_gains':{'TitleMusic':.12,'AmbientBed':.05,'DesignerHum':.09}}
for name,x in clips.items():
    report['clips'][name]={'seconds':round(len(x)/RATE,4),'peak_dbfs':round(float(20*np.log10(max(1e-9,abs(x).max()))),2),
                         'rms_dbfs':round(float(20*np.log10(max(1e-9,np.sqrt(np.mean(x*x))))),2),
                         'sha256':hashlib.sha256((BANK/(name+'.wav')).read_bytes()).hexdigest()}
(STUDY/'palette.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'Wrote {len(clips)} quiet cues; original recordings preserved.')
