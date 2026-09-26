"""Render the tabletop palette and an original, sparse upright-piano title theme.

Run from any directory with NumPy, SciPy and SoundFile installed. Sources and
licenses are checked in beside this script; no download or old weapon is used.
The first sound pass remains reproducible with prepare_subtle_audio.py.
"""
from pathlib import Path
from functools import lru_cache
from math import gcd
import hashlib
import json
import uuid
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, resample_poly

ROOT = Path(__file__).resolve().parents[2]
SOURCES = Path(__file__).resolve().parent / 'sources'
AUDIO = ROOT / 'Assets/HexWars/Resources/Audio'
BANK = AUDIO / 'Soft'
MUSIC = AUDIO / 'Tabletop'
STUDY = ROOT / 'engine/HexWars.NetServer/wwwroot/sound-study'
RATE = 44100
SEED = 250926


def read(path):
    x, rate = sf.read(path, always_2d=True)
    if rate != RATE:
        d = gcd(rate, RATE)
        x = resample_poly(x, RATE // d, rate // d, axis=0)
    return x


def write(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert np.isfinite(x).all() and np.max(np.abs(x)) < .999, path
    sf.write(path, x, RATE, subtype='PCM_16')


def taper(x, attack=.001, release=.025):
    x = x.copy()
    for at, n, reverse in [(0, round(attack*RATE), False), (-round(release*RATE), round(release*RATE), True)]:
        n = min(n, len(x))
        if not n:
            continue
        e = np.sin(np.linspace(0, np.pi/2, n)) ** 2
        if reverse:
            e = e[::-1]
            at = -n
        sl = slice(at, None if reverse else n)
        x[sl] *= e[:, None] if x.ndim == 2 else e
    return x


def filter_audio(x, low=65, high=2600):
    x = sosfilt(butter(2, low, 'highpass', fs=RATE, output='sos'), x, axis=0)
    return sosfilt(butter(3, high, 'lowpass', fs=RATE, output='sos'), x, axis=0)


def master(x, peak, rms):
    x = taper(x)
    # Correct the finite clip baseline without moving its silent endpoints. Some importers
    # remove DC during decoding; zero mean prevents that from turning silence into a step.
    window = np.sin(np.linspace(0, np.pi, len(x))) ** 2
    x -= window * (np.mean(x) / np.mean(window))
    gain = min(peak / max(np.max(np.abs(x)), 1e-9), rms / max(np.sqrt(np.mean(x*x)), 1e-9))
    # A short silent guard keeps platform codec pre-ringing away from the clip boundary.
    return np.pad(x * gain, (round(.016*RATE), round(.024*RATE)))


@lru_cache(None)
def foley(pack, name, pitch=1., high=2600):
    x = read(SOURCES/pack/(name+'.ogg')).mean(axis=1)
    # Preserve the tiny attack lead but remove the recording's empty pre-roll.
    onset = np.flatnonzero(np.abs(x) > np.max(np.abs(x)) * .012)
    if len(onset):
        x = x[max(0, onset[0]-round(.002*RATE)):]
    x = np.interp(np.arange(0, len(x)-1, pitch), np.arange(len(x)), x)
    x = filter_audio(x, high=high)
    x /= max(np.max(np.abs(x)), 1e-9)
    return taper(x)


def layer(seconds, *parts):
    x = np.zeros(round(seconds*RATE))
    for at, sample, gain in parts:
        start = round(at*RATE)
        n = min(len(sample), len(x)-start)
        x[start:start+n] += sample[:n] * gain
    return x


def effects():
    clips = {}
    for i in range(4):
        wood = foley('impact', f'impactWood_medium_{i:03}', .92 + i*.018, 2350)
        chip = foley('casino', f'chip-lay-{i%3+1}', .92, 2850)
        # Small hard contact, then the softer wooden body. No electronic pitch sweep.
        x = layer(.19, (0, chip, .20), (.012, wood, 1.))
        clips['Move' if i == 0 else f'Move_{i}'] = master(x, .135, .025)
    clips['Select'] = master(layer(.09, (0, foley('casino','chip-lay-2',1.05,2600),1)), .045, .010)
    clips['Place'] = master(layer(.22, (0, foley('casino','chip-lay-1',.9,2600),.2),
                                     (.009,foley('impact','impactWood_medium_002',.84,2300),1)), .15,.029)
    for tier, weight, pitch, seconds, peak, rms, high in [
        ('Light','light',.94,.17,.115,.022,2600),
        ('Mid','medium',.82,.22,.15,.030,2250),
        ('Heavy','heavy',.72,.28,.18,.038,1850)]:
        for i in range(4):
            body = foley('impact',f'impactWood_{weight}_{i:03}',pitch+i*.012,high)
            cushion = foley('impact',f'impactSoft_heavy_{i:03}',.88,1100)
            x = layer(seconds, (0, body, 1), (.006,cushion,.14 if tier=='Light' else .32))
            clips[f'Attack{tier}_{i}'] = master(x,peak,rms)
    clips['Death'] = master(layer(.32,
        (0,foley('impact','impactWood_heavy_001',.75,1800),.8),
        (.072,foley('casino','chip-lay-3',.78,2100),.22),
        (.012,foley('impact','impactSoft_heavy_002',.78,1000),.36)), .15,.028)
    clips['Deploy'] = master(layer(.24,(0,foley('impact','impactWood_medium_001',.9,2400),1),
        (.038,foley('casino','chip-lay-1',1,2600),.15)), .13,.025)
    clips['Design'] = master(layer(.18,(0,foley('casino','chip-lay-2',1,2600),1),
        (.038,foley('impact','impactWood_light_002',1,2400),.3)), .085,.016)
    clips['Build'] = master(layer(.25,(0,foley('impact','impactWood_heavy_003',.85,2100),1),
        (.050,foley('casino','chip-lay-1',.9,2200),.18)), .14,.028)
    # Match confirmations to the piano's D-major palette using the same sampled instrument.
    for name, score, seconds, peak, rms in [
        ('EndTurn',[(0,62),(.12,69)],.50,.075,.016),
        ('Claim',[(0,62),(.09,66)],.44,.075,.016),
        ('Win',[(0,62),(.15,66),(.3,69)],1.15,.105,.022)]:
        parts=[(at,piano(note,.48,.6).mean(axis=1),1) for at,note in score]
        clips[name]=master(layer(seconds,*parts),peak,rms)
    return clips


NOTES = {'C':0,'D#':3,'F#':6,'A':9}
PIANO_SAMPLES = {12*(int(f.stem[-3])+1)+NOTES[f.stem[:-3]]:f
                 for f in (SOURCES/'upright').glob('*vL.flac')}


@lru_cache(None)
def piano(midi, sustain, velocity):
    key = min(PIANO_SAMPLES, key=lambda key:abs(key-midi))
    raw=read(PIANO_SAMPLES[key]); ratio=2**((midi-key)/12)
    length=min(round((sustain+.8)*RATE),round(len(raw)/ratio))
    t=np.arange(length)*ratio
    x=np.stack([np.interp(t,np.arange(len(raw)),raw[:,c]) for c in range(raw.shape[1])],axis=1)
    # Use the quiet recorded velocity layer and gently soften its remaining hammer brightness.
    x=filter_audio(x,low=55,high=1900+velocity*1100)
    release=round(sustain*RATE)
    if release<len(x):x[release:]*=np.exp(-np.arange(len(x)-release)[:,None]/(RATE*.21))
    return taper(x,.003,.04) * velocity


def title_theme():
    rng=np.random.default_rng(SEED)
    beat=60/62
    length=round(64*beat*RATE)
    track=np.zeros((length,2))
    score=[]
    # Sixteen bars: a stated motif, a reply, and a lighter variation. Voicings leave
    # the melody room; the final suspended A returns gently to the opening D.
    chords=[(50,57,66),(49,57,64),(47,54,62),(43,50,59),
            (42,50,57),(40,47,55),(45,55,62),(45,52,61)]*2
    melody=[[(.5,66,1.0),(2.,69,1.3)],[(.25,68,.8),(1.5,64,1.5)],
            [(.5,66,.9),(2.,62,1.4)],[(1.,64,.8),(2.5,62,1.2)],
            [(.5,66,.8),(1.75,69,.8),(2.75,71,1.0)],[(.5,67,1.),(2.,66,1.2)],
            [(.75,64,1.2)],[(1.,61,1.2)],
            [(.5,66,.9),(2.,69,.8),(3.,74,.9)],[(.5,73,1.),(2.,69,1.4)],
            [(.5,71,.8),(1.75,69,.8),(2.75,66,1.0)],[(1.,67,1.),(2.5,66,.9)],
            [(.5,64,1.),(2.,62,1.4)],[(1.,64,1.2),(2.75,66,.8)],
            [(.5,62,1.4)],[(.75,61,1.),(2.5,64,.7)]]
    def put(at,midi,hold,velocity):
        # Timing offsets are deterministic, with deliberate phrase rests in the score.
        seconds=at*beat+float(rng.uniform(-.014,.014))
        voice=piano(midi,round(hold*beat,3),round(velocity,3))
        start=round(seconds*RATE)%length
        idx=(start+np.arange(len(voice)))%length
        np.add.at(track,idx,voice)
        score.append(dict(beat=round(at,3),note=midi,hold_beats=hold,velocity=round(velocity,3)))
    for bar,(root,fifth,color) in enumerate(chords):
        pulse=1+.06*np.sin(bar*np.pi/8)
        put(bar*4+.04,root,2.3,.55*pulse)
        put(bar*4+1.35,fifth,1.6,.35*pulse)
        if bar not in (6,7,14,15):put(bar*4+2.75,color,1.1,.28*pulse)
        for at,note,hold in melody[bar]:put(bar*4+at,note,hold,.59*pulse)
    # A small, dark room, folded around the loop so the decay never gets cut off.
    dry=track.copy()
    for seconds,gain in [(.037,.10),(.061,.075),(.113,.052),(.173,.035)]:
        track+=np.roll(dry[:,::-1],round(seconds*RATE),axis=0)*gain
    for seconds,gain in [(.241,.024),(.389,.016),(.577,.009)]:
        track+=np.roll(dry,round(seconds*RATE),axis=0)*gain
    track-=track.mean(axis=0)
    # Match the existing music bus gain; master the composition itself, not game preferences.
    track*=min(.65/np.max(np.abs(track)),.075/np.sqrt(np.mean(track*track)))
    return track,score


def meta(path, music=False):
    f=Path(str(path)+'.meta')
    if f.exists():return
    guid=uuid.uuid5(uuid.NAMESPACE_URL,str(path.relative_to(ROOT))).hex
    if path.is_dir():
        f.write_text(f'fileFormatVersion: 2\nguid: {guid}\nfolderAsset: yes\nDefaultImporter:\n  externalObjects: {{}}\n  userData:\n  assetBundleName:\n  assetBundleVariant:\n')
        return
    template=(AUDIO/('TitleMusic.wav.meta' if music else 'Soft/Move.wav.meta')).read_text()
    import re
    template=re.sub(r'guid: \w+',f'guid: {guid}',template)
    if music:template=template.replace('quality: 0.3','quality: 0.65')
    f.write_text("\n".join(line.rstrip() for line in template.splitlines()) + "\n")


def describe(path):
    x=read(path)
    return dict(seconds=round(len(x)/RATE,6),peak=float(np.max(np.abs(x))),
                rms=float(np.sqrt(np.mean(x*x))),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    STUDY.mkdir(parents=True,exist_ok=True)
    clips=effects()
    for name,x in clips.items():
        write(BANK/(name+'.wav'),x);meta(BANK/(name+'.wav'))
    music,score=title_theme()
    write(MUSIC/'TitleTheme.wav',music);meta(MUSIC);meta(MUSIC/'TitleTheme.wav',True)
    comparisons=[]
    descriptions=[('Move','Move a piece','A small contact and a rounded wooden body, in four takes.'),
        ('AttackLight_0','Light shot','A dry wooden pop.'),('AttackMid_0','Medium shot','A soft, compact knock.'),
        ('AttackHeavy_0','Heavy shot','A deeper wooden thump with a cushioned tail.'),
        ('Deploy','Place from barracks','A firm set-down with a tiny settling contact.'),
        ('Death','Remove a piece','A low contact and a small tumble.'),('EndTurn','Turn change','Two quiet piano notes.')]
    for name,label,description in descriptions:
        write(STUDY/(name+'.wav'),clips[name])
        write(STUDY/(name+'-before.wav'),read(SOURCES/'previous'/(name+'.wav')))
        comparisons.append(dict(name=name,label=label,description=description,before=name+'-before.wav',after=name+'.wav'))
    for name,x in clips.items():write(STUDY/(name+'.wav'),x)
    sequence=[(.45,'Select'),(1.15,'Move'),(2.9,'Move_1'),(4.8,'Place'),(6.8,'AttackLight_0'),
              (9.2,'AttackMid_1'),(11.7,'AttackHeavy_2'),(12.3,'Death'),(15.1,'Deploy'),(17.3,'EndTurn')]
    passage=np.zeros(RATE*20)
    for at,name in sequence:
        start=round(at*RATE);passage[start:start+len(clips[name])]+=clips[name]
    write(STUDY/'sequence.wav',passage)
    variations=np.zeros(RATE*8)
    for i in range(8):
        clip=clips['Move' if i%4==0 else f'Move_{i%4}']
        start=round((.3+i*.9)*RATE);variations[start:start+len(clip)]+=clip
    write(STUDY/'moves.wav',variations)
    # Music comparisons contain the actual default music-bus gain (.24 x .5).
    write(STUDY/'TitleMusic.wav',music*.12)
    write(STUDY/'TitleMusic-before.wav',taper(read(AUDIO/'TitleMusic.wav')[:RATE*30]*.12,.6,.8))
    seam=np.concatenate([music[-RATE*4:],music[:RATE*4]])*.12
    write(STUDY/'music-seam.wav',seam)
    for name,gain in [('AmbientBed',.05),('DesignerHum',.09)]:
        x=read(AUDIO/(name+'.wav'));x=np.tile(x,(int(np.ceil(RATE*12/len(x))),1))[:RATE*12]
        write(STUDY/(name+'.wav'),taper(x*gain,.8,.8))
    report=dict(revision='tabletop-v2',sample_rate=RATE,seed=SEED,previous_commit='cf4adbeda618216ae711cb6e93aedbd9693290e1',
                comparison=comparisons,sequence=[dict(seconds=at,cue=name) for at,name in sequence],
                clips={name:describe(BANK/(name+'.wav')) for name in clips},
                music=describe(MUSIC/'TitleTheme.wav'),music_score=score,music_bpm=62,music_bars=16,
                music_default_gain=.12,music_seam_sample_jump=np.abs(music[0]-music[-1]).tolist())
    report['files']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(STUDY.glob('*.wav'))}
    (STUDY/'palette.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Rendered {len(clips)} effects and {len(music)/RATE:.2f}s of piano; music peak {np.max(np.abs(music)):.3f}.')


if __name__=='__main__':main()
