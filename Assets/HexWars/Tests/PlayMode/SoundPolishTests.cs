using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace HexWars.Presentation.PlayModeTests
{
    public class SoundPolishTests
    {
        readonly string[] _keys = { "HexWars.Volume", "HexWars.EffectsVolume", "HexWars.AtmosphereVolume", "HexWars.MusicVolume" };
        readonly Dictionary<string, float?> _before = new Dictionary<string, float?>();
        bool _hadMute, _wasMuted, _demoMuted; float _listenerVolume;
        GameObject _host; SoundMixDriver _driver; AudioClip _longClip;
        [UnitySetUp]
        public IEnumerator SetUp()
        {
            _listenerVolume=AudioListener.volume; _hadMute=PlayerPrefs.HasKey("HexWars.MuteAll");
            _wasMuted=SoundSettings.MuteAll; _demoMuted=SoundManager.Muted;
            foreach(var key in _keys) _before[key]=PlayerPrefs.HasKey(key)?PlayerPrefs.GetFloat(key):(float?)null;
            SoundSettings.MuteAll=false;SoundSettings.Volume=0;SoundSettings.Effects=1;SoundSettings.Music=.5f;SoundSettings.Atmosphere=.5f;
            _host=new GameObject("Sound mix test");_driver=_host.AddComponent<SoundMixDriver>();
            _longClip=AudioClip.Create("Voice-limit fixture",44100*8,1,44100,false);
            yield return null;
            _driver.SendMessage("OnApplicationFocus",true);
        }
        [UnityTearDown]
        public IEnumerator TearDown()
        {
            Object.Destroy(_host);Object.Destroy(_longClip);
            foreach(var pair in _before)if(pair.Value.HasValue)PlayerPrefs.SetFloat(pair.Key,pair.Value.Value);else PlayerPrefs.DeleteKey(pair.Key);
            if(_hadMute)PlayerPrefs.SetInt("HexWars.MuteAll",_wasMuted?1:0);else PlayerPrefs.DeleteKey("HexWars.MuteAll");
            PlayerPrefs.Save();SoundManager.Muted=_demoMuted;AudioListener.volume=_listenerVolume;
            yield return null;
        }
        void Tick(float dt) => typeof(SoundMixDriver).GetMethod("Tick",BindingFlags.Instance|BindingFlags.NonPublic).Invoke(_driver,new object[]{dt});
        AudioSource Source(string name)=>_host.transform.Find(name).GetComponent<AudioSource>();

        [UnityTest]
        public IEnumerator RepeatedClicksAndFullVoicePoolsStayBounded()
        {
            Assert.That(_driver.Play("select",_longClip,1,.1f,0),Is.True);
            Assert.That(_driver.Play("select",_longClip,1,.1f,0),Is.False,"Same-frame repeat must be silent.");
            for(int i=1;i<6;i++)Assert.That(_driver.Play("shot"+i,_longClip,1,0,2),Is.True);
            Assert.That(_driver.Play("another tick",_longClip,1,0,0),Is.False);
            Assert.That(_driver.Play("conclusion",_longClip,1,0,3),Is.True,"Important feedback can replace a quiet tick.");
            Assert.That(_host.GetComponentsInChildren<AudioSource>().Count(s=>!s.loop&&s.isPlaying),Is.EqualTo(6));
            yield return null;
        }
        [UnityTest]
        public IEnumerator IndependentBusesFadeAndMuteWithoutResettingPreferences()
        {
            _driver.Ambience(true);_driver.Designer(true);Tick(.25f);
            var bed=Source("Ambience");var hum=Source("Workshop");
            Assert.That(bed.clip,Is.Not.Null);Assert.That(hum.clip,Is.Not.Null);
            Assert.That(bed.volume,Is.GreaterThan(0));Assert.That(bed.volume,Is.LessThan(.05f));
            SoundSettings.Atmosphere=0;Tick(3);
            Assert.That(bed.isPlaying,Is.False);Assert.That(hum.isPlaying,Is.False);
            Assert.That(SoundSettings.Effects,Is.EqualTo(1));Assert.That(SoundSettings.Music,Is.EqualTo(.5f));
            SoundSettings.Atmosphere=.5f;_driver.Title(true);Tick(.25f);yield return null;
            Assert.That(Source("Music").clip,Is.Not.Null);
            Assert.That(bed.isPlaying,Is.False,"Opening the title must not restart ambience.");
            SoundSettings.MuteAll=true;Tick(3);
            Assert.That(Source("Music").isPlaying,Is.False);
            Assert.That(SoundSettings.Music,Is.EqualTo(.5f));
            SoundSettings.MuteAll=false;Tick(.1f);
            Assert.That(Source("Music").isPlaying,Is.True);
            Assert.That(Source("Music").volume,Is.LessThan(.12f),"Unmute resumes gently.");
        }
        [UnityTest]
        public IEnumerator FocusLossStopsEffectsAndBedsResumeGently()
        {
            _driver.Ambience(true);Tick(1);
            Assert.That(_driver.Play("move",_longClip,1,0,1),Is.True);
            _driver.SendMessage("OnApplicationFocus",false);
            Assert.That(_host.GetComponentsInChildren<AudioSource>().Where(s=>!s.loop).All(s=>!s.isPlaying),Is.True);
            Assert.That(Source("Ambience").volume,Is.Zero);
            Assert.That(_driver.Play("hidden move",_longClip,1,0,1),Is.False);
            _driver.SendMessage("OnApplicationFocus",true);Tick(.05f);
            Assert.That(Source("Ambience").volume,Is.GreaterThan(0));
            Assert.That(Source("Ambience").volume,Is.LessThan(.05f));
            yield return null;
        }
        [UnityTest]
        public IEnumerator PreparedClipsHaveHeadroomAndAudioDoesNotConsumeGameplayRandomness()
        {
            var clips = Resources.LoadAll<AudioClip>("Audio/Soft");
            Assert.That(clips.Length, Is.EqualTo(22), "The complete mastered palette must be imported.");
            foreach(var clip in clips)
            {
                var samples=new float[clip.samples*clip.channels];Assert.That(clip.GetData(samples,0),Is.True);
                Assert.That(samples.Max(x=>Mathf.Abs(x)),Is.LessThanOrEqualTo(.221f),clip.name);
                Assert.That(Mathf.Abs(samples[0]),Is.LessThan(.0001f),clip.name+" starts smoothly");
                Assert.That(Mathf.Abs(samples[samples.Length-1]),Is.LessThan(.0001f),clip.name+" ends smoothly");
            }
            SoundManager.Muted=false;var before=Random.state;
            SoundManager.PlayAttack(0);SoundManager.PlayAttack(1);SoundManager.PlayAttack(2);
            Assert.That(Random.state,Is.EqualTo(before));
            SoundSettings.Effects=0;Tick(.01f);
            Assert.That(_driver.Play("muted effect",_longClip,1,0,1),Is.False);
            yield return null;
        }
    }
}
