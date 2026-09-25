using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using HexWars.Engine;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace HexWars.Presentation.PlayModeTests
{
    public class SoundPolishTests
    {
        readonly string[] _keys = { "HexWars.Volume", "HexWars.EffectsVolume", "HexWars.AtmosphereVolume", "HexWars.MusicVolume" };
        readonly Dictionary<string, float?> _before = new Dictionary<string, float?>();
        bool _hadMute, _wasMuted, _demoMuted, _hadGameMusic, _gameMusic; float _listenerVolume;
        GameObject _host; SoundMixDriver _driver; AudioClip _longClip;
        SoundMixDriver _previousDriver;
        static readonly FieldInfo DriverField = typeof(SoundManager).GetField("_driver", BindingFlags.Static | BindingFlags.NonPublic);
        [UnitySetUp]
        public IEnumerator SetUp()
        {
            _listenerVolume=AudioListener.volume; _hadMute=PlayerPrefs.HasKey("HexWars.MuteAll");
            _wasMuted=SoundSettings.MuteAll; _demoMuted=SoundManager.Muted;
            _hadGameMusic=PlayerPrefs.HasKey("HexWars.MusicDuringGame"); _gameMusic=SoundSettings.MusicDuringGame;
            PlayerPrefs.DeleteKey("HexWars.MusicDuringGame");
            foreach(var key in _keys) _before[key]=PlayerPrefs.HasKey(key)?PlayerPrefs.GetFloat(key):(float?)null;
            SoundSettings.MuteAll=false;SoundSettings.Volume=0;SoundSettings.Effects=1;SoundSettings.Music=.5f;SoundSettings.Atmosphere=.5f;
            _host=new GameObject("Sound mix test");_driver=_host.AddComponent<SoundMixDriver>();
            _previousDriver = (SoundMixDriver)DriverField.GetValue(null); DriverField.SetValue(null, _driver);
            _longClip=AudioClip.Create("Voice-limit fixture",44100*8,1,44100,false);
            yield return null;
            _driver.SendMessage("OnApplicationFocus",true);
        }
        [UnityTearDown]
        public IEnumerator TearDown()
        {
            DriverField.SetValue(null, _previousDriver);
            Object.Destroy(_host);Object.Destroy(_longClip);
            foreach(var pair in _before)if(pair.Value.HasValue)PlayerPrefs.SetFloat(pair.Key,pair.Value.Value);else PlayerPrefs.DeleteKey(pair.Key);
            if(_hadMute)PlayerPrefs.SetInt("HexWars.MuteAll",_wasMuted?1:0);else PlayerPrefs.DeleteKey("HexWars.MuteAll");
            if(_hadGameMusic)PlayerPrefs.SetInt("HexWars.MusicDuringGame",_gameMusic?1:0);else PlayerPrefs.DeleteKey("HexWars.MusicDuringGame");
            PlayerPrefs.Save();SoundManager.Muted=_demoMuted;AudioListener.volume=_listenerVolume;
            yield return null;
        }
        void Tick(float dt) => typeof(SoundMixDriver).GetMethod("Tick",BindingFlags.Instance|BindingFlags.NonPublic).Invoke(_driver,new object[]{dt});
        AudioSource Source(string name)=>_host.transform.Find(name).GetComponent<AudioSource>();

        [UnityTest]
        public IEnumerator MusicContinuesAtTheSamePositionAcrossTitleAndMatchWhenEnabled()
        {
            SoundSettings.MusicDuringGame = true;
            SoundManager.StartTitleMusic(); Tick(1); yield return null;
            var music = Source("Music");
            Assert.That(music.isPlaying, Is.True);
            var clip = music.clip;
            music.time = 8f;
            SoundManager.StopTitleMusic(); SoundManager.StartAmbience(); Tick(1);
            Assert.That(music.isPlaying, Is.True);
            Assert.That(music.clip, Is.SameAs(clip));
            Assert.That(music.time, Is.GreaterThanOrEqualTo(7.9f), "Entering a match must not restart the theme.");
            Assert.That(Source("Ambience").isPlaying, Is.True);
            SoundManager.StartTitleMusic(); SoundManager.StopAmbience(); Tick(3);
            Assert.That(music.time, Is.GreaterThanOrEqualTo(7.9f), "Returning to the title must keep the playhead too.");
            Assert.That(Source("Ambience").isPlaying, Is.False);
        }

        [UnityTest]
        public IEnumerator GameMusicIsOptionalAndChangesLiveWithoutChangingOtherBuses()
        {
            Assert.That(SoundSettings.MusicDuringGame, Is.False);
            SoundManager.StartTitleMusic(); Tick(1); yield return null;
            Assert.That(Source("Music").isPlaying, Is.True, "The existing title default is retained.");
            SoundManager.StopTitleMusic(); SoundManager.StartAmbience(); Tick(3);
            Assert.That(Source("Music").isPlaying, Is.False, "Music stays title-only until enabled.");
            SoundSettings.MusicDuringGame = true; Tick(.25f);
            Assert.That(Source("Music").isPlaying, Is.True);
            Assert.That(Source("Music").volume, Is.GreaterThan(0));
            SoundSettings.Music = 0; Tick(3);
            Assert.That(Source("Music").isPlaying, Is.False);
            Assert.That(Source("Ambience").isPlaying, Is.True);
            Assert.That(_driver.Play("move", _longClip, 1, 0, 1), Is.True);
            SoundSettings.Music = .5f; Tick(.25f);
            SoundSettings.MuteAll = true; Tick(3);
            Assert.That(Source("Music").isPlaying, Is.False);
            Assert.That(SoundSettings.MusicDuringGame, Is.True);
            SoundSettings.MuteAll = false; Tick(.25f);
            Assert.That(Source("Music").isPlaying, Is.True);
            SoundSettings.MusicDuringGame = false; Tick(3);
            Assert.That(Source("Music").isPlaying, Is.False);
            Assert.That(Source("Ambience").isPlaying, Is.True);
            Assert.That(SoundSettings.Music, Is.EqualTo(.5f));
            Assert.That(SoundSettings.Effects, Is.EqualTo(1));
        }

        [UnityTest]
        public IEnumerator MovementUsesDifferentRecordedTakesWithoutChangingGameplayRandomness()
        {
            foreach (var name in new[] { "Move", "Move_1", "Move_2", "Move_3" })
                Assert.That(Resources.Load<AudioClip>("Audio/Soft/" + name), Is.Not.Null,
                    "The recorded movement take must be present: " + name);
            SoundManager.Muted = false;
            string previous = null;
            var heard = new HashSet<string>();
            for (int i = 0; i < 12; i++)
            {
                foreach (var source in _host.GetComponentsInChildren<AudioSource>()) source.Stop();
                var randomBefore = Random.state;
                SoundManager.Play(SoundKind.Move);
                var voice = _host.GetComponentsInChildren<AudioSource>().Single(s => !s.loop && s.isPlaying);
                Assert.That(voice.clip.name, Does.StartWith("Move"));
                Assert.That(voice.clip.name, Is.Not.EqualTo(previous), "Consecutive moves must use different takes.");
                heard.Add(voice.clip.name); previous = voice.clip.name;
                SoundManager.Play(SoundKind.Move);
                Assert.That(_host.GetComponentsInChildren<AudioSource>().Count(s => !s.loop && s.isPlaying), Is.EqualTo(1),
                    "A same-frame repeated call must still respect the shared movement cooldown.");
                Assert.That(Random.state, Is.EqualTo(randomBefore));
                yield return new WaitForSecondsRealtime(.09f);
            }
            Assert.That(heard.Count, Is.GreaterThan(1));
        }

        [UnityTest]
        public IEnumerator FastForwardKeepsOneConclusionAcrossQueuedDeaths()
        {
            yield return CheckQueuedResolution(true, SoundKind.Win);
        }

        [UnityTest]
        public IEnumerator FastForwardKeepsOneDeathCueWithoutStackingTheQueue()
        {
            yield return CheckQueuedResolution(false, SoundKind.Death);
        }

        IEnumerator CheckQueuedResolution(bool decisive, SoundKind expected)
        {
            bool reduced = MotionSettings.Reduced;
            bool hadReduced = PlayerPrefs.HasKey("HexWars.ReducedMotion");
            try
            {
                MotionSettings.Reduced = false; SoundManager.Muted = false;
                var host = new GameObject("Queued battle", typeof(BoardRenderer), typeof(TokenStore), typeof(GameBootstrap));
                host.transform.SetParent(_host.transform);
                var game = host.GetComponent<GameBootstrap>(); game.enabled = false;
                var presenter = host.AddComponent<ActionPresenter>();
                var tiles = Enumerable.Range(0, 5).SelectMany(q => Enumerable.Range(0, 3)
                    .Select(r => new Tile(new HexCoord(q, r), 0, TerrainType.Plains))).ToArray();
                var victims = new List<Unit> {
                    new Unit(10, PlayerId.Player0, new UnitStats(2,1,0,1,1,1,1,2,1), new HexCoord(0,1), 0),
                    new Unit(11, PlayerId.Player0, new UnitStats(2,1,0,1,1,1,1,2,1), new HexCoord(0,2), 0)
                };
                if (!decisive) victims.Add(new Unit(12, PlayerId.Player0,
                    new UnitStats(2,1,0,1,1,1,1,2,1), new HexCoord(0,0), 0));
                var attackers = new[] {
                    new Unit(20, PlayerId.Player1, new UnitStats(4,6,0,3,1,5,1,6,1), new HexCoord(3,1), 0),
                    new Unit(21, PlayerId.Player1, new UnitStats(4,6,0,3,1,5,1,6,1), new HexCoord(3,2), 0)
                };
                var state = new GameState(new Board(tiles), GameConfig.Default(), new[] {
                    new PlayerState(PlayerId.Player0, 0, unitsOnBoard: victims),
                    new PlayerState(PlayerId.Player1, 0, unitsOnBoard: attackers)
                }, PlayerId.Player1, 2, 22); // Annihilation activates from round two.
                typeof(GameBootstrap).GetProperty("State").SetValue(game, state);
                host.GetComponent<BoardRenderer>().Render(state.Board);
                host.GetComponent<BoardRenderer>().RenderEntities(state);
                int committed = 0; GameState presented = null;
                presenter.ItemCommitted += (prev, cmd, next) => { committed++; presented = next; };
                foreach (var command in new Command[] {
                    new MoveUnit(PlayerId.Player1, 20, new HexCoord(2,1)),
                    new AttackUnit(PlayerId.Player1, 20, 10),
                    new AttackUnit(PlayerId.Player1, 21, 11)
                })
                {
                    var applied = GameEngine.Apply(state, command);
                    Assert.That(applied.Success, Is.True, command.ToString());
                    presenter.Enqueue(state, command, applied.NewState, false);
                    state = applied.NewState;
                }
                Assert.That(presenter.IsBusy, Is.True, "The opponent's attacks must wait behind the current move.");
                Assert.That(state.IsGameOver, Is.EqualTo(decisive));
                // Observe only resolution audio; the already-started move remains ordinary feedback.
                foreach (var source in _host.GetComponentsInChildren<AudioSource>()) source.Stop();
                MotionSettings.Reduced = true;
                presenter.FastForward();
                Assert.That(presenter.IsBusy, Is.False);
                Assert.That(committed, Is.EqualTo(3)); Assert.That(presented, Is.SameAs(state));
                var audible = _host.GetComponentsInChildren<AudioSource>().Where(s => !s.loop && s.isPlaying).ToArray();
                Assert.That(audible.Length, Is.EqualTo(1), "The whole skipped batch gets one important result cue.");
                Assert.That(audible[0].clip.name, Is.EqualTo(expected.ToString()));
                presenter.FastForward(); Assert.That(committed, Is.EqualTo(3), "Repeated fast-forward cannot recommit the result.");
                yield return null;
            }
            finally
            {
                if (hadReduced) MotionSettings.Reduced = reduced;
                else { PlayerPrefs.DeleteKey("HexWars.ReducedMotion"); PlayerPrefs.Save(); }
            }
        }

        [UnityTest]
        public IEnumerator HiddenQueuedMoveStaysSilentWhenItAutomaticallyEndsTheTurn()
        {
            yield return CheckQueuedAutomaticTurn(true);
        }

        [UnityTest]
        public IEnumerator VisibleQueuedMoveKeepsItsAutomaticTurnCue()
        {
            yield return CheckQueuedAutomaticTurn(false);
        }

        IEnumerator CheckQueuedAutomaticTurn(bool hidden)
        {
            bool reduced = MotionSettings.Reduced;
            bool hadReduced = PlayerPrefs.HasKey("HexWars.ReducedMotion");
            try
            {
                MotionSettings.Reduced = false;
                var host = new GameObject("Queued fog turn", typeof(BoardRenderer), typeof(TokenStore), typeof(GameBootstrap));
                host.transform.SetParent(_host.transform);
                var game = host.GetComponent<GameBootstrap>(); game.enabled = false;
                typeof(GameBootstrap).GetProperty("Seat").SetValue(game, PlayerId.Player0);
                var presenter = host.AddComponent<ActionPresenter>();
                var tiles = Enumerable.Range(0, 10).SelectMany(q => Enumerable.Range(0, 3)
                    .Select(r => new Tile(new HexCoord(q, r), 0, TerrainType.Plains))).ToArray();
                var stats = new UnitStats(4,1,0,1,1,1,1,1,1);
                var state = new GameState(new Board(tiles),
                    GameConfig.Default(turnPolicy: new OneActionPolicy(), fogOfWar: hidden), new[] {
                        new PlayerState(PlayerId.Player0, 0, unitsOnBoard: new[] { new Unit(10, PlayerId.Player0, stats, new HexCoord(0,1), 0) }),
                        new PlayerState(PlayerId.Player1, 0, unitsOnBoard: new[] { new Unit(20, PlayerId.Player1, stats, new HexCoord(8,1), 0) })
                    }, PlayerId.Player0, 2, 21);
                typeof(GameBootstrap).GetProperty("State").SetValue(game, state);
                host.GetComponent<BoardRenderer>().Render(state.Board);
                host.GetComponent<BoardRenderer>().RenderEntities(state);
                int committed = 0; GameState presented = null;
                presenter.ItemCommitted += (prev, cmd, next) => { committed++; presented = next; };
                // Mute the current explicit turn cue without entering its cooldown. This isolates the
                // queued move's resolution and ensures the visible positive control can still sound.
                SoundManager.Muted = true;
                var end = new EndTurn(PlayerId.Player0);
                var afterEnd = GameEngine.Apply(state, end);
                Assert.That(afterEnd.Success, Is.True);
                presenter.Enqueue(state, end, afterEnd.NewState, false);
                var move = new MoveUnit(PlayerId.Player1, 20, new HexCoord(7,1));
                var afterMove = GameEngine.Apply(afterEnd.NewState, move);
                Assert.That(afterMove.Success, Is.True);
                Assert.That(afterMove.NewState.ActivePlayer, Is.EqualTo(PlayerId.Player0));
                var span = FogPresentation.VisibleSpan(afterMove.NewState, game.FogViewerFor(afterMove.NewState),
                    new[] { new HexCoord(8,1), new HexCoord(7,1) });
                Assert.That(span.First < 0, Is.EqualTo(hidden), "Control the queued move's actual visibility.");
                presenter.Enqueue(afterEnd.NewState, move, afterMove.NewState, false);
                Assert.That(presenter.IsBusy, Is.True);
                SoundManager.Muted = false;
                MotionSettings.Reduced = true;
                presenter.FastForward();
                Assert.That(presenter.IsBusy, Is.False);
                Assert.That(committed, Is.EqualTo(2)); Assert.That(presented, Is.SameAs(afterMove.NewState));
                var audible = _host.GetComponentsInChildren<AudioSource>().Where(s => !s.loop && s.isPlaying).ToArray();
                Assert.That(audible.Length, Is.EqualTo(hidden ? 0 : 1));
                if (!hidden) Assert.That(audible[0].clip.name, Is.EqualTo(SoundKind.EndTurn.ToString()));
                yield return null;
            }
            finally
            {
                if (hadReduced) MotionSettings.Reduced = reduced;
                else { PlayerPrefs.DeleteKey("HexWars.ReducedMotion"); PlayerPrefs.Save(); }
            }
        }

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
            Assert.That(Source("Music").clip.name,Is.EqualTo("TitleTheme"),"Title playback must use the new piano arrangement.");
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
            Assert.That(clips.Length, Is.EqualTo(25), "The complete palette, including four movement takes, must be imported.");
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
