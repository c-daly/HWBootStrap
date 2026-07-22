using System;
using System.Linq;
using System.Reflection;
using HexWars.Engine;
using NUnit.Framework;
using UnityEngine;

namespace HexWars.Presentation.Tests
{
    public sealed class OnlineBarracksReconcileTests
    {
        GameObject _gameObject;

        [SetUp]
        public void SetUp()
        {
            SessionBarracksCache.ResetForTests();
            _gameObject = new GameObject("Online barracks test", typeof(BoardRenderer), typeof(TokenStore),
                typeof(GameBootstrap));
        }

        [TearDown]
        public void TearDown()
        {
            SessionBarracksCache.ResetForTests();
            if (_gameObject != null) UnityEngine.Object.DestroyImmediate(_gameObject);
        }

        [Test]
        public void StartRedeal_ReplacesSessionCacheFromAuthoritativeSeatBarracks()
        {
            var stale = SessionBarracksCache.ForLocalPlayer(0);
            stale.Add(new UnitTemplate("Stale local", Stats(4)));
            var authoritative = new UnitTemplate("Server fallback", Stats(9));
            var state = GameFactory.Build(new GameSetup(GameMode.Annihilation, 5, 5, 0, 7),
                new[] { authoritative }, BarracksCatalog.DefaultTemplates);
            var game = _gameObject.GetComponent<GameBootstrap>();
            typeof(TokenStore).GetField("_board", BindingFlags.Instance | BindingFlags.NonPublic)
                .SetValue(_gameObject.GetComponent<TokenStore>(), _gameObject.GetComponent<BoardRenderer>());
            game.Networked = true;
            Invoke(game, "OnNetSeat", PlayerId.Player0);

            Invoke(game, "OnNetStart", ReplayFile.Write(state, Array.Empty<Command>()));

            var cached = SessionBarracksCache.ForLocalPlayer(0).Snapshot();
            Assert.That(cached.Select(template => template.Name), Is.EqualTo(new[] { "Server fallback" }));
            Assert.That(cached[0].Stats.Health, Is.EqualTo(9));
        }

        static UnitStats Stats(int health) => new UnitStats(health, 1, 2, 3, 4, 5, 6, 7, 8);

        static void Invoke(object target, string method, object argument) =>
            target.GetType().GetMethod(method, BindingFlags.Instance | BindingFlags.NonPublic)
                .Invoke(target, new[] { argument });
    }
}
