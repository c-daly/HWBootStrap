using System;
using UnityEngine;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>
    /// The verbose rules reference, opened from the lobby ("Rules") and in-game ("?"). One source of truth:
    /// a scrollable card built in code (touch-drag or wheel to scroll) over a dim backdrop; tap outside or
    /// "Close" to dismiss. <see cref="Show"/> builds a fresh instance under the given canvas.
    /// </summary>
    public static class GameRules
    {
        public static void Show(Transform canvasParent, Font font, int sortingOrder)
        {
            var canvasGo = new GameObject("RulesCanvas");
            canvasGo.transform.SetParent(canvasParent, false);
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = sortingOrder;
            var scaler = canvasGo.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1200f, 800f);
            scaler.matchWidthOrHeight = 0.5f;
            canvasGo.AddComponent<GraphicRaycaster>();

            // dim backdrop — tap outside the card to close
            var dim = Panel(canvasGo.transform, new Color(0.02f, 0.03f, 0.06f, 0.86f));
            Stretch(dim);
            dim.gameObject.AddComponent<Button>().onClick.AddListener(() => UnityEngine.Object.Destroy(canvasGo));

            // card
            var card = Panel(canvasGo.transform, new Color(0.10f, 0.12f, 0.18f, 1f));
            card.anchorMin = card.anchorMax = new Vector2(0.5f, 0.5f);
            card.pivot = new Vector2(0.5f, 0.5f);
            card.sizeDelta = new Vector2(760f, 720f);
            card.anchoredPosition = Vector2.zero;

            Label(card, "HexWars — Rules", font, 26, TextAnchor.UpperCenter,
                  new Vector2(0f, -18f), new Vector2(760f, 36f));

            // scroll viewport (clips the content)
            var viewportGo = new GameObject("Viewport");
            viewportGo.transform.SetParent(card.transform, false);
            var vpImg = viewportGo.AddComponent<Image>();
            vpImg.color = new Color(1f, 1f, 1f, 0.02f);
            viewportGo.AddComponent<RectMask2D>();
            var vp = viewportGo.GetComponent<RectTransform>();
            vp.anchorMin = new Vector2(0f, 0f); vp.anchorMax = new Vector2(1f, 1f);
            vp.offsetMin = new Vector2(26f, 78f);  // leave room for the Close button
            vp.offsetMax = new Vector2(-26f, -64f); // and the title

            // scrolling text content
            var contentGo = new GameObject("Content");
            contentGo.transform.SetParent(viewportGo.transform, false);
            var text = contentGo.AddComponent<Text>();
            text.font = font; text.fontSize = 18; text.color = new Color(0.92f, 0.94f, 0.97f);
            text.alignment = TextAnchor.UpperLeft; text.raycastTarget = false;
            text.horizontalOverflow = HorizontalWrapMode.Wrap; text.verticalOverflow = VerticalWrapMode.Overflow;
            text.text = RulesText;
            var ct = contentGo.GetComponent<RectTransform>();
            ct.anchorMin = new Vector2(0f, 1f); ct.anchorMax = new Vector2(1f, 1f);
            ct.pivot = new Vector2(0.5f, 1f);
            ct.offsetMin = new Vector2(6f, 0f); ct.offsetMax = new Vector2(-6f, 0f);
            contentGo.AddComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;

            var scroll = viewportGo.AddComponent<ScrollRect>();
            scroll.horizontal = false; scroll.vertical = true; scroll.scrollSensitivity = 24f;
            scroll.viewport = vp; scroll.content = ct; scroll.movementType = ScrollRect.MovementType.Clamped;

            // close button
            var close = Panel(card.transform, new Color(0.22f, 0.36f, 0.58f, 1f));
            close.anchorMin = close.anchorMax = new Vector2(0.5f, 0f);
            close.pivot = new Vector2(0.5f, 0f);
            close.sizeDelta = new Vector2(200f, 46f);
            close.anchoredPosition = new Vector2(0f, 16f);
            close.gameObject.AddComponent<Button>().onClick.AddListener(() => UnityEngine.Object.Destroy(canvasGo));
            Label(close, "Close", font, 20, TextAnchor.MiddleCenter, Vector2.zero, new Vector2(200f, 46f));
        }

        static RectTransform Panel(Transform parent, Color color)
        {
            var go = new GameObject("Panel");
            go.transform.SetParent(parent, false);
            go.AddComponent<Image>().color = color;
            return go.GetComponent<RectTransform>();
        }

        static void Stretch(RectTransform rt)
        {
            rt.anchorMin = Vector2.zero; rt.anchorMax = Vector2.one; rt.offsetMin = Vector2.zero; rt.offsetMax = Vector2.zero;
        }

        static void Label(RectTransform parent, string s, Font font, int size, TextAnchor anchor, Vector2 pos, Vector2 sz)
        {
            var go = new GameObject("Label");
            go.transform.SetParent(parent, false);
            var t = go.AddComponent<Text>();
            t.font = font; t.fontSize = size; t.color = Color.white; t.alignment = anchor; t.text = s;
            t.raycastTarget = false;
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = rt.anchorMax = new Vector2(0.5f, 1f); rt.pivot = new Vector2(0.5f, 1f);
            rt.sizeDelta = sz; rt.anchoredPosition = pos;
        }

        const string RulesText =
@"GOAL
Wipe out the enemy army (Annihilation). If the round cap is reached, the higher score wins.

MODES
  Annihilation — a straight fight with the army you start with.
  Territory — the same fight, plus an economy: hold ground, build generators for income, and spend
    points on more units. The economy funds your war; it does not win on its own.

UNITS
Each unit has: Health, Damage, Defense, Move, Vertical (climb), Range, and Vision.
  Brute    — tough melee (high HP/defense, range 1).
  Striker  — fast glass cannon (high damage, low HP).
  Sniper   — fragile but long range.
Hover (or touch) a unit in-game to see its full stats.

MOVING
A unit has two separate budgets per turn:
  Move — horizontal steps (1 per hex).
  Vertical — how many elevation levels it can climb (descending and level moves are free).
So a steep cliff needs enough Vertical, and a long dash needs enough Move.

COMBAT
Damage dealt = attacker Damage − target Defense (always at least 1 if it lands).
  High ground — attacking from higher elevation adds damage and reach.
  Range & Vision — you can only fire at targets in range that you can see (line of sight).

TERRITORY & ECONOMY (Territory mode)
  Control — you start owning your home rows (tinted your colour). Generators and building need control.
  Income — comes ONLY from generators you build on hexes you control. Bare territory pays nothing.
  Build — select a unit, tap ""Build generators"", then tap any hex you control to place one.
  Claim — select a unit standing on an UNOWNED hex; the action button claims it. Claiming costs points,
    must be your turn's first action, and ends your turn — so expanding leaves you exposed.
  Points — fund claiming, building, and deploying more units. (Decay is off; you won't lose banked points.)

PACE (host setting)
How many actions you take before the turn passes:
  Whole army — act with everything, then pass. Snappy, but going second is a real advantage.
  K per turn (3 is the default) — commit K actions, then your opponent responds. Fairer and more
    tactical; also makes the claim-ends-turn tax proportionate.

CONTROLS
  Phone — drag to pan, pinch to zoom. Tap a unit to select; tap a hex to move; tap an enemy to attack.
  Desktop — WASD / arrows to pan, scroll to zoom; click to select / move / attack.
  End your turn with the End Turn button (bottom-left).";
    }
}
