# Generates the share-card / PWA static assets under engine/HexWars.NetServer/wwwroot/:
#   manifest.json, icon-192.png, icon-512.png, favicon.png, preview.png (1200x630 og:image)
# Procedural (no external art asset), on UiKit's palette (Bg #0A0E1C, Accent #45AEFF, CtaGreen #33845C,
# TextDim #9AA3B8). Re-run any time; overwrites in place. Run from repo root or anywhere (path is
# resolved relative to this script's own location).
Add-Type -AssemblyName System.Drawing
$ErrorActionPreference = "Stop"

$dst = Join-Path $PSScriptRoot "HexWars.NetServer\wwwroot"
if (-not (Test-Path $dst)) { New-Item -ItemType Directory -Force -Path $dst | Out-Null }

$bg     = [System.Drawing.Color]::FromArgb(255, 0x0A, 0x0E, 0x1C)
$accent = [System.Drawing.Color]::FromArgb(255, 0x45, 0xAE, 0xFF)
$cta    = [System.Drawing.Color]::FromArgb(255, 0x33, 0x84, 0x5C)
$dim    = [System.Drawing.Color]::FromArgb(255, 0x9A, 0xA3, 0xB8)

function Get-HexPoints([double]$cx, [double]$cy, [double]$r, [double]$rotationDeg = 0) {
    $pts = New-Object System.Collections.Generic.List[System.Drawing.PointF]
    for ($i = 0; $i -lt 6; $i++) {
        $angle = [Math]::PI / 180.0 * (60 * $i + $rotationDeg - 90)
        $x = $cx + $r * [Math]::Cos($angle)
        $y = $cy + $r * [Math]::Sin($angle)
        $pts.Add((New-Object System.Drawing.PointF([float]$x, [float]$y)))
    }
    return $pts.ToArray()
}

function New-Icon([int]$size, [string]$path) {
    $bmp = New-Object System.Drawing.Bitmap($size, $size)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.Clear($bg)

    $cx = $size / 2.0; $cy = $size / 2.0
    $outerR = $size * 0.42
    $innerR = $size * 0.24

    $outerPen = New-Object System.Drawing.Pen($accent, [Math]::Max(2, $size * 0.035))
    $g.DrawPolygon($outerPen, (Get-HexPoints $cx $cy $outerR 0))

    $innerBrush = New-Object System.Drawing.SolidBrush($cta)
    $g.FillPolygon($innerBrush, (Get-HexPoints $cx $cy $innerR 0))

    $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose()
}

New-Icon 192 (Join-Path $dst "icon-192.png")
New-Icon 512 (Join-Path $dst "icon-512.png")
New-Icon 64  (Join-Path $dst "favicon.png")

# preview.png — 1200x630 OpenGraph/Twitter card: wordmark + tagline over a scattered hex-tile motif
$pw = 1200; $ph = 630
$bmp = New-Object System.Drawing.Bitmap($pw, $ph)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$g.Clear($bg)

$motifPen1 = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(40, $accent.R, $accent.G, $accent.B), 2)
$motifPen2 = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(30, $cta.R, $cta.G, $cta.B), 2)
$hr = 46.0
$stepX = $hr * 1.7; $stepY = $hr * 1.5
for ($row = -1; $row -lt ($ph / $stepY) + 2; $row++) {
    for ($col = -1; $col -lt ($pw / $stepX) + 2; $col++) {
        $x = $col * $stepX + (($row % 2) * $stepX * 0.5)
        $y = $row * $stepY
        $pen = if ((($row + $col) % 2) -eq 0) { $motifPen1 } else { $motifPen2 }
        $g.DrawPolygon($pen, (Get-HexPoints $x $y $hr 0))
    }
}

$titleFont = New-Object System.Drawing.Font("Arial", 92, [System.Drawing.FontStyle]::Bold)
$tagFont   = New-Object System.Drawing.Font("Arial", 30, [System.Drawing.FontStyle]::Regular)
$titleBrush = New-Object System.Drawing.SolidBrush($accent)
$tagBrush   = New-Object System.Drawing.SolidBrush($dim)

$g.DrawString("HEXWARS", $titleFont, $titleBrush, 90, 220)
$g.DrawString("hex-grid tactics - design an army, take the field", $tagFont, $tagBrush, 92, 340)

$bmp.Save((Join-Path $dst "preview.png"), [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()

# manifest.json — "Add to Home Screen" / standalone PWA
$manifest = @'
{
  "name": "HexWars",
  "short_name": "HexWars",
  "display": "standalone",
  "start_url": "/",
  "background_color": "#0A0E1C",
  "theme_color": "#0A0E1C",
  "icons": [
    { "src": "/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/icon-512.png", "sizes": "512x512", "type": "image/png" }
  ]
}
'@
Set-Content (Join-Path $dst "manifest.json") -Value $manifest -Encoding utf8 -NoNewline

Write-Host "Generated manifest.json, icon-192.png, icon-512.png, favicon.png, preview.png under $dst"
Get-ChildItem $dst -Filter "*.png" | ForEach-Object { Write-Host "  $($_.Name): $($_.Length) bytes" }
