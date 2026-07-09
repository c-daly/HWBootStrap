# Stages a freshly built WebGL client for deployment: copies Build/WebGL (Unity's output — see
# Assets/HexWars/Editor/WebGLBuild.cs) into engine/HexWars.NetServer/wwwroot, which is the folder
# the Docker image actually serves. Committing the build anywhere else deploys nothing.
# After running: git add + commit, then push (from WSL — the SSH key lives there).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot               # repo root (this script lives in engine/)
$src = Join-Path $root "Build\WebGL"
$dst = Join-Path $root "engine\HexWars.NetServer\wwwroot"

if (-not (Test-Path (Join-Path $src "Build"))) { throw "No build at $src - run HexWars > Build WebGL first." }

if (Test-Path (Join-Path $dst "Build")) { Remove-Item (Join-Path $dst "Build") -Recurse -Force }
Copy-Item (Join-Path $src "Build") (Join-Path $dst "Build") -Recurse
Copy-Item (Join-Path $src "index.html") $dst -Force
if (Test-Path (Join-Path $src "StreamingAssets")) {
    Copy-Item (Join-Path $src "StreamingAssets\*") (Join-Path $dst "StreamingAssets\") -Recurse -Force
}

# Cache-bust the payload URLs: builds ship under the SAME filenames, and Unity's IndexedDB cache
# keys by URL — after a redeploy a browser can pair an old cached .data with the new .wasm and die
# at boot ("RuntimeError: memory access out of bounds" in callMain). A per-deploy ?v= makes every
# build's URLs unique so old and new can never mix.
$v = (Get-FileHash (Join-Path $dst "Build\WebGL.data.unityweb") -Algorithm SHA256).Hash.Substring(0, 8).ToLower()
$idx = Join-Path $dst "index.html"
$html = (Get-Content $idx -Raw) -replace '(Build/WebGL\.(?:data\.unityweb|framework\.js\.unityweb|wasm\.unityweb|loader\.js))', ('$1?v=' + $v)

# Share-card + PWA tags: the Unity template owns index.html, so staging is the one place post-build
# HTML edits happen — this extends the same rewrite step above rather than adding a second pass.
# Statics (manifest.json, icon-*.png, favicon.png, preview.png) live under wwwroot/ directly, generated
# by generate-share-assets.ps1; this script never deletes wwwroot root-level files (only wwwroot/Build/
# is removed+recreated, above), so they survive every redeploy without being re-copied here.
# Guarded on og:title so re-running this against an already-injected file — shouldn't happen, since
# index.html is always a fresh copy from $src above, but is cheap insurance — never double-inserts.
if ($html -notmatch 'og:title') {
    $headInject = @"
    <meta property="og:title" content="HexWars &mdash; hex-grid tactics" />
    <meta property="og:description" content="Design your army from raw points. Outbuild, outthink, dominate." />
    <meta property="og:image" content="https://hwbootstrap.onrender.com/preview.png" />
    <meta property="og:type" content="website" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content="HexWars &mdash; hex-grid tactics" />
    <meta name="twitter:description" content="Design your army from raw points. Outbuild, outthink, dominate." />
    <meta name="twitter:image" content="https://hwbootstrap.onrender.com/preview.png" />
    <link rel="manifest" href="/manifest.json" />
    <meta name="theme-color" content="#0A0E1C" />
    <link rel="apple-touch-icon" href="/icon-192.png" />
    <link rel="icon" type="image/png" href="/favicon.png" />
</head>
"@
    $html = $html -replace '</head>', $headInject
}

Set-Content $idx -Value $html -Encoding utf8 -NoNewline

Write-Host "Staged $src -> $dst with cache-bust v=$v  (now: git add/commit, then push from WSL)"
