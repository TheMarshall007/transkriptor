param(
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not [System.IO.Path]::IsPathRooted($Python)) {
    $Python = Join-Path $projectRoot $Python
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python não encontrado em '$Python'. Crie .venv e instale requirements-build.txt."
}

Push-Location $projectRoot
try {
    & $Python -m PyInstaller --clean --noconfirm Transkriptor.spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller terminou com código $LASTEXITCODE."
    }
    $executable = Join-Path $projectRoot "dist\Transkriptor\Transkriptor.exe"
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "Build terminou sem gerar '$executable'."
    }
    Write-Host "Executável criado: $executable"
} finally {
    Pop-Location
}
