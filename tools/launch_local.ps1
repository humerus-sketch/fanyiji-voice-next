#requires -Version 5.1
[CmdletBinding()]
param([switch]$CheckOnly, [switch]$ModelValidation, [switch]$Preview, [switch]$PrepareOnly)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($ModelValidation -and $Preview) { throw 'Choose one isolated profile.' }
if ($PrepareOnly -and -not $ModelValidation) { throw 'PrepareOnly requires ModelValidation.' }
$projectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$registration = Join-Path $projectRoot 'local-installation.json'
$state = Get-Content -LiteralPath $registration -Raw -Encoding UTF8 | ConvertFrom-Json
$versionPattern = '^\d+\.\d+\.\d+(?:-[A-Za-z0-9][A-Za-z0-9.-]*)?$'
if ($state.schema_version -ne 1 -or $state.application -ne 'fanyiji-voice-next' -or $state.role -ne 'primary' -or
    $state.current_version -notmatch $versionPattern -or
    $state.program_relative -cne ('dist/'+$state.current_version+'/FanyiJiVoiceNext') -or
    $state.model_relative -cne 'FanyiJiVoiceNext_models') { throw 'Invalid local installation registration.' }
$program = [IO.Path]::GetFullPath((Join-Path $projectRoot $state.program_relative))
$models = [IO.Path]::GetFullPath((Join-Path $projectRoot $state.model_relative))
foreach ($folder in @($projectRoot,$program,$models)) {
    $check = Get-Item -LiteralPath $folder -Force
    if (-not $check.PSIsContainer -or ($check.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw "Unsafe application/model directory: $folder" }
    if ($folder -ne $projectRoot -and -not $folder.StartsWith($projectRoot+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Path outside project.' }
}
$exe = Join-Path $program 'FanyiJiVoiceNext.exe'
$build = Get-Content -LiteralPath (Join-Path $program 'build-info.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not (Test-Path -LiteralPath $exe -PathType Leaf) -or $build.version -ne $state.current_version) { throw 'Program version and directory do not match.' }
$profile = Join-Path $projectRoot 'settings\voice.settings.json'
if ($ModelValidation) { $profile = Join-Path $projectRoot 'validation\model-checks\settings.json' }
if ($Preview) { $profile = Join-Path $projectRoot 'settings\voice.preview.settings.json' }
if (-not $ModelValidation -and -not (Test-Path -LiteralPath $profile -PathType Leaf)) { throw "Settings file missing: $profile" }
if ($CheckOnly) {
    [ordered]@{version=$state.current_version;executable=$exe;model_root=$models;settings_file=$profile;model_validation=[bool]$ModelValidation;preview=[bool]$Preview;process_started=$false} | ConvertTo-Json
    exit
}
if ($ModelValidation -and -not (Test-Path -LiteralPath $profile -PathType Leaf)) {
    # The model check uses the one local program/model set with a separate
    # profile. Neither channel is enabled; starting capture requires the user.
    $qaSettings = Get-Content -LiteralPath (Join-Path $projectRoot 'settings\voice.settings.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $qaSettings.microphone.enabled = $false
    $qaSettings.system.enabled = $false
    New-Item -ItemType Directory -Path (Split-Path -Parent $profile) -Force | Out-Null
    # Windows PowerShell 5.1 Set-Content -Encoding UTF8 adds a BOM, which the
    # strict application settings loader intentionally does not accept.
    [IO.File]::WriteAllText($profile, ($qaSettings | ConvertTo-Json -Depth 15), [Text.UTF8Encoding]::new($false))
}
if (-not (Test-Path -LiteralPath $profile -PathType Leaf)) { throw "Settings file missing: $profile" }
if ($PrepareOnly) {
    [ordered]@{settings_file=$profile;model_root=$models;process_started=$false;profile_prepared=$true} | ConvertTo-Json
    exit
}
$running = @(Get-CimInstance Win32_Process -Filter "Name='FanyiJiVoiceNext.exe'" | Where-Object { $_.ExecutablePath -eq $exe })
if ($running.Count) { throw 'The main program is already open. Close it before starting another profile.' }
$launchArguments = @('--settings-file', ('"'+$profile+'"'), '--require-settings', '--model-root', ('"'+$models+'"'))
Start-Process -FilePath $exe -ArgumentList $launchArguments -WorkingDirectory $program
