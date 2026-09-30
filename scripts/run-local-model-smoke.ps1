[CmdletBinding()]
param(
  [ValidateSet("Debug", "Release")]
  [string]$Configuration = "Release",
  [ValidateSet("CPU", "GPU", "GPU.0", "GPU.1", "NPU", "AUTO")]
  [string]$Device = "CPU",
  [ValidateRange(1, 4096)]
  [int]$MaxTokens = 12,
  [string]$ModelPath = "D:\Model\gpt-j-6b-int4-ov\openvino_model.xml",
  [string]$Prompt = "What is OpenVINO?"
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$preset = "windows-msvc-$($Configuration.ToLowerInvariant())"
$application = Join-Path $repositoryRoot "out\build\$preset\$Configuration\Isvik.exe"

if (-not (Test-Path -LiteralPath $application -PathType Leaf)) {
  Write-Error "Isvik.exe was not found at '$application'. Build it with: cmake --build --preset $preset"
  exit 1
}

if (-not (Test-Path -LiteralPath $ModelPath)) {
  Write-Error "Model directory/file was not found at '$ModelPath'. Pass -ModelPath with a tokenizer-equipped OpenVINO IR model."
  exit 1
}

$arguments = @(
  "--run",
  "--model", $ModelPath,
  "--device", $Device,
  "--max-tokens", [string]$MaxTokens,
  "--greedy",
  "--prompt", $Prompt
)

& $application @arguments
exit $LASTEXITCODE
