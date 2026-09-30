[CmdletBinding()]
param(
  [string]$Source = "D:\php\Business-card\assets\logo-mark-light.png",
  [string]$LockupSource = "D:\php\Business-card\assets\logo-lockup.png",
  [string]$Destination = (Join-Path (Split-Path -Parent $PSScriptRoot) "resources")
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
New-Item -ItemType Directory -Path $Destination -Force | Out-Null
$original = [System.Drawing.Bitmap]::new($Source)
$lockup = [System.Drawing.Bitmap]::new($LockupSource)
$frames = [System.Collections.Generic.List[object]]::new()
try {
  $blackMark = [System.Drawing.Bitmap]::new($original.Width, $original.Height, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
  $blackLockup = [System.Drawing.Bitmap]::new($lockup.Width, $lockup.Height, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
  try {
    foreach ($sourceBitmap in @(@($original, $blackMark), @($lockup, $blackLockup))) {
      $input = $sourceBitmap[0]
      $output = $sourceBitmap[1]
      for ($y = 0; $y -lt $input.Height; $y++) {
        for ($x = 0; $x -lt $input.Width; $x++) {
          $alpha = $input.GetPixel($x, $y).A
          if ($alpha -ne 0) { $output.SetPixel($x, $y, [System.Drawing.Color]::FromArgb($alpha, 20, 24, 29)) }
        }
      }
    }
    $blackMark.Save((Join-Path $Destination "isvik-mark-black.png"), [System.Drawing.Imaging.ImageFormat]::Png)
    $blackLockup.Save((Join-Path $Destination "isvik-lockup-black.png"), [System.Drawing.Imaging.ImageFormat]::Png)
  foreach ($iconSize in @(16, 20, 24, 28, 32, 40, 48, 64, 128, 256)) {
    $bitmap = [System.Drawing.Bitmap]::new($iconSize, $iconSize, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $stream = [System.IO.MemoryStream]::new()
    try {
      $graphics.Clear([System.Drawing.Color]::Transparent)
      $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
      $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
      $scale = [single]($iconSize * 0.78 / [Math]::Max($blackMark.Width, $blackMark.Height))
      $drawWidth = [single]($blackMark.Width * $scale)
      $drawHeight = [single]($blackMark.Height * $scale)
      $rectangle = [System.Drawing.RectangleF]::new(($iconSize - $drawWidth) / 2, ($iconSize - $drawHeight) / 2, $drawWidth, $drawHeight)
      $graphics.DrawImage($blackMark, $rectangle)
      $bitmap.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png)
      $frames.Add([pscustomobject]@{ Size = $iconSize; Bytes = $stream.ToArray() })
      if ($iconSize -eq 256) { $bitmap.Save((Join-Path $Destination "isvik-icon.png"), [System.Drawing.Imaging.ImageFormat]::Png) }
    } finally {
      $stream.Dispose(); $graphics.Dispose(); $bitmap.Dispose()
    }
  }
  $output = [System.IO.File]::Create((Join-Path $Destination "Isvik.ico"))
  $writer = [System.IO.BinaryWriter]::new($output)
  try {
    $writer.Write([uint16]0); $writer.Write([uint16]1); $writer.Write([uint16]$frames.Count)
    $offset = [uint32](6 + 16 * $frames.Count)
    foreach ($frame in $frames) {
      $dimension = if ($frame.Size -eq 256) { 0 } else { $frame.Size }
      $writer.Write([byte]$dimension); $writer.Write([byte]$dimension)
      $writer.Write([byte]0); $writer.Write([byte]0)
      $writer.Write([uint16]1); $writer.Write([uint16]32)
      $writer.Write([uint32]$frame.Bytes.Length); $writer.Write($offset)
      $offset += [uint32]$frame.Bytes.Length
    }
    foreach ($frame in $frames) { $writer.Write([byte[]]$frame.Bytes) }
  } finally { $writer.Dispose() }
  } finally { $blackMark.Dispose(); $blackLockup.Dispose() }
} finally { $original.Dispose(); $lockup.Dispose() }
Write-Output "Generated black transparent Isvik icon and lockup with $($frames.Count) ICO resolutions."
