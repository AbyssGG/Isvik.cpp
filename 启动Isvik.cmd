@echo off
set "ISVIK_APP=%~dp0out\build\windows-msvc-release\Release\Isvik.exe"
if exist "%~dp0out\install-release\bin\Isvik.exe" set "ISVIK_APP=%~dp0out\install-release\bin\Isvik.exe"
if not exist "%ISVIK_APP%" (
  echo Build Release first: cmake --build --preset windows-msvc-release
  pause
  exit /b 1
)
echo Starting Isvik with console logs. Close the app to return here.
"%ISVIK_APP%" %*
set "ISVIK_EXIT_CODE=%ERRORLEVEL%"
echo Isvik exited with code %ISVIK_EXIT_CODE%.
pause
exit /b %ISVIK_EXIT_CODE%
