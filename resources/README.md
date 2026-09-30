# Application resources

This directory contains the Isvik.cpp application icon, version resource, and related artwork.

The application embeds its icon in the Windows executable and uses matching PNG artwork in the desktop interface. Regenerate the Windows icon after updating the source mark:

~~~powershell
.\scripts\generate-icon.ps1 -Source "path\to\logo-mark.png"
~~~

`Isvik.rc` embeds the icon and version metadata. `resource.h` defines the resource identifiers. CMake enables the Windows resource compiler and tracks the icon files as build dependencies.

The project artwork is distributed under the project license unless a separate notice accompanies the asset.
