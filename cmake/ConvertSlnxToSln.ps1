param(
    [Parameter(Mandatory = $true)][string]$InputPath,
    [Parameter(Mandatory = $true)][string]$OutputPath
)

$ErrorActionPreference = 'Stop'

$document = [System.Xml.XmlDocument]::new()
$document.Load((Resolve-Path -LiteralPath $InputPath).Path)
$solution = $document.SelectSingleNode('/Solution')
if ($null -eq $solution) {
    throw "The input file does not contain a Solution root element: $InputPath"
}
$inputDirectory = [System.IO.Path]::GetFullPath((Split-Path -Parent $InputPath))
$outputDirectory = [System.IO.Path]::GetFullPath((Split-Path -Parent $OutputPath))
$outputUri = [System.Uri]::new($outputDirectory.TrimEnd('\') + '\')
$inputUri = [System.Uri]::new($inputDirectory.TrimEnd('\') + '\')
$relativeBuildRoot = [System.Uri]::UnescapeDataString($outputUri.MakeRelativeUri($inputUri).ToString())
$relativeBuildRoot = $relativeBuildRoot.Replace('/', '\').TrimEnd('\')
if ([string]::IsNullOrEmpty($relativeBuildRoot)) {
    $relativeBuildRoot = '.'
}

$buildTypes = @($solution.SelectNodes('./Configurations/BuildType') | ForEach-Object { $_.Name })
$platforms = @($solution.SelectNodes('./Configurations/Platform') | ForEach-Object { $_.Name })
$configurations = @(
    foreach ($buildType in $buildTypes) {
        foreach ($platform in $platforms) {
            "$buildType|$platform"
        }
    }
)

$projects = @($solution.SelectNodes('./Project'))
$projectByPath = @{}
foreach ($project in $projects) {
    $normalizedPath = $project.Path.Replace('/', '\')
    $projectByPath[$normalizedPath.ToLowerInvariant()] = $project
}

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add('Microsoft Visual Studio Solution File, Format Version 12.00')
$lines.Add('# Visual Studio Version 18')
$lines.Add('VisualStudioVersion = 18.0.0.0')
$lines.Add('MinimumVisualStudioVersion = 10.0.40219.1')

foreach ($project in $projects) {
    $projectPath = $project.Path.Replace('/', '\')
    if ($relativeBuildRoot -ne '.') {
        $projectPath = [System.IO.Path]::Combine($relativeBuildRoot, $projectPath)
    }
    $projectPath = $projectPath.Replace('/', '\')
    $projectName = [System.IO.Path]::GetFileNameWithoutExtension($projectPath)
    $projectId = '{' + $project.Id.ToUpperInvariant() + '}'
    $projectType = '{' + $project.Type.ToUpperInvariant() + '}'
    $lines.Add("Project(`"$projectType`") = `"$projectName`", `"$projectPath`", `"$projectId`"")

    $dependencies = @(
        $project.SelectNodes('./BuildDependency') | ForEach-Object {
            $dependencyPath = $_.Project.Replace('/', '\').ToLowerInvariant()
            $projectByPath[$dependencyPath]
        } | Where-Object { $null -ne $_ }
    )
    if ($dependencies.Count -gt 0) {
        $lines.Add("`tProjectSection(ProjectDependencies) = postProject")
        foreach ($dependency in $dependencies) {
            $dependencyId = '{' + $dependency.Id.ToUpperInvariant() + '}'
            $lines.Add("`t`t$dependencyId = $dependencyId")
        }
        $lines.Add("`tEndProjectSection")
    }
    $lines.Add('EndProject')
}

$lines.Add('Global')
$lines.Add("`tGlobalSection(SolutionConfigurationPlatforms) = preSolution")
foreach ($configuration in $configurations) {
    $lines.Add("`t`t$configuration = $configuration")
}
$lines.Add("`tEndGlobalSection")
$lines.Add("`tGlobalSection(ProjectConfigurationPlatforms) = postSolution")
foreach ($project in $projects) {
    $projectId = '{' + $project.Id.ToUpperInvariant() + '}'
    foreach ($configuration in $configurations) {
        $parts = $configuration.Split('|', 2)
        $buildType = $parts[0]
        $excluded = @(
            $project.SelectNodes('./Build') | Where-Object {
                $buildParts = $_.Solution.Split('|', 2)
                $buildParts[0] -eq $buildType -and
                    ($buildParts[1] -eq '*' -or $buildParts[1] -eq $parts[1]) -and
                    $_.Project -eq 'false'
            }
        ).Count -gt 0

        $lines.Add("`t`t$projectId.$configuration.ActiveCfg = $configuration")
        if (-not $excluded) {
            $lines.Add("`t`t$projectId.$configuration.Build.0 = $configuration")
        }
    }
}
$lines.Add("`tEndGlobalSection")
$lines.Add("`tGlobalSection(SolutionProperties) = preSolution")
$lines.Add("`t`tHideSolutionNode = FALSE")
$lines.Add("`tEndGlobalSection")
$lines.Add('EndGlobal')

New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$outputFilePath = [System.IO.Path]::GetFullPath($OutputPath)
$outputEncoding = [System.Text.UTF8Encoding]::new($false)
$outputText = [string]::Join([System.Environment]::NewLine, $lines.ToArray()) + [System.Environment]::NewLine
$shouldWrite = $true
if (Test-Path -LiteralPath $outputFilePath -PathType Leaf) {
    $existingText = [System.IO.File]::ReadAllText($outputFilePath, $outputEncoding)
    $shouldWrite = -not [string]::Equals(
        $existingText,
        $outputText,
        [System.StringComparison]::Ordinal)
}

if ($shouldWrite) {
    [System.IO.File]::WriteAllText($outputFilePath, $outputText, $outputEncoding)
}
