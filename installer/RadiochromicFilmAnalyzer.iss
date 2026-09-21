; Build with:
;   iscc /DAppVersion=<version> installer\RadiochromicFilmAnalyzer.iss

#ifndef AppVersion
  #error AppVersion must be supplied with /DAppVersion=<version>
#endif

#define AppName "Radiochromic Film Analyzer"
#define AppPublisher "Pablo de la Fuente Fernández"
#define AppExeName "RadiochromicFilmAnalyzer.exe"

[Setup]
AppId={{3DBBBEFE-3633-4E16-9347-2DD7D8627E68}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist
OutputBaseFilename=RadiochromicFilmAnalyzer-Setup
SetupIconFile=..\resources\radiochromic_film_analyzer.ico
UninstallDisplayIcon={app}\{#AppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no
WizardStyle=modern

[Files]
Source: "..\dist\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\docs\*"; DestDir: "{app}\docs"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\custom_plugins\PLUGIN_DEVELOPMENT_GUIDE.md"; DestDir: "{app}\docs"; DestName: "PLUGIN_DEVELOPMENT_GUIDE.md"; Flags: ignoreversion
Source: "PLUGIN_DIRECTORY_README.md"; DestDir: "{app}\custom_plugins"; DestName: "README.md"; Flags: ignoreversion

[Dirs]
Name: "{app}\logs"; Flags: uninsneveruninstall
Name: "{app}\temp"; Flags: uninsneveruninstall
Name: "{app}\custom_plugins"; Flags: uninsneveruninstall
Name: "{app}\calibration_data"; Flags: uninsneveruninstall

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
