#define MyAppName "DopplerView"

#ifndef AppVersion
  #define AppVersion "0.3.0"
#endif

#define MyAppId MyAppName + "-" + AppVersion
#define MyAppVersionedName MyAppName + " " + AppVersion

#ifndef PayloadDir
  #error PayloadDir must be provided on the ISCC command line.
#endif

#ifndef OutputDir
  #define OutputDir "dist"
#endif

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#AppVersion}
AppVerName={#MyAppVersionedName}
DefaultDirName={autopf}\{#MyAppName}\{#AppVersion}
DefaultGroupName={#MyAppVersionedName}
DisableProgramGroupPage=yes
LicenseFile={#PayloadDir}\LICENSE
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=DopplerView-setup-{#AppVersion}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
SetupIconFile={#PayloadDir}\DopplerView.ico
UninstallDisplayName={#MyAppVersionedName}
UninstallDisplayIcon={app}\DopplerView.exe
UsePreviousAppDir=no
UsePreviousGroup=no
UsePreviousTasks=no

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "{#PayloadDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppVersionedName}"; Filename: "{app}\DopplerView.exe"
Name: "{autodesktop}\{#MyAppVersionedName}"; Filename: "{app}\DopplerView.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\DopplerView.exe"; Description: "Launch {#MyAppVersionedName}"; Flags: nowait postinstall skipifsilent
