@echo off
setlocal EnableDelayedExpansion

REM === SET YOUR VIDEO ROOT DIRECTORY HERE ===
set "VIDEODIR=D:\data\UCF101\UCF-101"

REM === OUTPUT FILES (created next to the BAT file) ===
set "OKCSV=%~dp0videos_ok2.csv"
set "ERRCSV=%~dp0videos_error2.csv"

echo File,Codec,Width,Height,Duration > "%OKCSV%"
echo File,Error > "%ERRCSV%"

for /r "%VIDEODIR%" %%F in (*) do (

    REM --- Try to read first video stream ---
    for /f "usebackq tokens=1-4 delims=," %%A in (`
        ffprobe -v error -select_streams v:0 ^
          -show_entries stream=codec_name,width,height ^
          -show_entries format=duration ^
          -of csv=p=0 "%%F" 2^>^&1
    `) do (

        set "LINE=%%A,%%B,%%C,%%D"

        REM --- If ffprobe output starts with a letter, it's an error ---
        echo %%A | findstr /R "^[A-Za-z]" >nul
        if errorlevel 1 (
            echo "%%F",!LINE!>> "%OKCSV%"
        ) else (
            echo "%%F","%%A">> "%ERRCSV%"
        )
        goto :nextfile
    )

    :nextfile
)

echo Done!
pause


REM @echo off
REM setlocal EnableDelayedExpansion

REM REM === SET YOUR VIDEO ROOT DIRECTORY HERE ===
REM set "VIDEODIR=D:\data\UCF101\UCF-101"

REM REM === OUTPUT FILES (created next to the BAT file) ===
REM set "OKCSV=%~dp0videos_ok.csv"
REM set "ERRCSV=%~dp0videos_error.csv"
REM set "TEMP=%~dp0temp_ffprobe.txt"

REM echo File,Codec,Width,Height,Duration > "%OKCSV%"
REM echo File,Error > "%ERRCSV%"

REM for /r "%VIDEODIR%" %%F in (*) do (

    REM ffprobe -v error -select_streams v:0 ^
      REM -show_entries stream=codec_name,width,height ^
      REM -show_entries format=duration ^
      REM -of csv=p=0 "%%F" > "%TEMP%" 2>&1

    REM if errorlevel 1 (
        REM for /f "usebackq delims=" %%E in ("%TEMP%") do (
            REM echo "%%F","%%E" >> "%ERRCSV%"
        REM )
    REM ) else (
        REM for /f "usebackq delims=" %%A in ("%TEMP%") do (
            REM echo "%%F",%%A >> "%OKCSV%"
        REM )
    REM )
REM )

REM del "%TEMP%" 2>nul
REM echo Done!
REM pause


