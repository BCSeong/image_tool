@echo off
REM Build image_tool executable with PyInstaller

pyinstaller --noconfirm --onefile --windowed --name ImageTool ^
    --hidden-import=plugins.focus_analysis ^
    --hidden-import=plugins.focus_analysis.focus_analyzer ^
    --hidden-import=plugins.focus_analysis.result_plotter ^
    --hidden-import=plugins.focus_analysis.blob_dialog ^
    --hidden-import=plugins.focus_analysis.dialog ^
    --hidden-import=scipy.ndimage ^
    --hidden-import=scipy.signal ^
    --hidden-import=scipy.optimize ^
    main.py

echo.
if %ERRORLEVEL% EQU 0 (
    echo Build complete: dist\ImageTool.exe
) else (
    echo Build FAILED.
)
pause
2