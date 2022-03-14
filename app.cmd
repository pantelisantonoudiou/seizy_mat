:: Check if Miniconda is present
IF EXIST %USERPROFILE%\miniconda3\Scripts\activate.bat (
    set conda_type=%USERPROFILE%\miniconda3\shell\condabin\conda-hook.ps1
) ELSE (
	IF EXIST %USERPROFILE%\anaconda3\Scripts\activate.bat (
        set conda_type=%USERPROFILE%\anaconda3\shell\condabin\conda-hook.ps1
)
)

:: Launch CLI
%windir%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -NoExit -Command "& '%conda_type%' ; conda activate seizy ; python cli.py"
