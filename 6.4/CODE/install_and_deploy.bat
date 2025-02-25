ECHO ************** BUILD AND DEPLOY:  - Pass ALM ****************************
cd "venv\Scripts"
call activate.bat
cd ..
cd ..
cd src
if exist .\build  rd /s /q .\build
if exist .\dist  rd /s /q .\dist
pyinstaller main.spec
move .\dist\*.* ..
if exist .\build  rd /s /q .\build
if exist .\dist  rd /s /q .\dist


ECHO **************************** END    *************** ****************************

cmd /k




