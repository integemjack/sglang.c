@echo off
rem ===========================================================================
rem  quickstart.bat -- zero-setup launcher for sglang-c (Windows, CPU + CUDA)
rem
rem  Usage:
rem    quickstart.bat                 build if needed, start the OpenAI-compatible
rem                                   HTTP server, wait for /health, then send one
rem                                   demo chat request (default mode)
rem    quickstart.bat cli  [model] [n]    one-shot CLI generation
rem    quickstart.bat gpu  [model] [port] CUDA HTTP server (cuda_http.exe)
rem    quickstart.bat test                run the CTest suite
rem    quickstart.bat build               full build: CPU (MinGW) + CUDA (MSVC)
rem    quickstart.bat help                this text
rem
rem  Environment overrides:
rem    SGL_MODEL=<dir>   model directory (default: first usable dir under models\)
rem    SGL_PORT=<port>   HTTP port (default 30000)
rem    SGL_NO_BUILD=1    never build, only run what is already built
rem    SGL_NO_PAUSE=1    do not wait for a key press at the end (CI / scripts)
rem ===========================================================================

setlocal EnableDelayedExpansion
set "MODE=%~1"
if "%MODE%"=="" set "MODE=serve"
if /i "%MODE%"=="-h"    set "MODE=help"
if /i "%MODE%"=="/?"   set "MODE=help"
if /i "%MODE%"=="--help" set "MODE=help"

rem --- HTTP port: 3rd argument, else SGL_PORT, else 30000 -------------------
set "PORT=%~3"
if "%PORT%"=="" set "PORT=%SGL_PORT%"
if "%PORT%"=="" set "PORT=30000"

rem --- locate the repository root (this script lives in <repo>\sglang-c) ----
set "SGC=%~dp0"
pushd "%SGC%.." 2>nul || goto :fail_root
set "ROOT=%CD%"
set "SGC=%ROOT%\sglang-c"

rem --- toolchains ----------------------------------------------------------
set "MINGW_BIN="
for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs*") do (
    if exist "%%~fD\mingw64\bin\gcc.exe" set "MINGW_BIN=%%~fD\mingw64\bin"
)
if not defined MINGW_BIN for %%G in (gcc.exe) do if not "%%~$PATH:G"=="" set "MINGW_BIN=%%~dp$PATH:G"

set "CUDA_BIN="
for /d %%D in ("%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v*") do (
    if exist "%%~fD\bin\nvcc.exe" set "CUDA_BIN=%%~fD\bin"
)

if /i "%MODE%"=="help"    goto :help
if /i "%MODE%"=="build"   goto :fullbuild
if /i "%MODE%"=="test"    goto :test
if /i "%MODE%"=="gpu"     goto :gpu
if /i "%MODE%"=="cli"     goto :cli
if /i "%MODE%"=="serve"   goto :serve
echo [quickstart] unknown mode "%MODE%"
goto :help

rem ===========================================================================
:help
echo.
echo   sglang-c quickstart  (pure-C CPU/CUDA inference stack)
echo   ---------------------------------------------------------------
echo   quickstart.bat                 build if needed + start server + demo request
echo   quickstart.bat cli  [model] [n]    one-shot CLI generation
echo   quickstart.bat gpu  [model] [port] CUDA HTTP server  (cuda_http.exe)
echo   quickstart.bat test                run the CTest suite
echo   quickstart.bat build               full build: CPU (MinGW) + CUDA (MSVC)
echo   quickstart.bat help                this text
echo.
echo   env: SGL_MODEL=^<dir^>   SGL_PORT=^<port^>   SGL_NO_BUILD=1   SGL_NO_PAUSE=1
echo   toolchain: MinGW GCC = %MINGW_BIN%
echo              CUDA      = %CUDA_BIN%
echo.
goto :done

rem ===========================================================================
rem  default: serve + demo
rem ===========================================================================
:serve
call :ensure_cpu_build || goto :fail
if not "%~2"=="" set "SGL_MODEL=%~2"
call :pick_model      || goto :fail
set "QUANT=--w8 --kv8"
if /i "!MODEL!" neq "!MODEL:gpt2=!" set "QUANT="

echo.
echo [quickstart] model : %MODEL%
echo [quickstart] server: http://127.0.0.1:%PORT%   (window "sglang-c server :%PORT%")
start "sglang-c server :%PORT%" "%SGC%\build\bin\sglc.exe" --model "%MODEL%" --serve --port %PORT% %QUANT%
call :wait_health || goto :fail_health
echo [quickstart] /health ok -- sending a demo chat request ...
echo.
set "PAYLOAD=%TEMP%\sglang-c-demo-%RANDOM%.json"
> "%PAYLOAD%" echo {"messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":24}
curl.exe -s -X POST "http://127.0.0.1:%PORT%/v1/chat/completions" -H "Content-Type: application/json" --data-binary "@%PAYLOAD%"
echo.
del "%PAYLOAD%" >nul 2>nul
echo.
echo [quickstart] server is up. Endpoints:
echo     GET  http://127.0.0.1:%PORT%/health
echo     POST http://127.0.0.1:%PORT%/generate
echo     POST http://127.0.0.1:%PORT%/v1/completions
echo     POST http://127.0.0.1:%PORT%/v1/chat/completions
echo [quickstart] stop it by closing the "sglang-c server :%PORT%" window (or Ctrl+C there).
goto :done

rem ===========================================================================
:cli
call :ensure_cpu_build || goto :fail
if not "%~2"=="" set "SGL_MODEL=%~2"
call :pick_model || goto :fail
set "N=%~3"
if "%N%"=="" set "N=32"
set "QUANT=--w8 --kv8"
if /i "!MODEL!" neq "!MODEL:gpt2=!" set "QUANT="
echo [quickstart] %MODEL%  (max-new=%N% %QUANT%)
"%SGC%\build\bin\sglc.exe" --model "%MODEL%" --prompt "The meaning of life is" --max-new %N% %QUANT%
goto :done

rem ===========================================================================
:gpu
if not exist "%SGC%\build-cuda\bin\Release\cuda_http.exe" call :ensure_cuda_build || goto :fail
if not "%~2"=="" set "SGL_MODEL=%~2"
call :pick_model || goto :fail
echo.
echo [quickstart] CUDA server: model=%MODEL% port=%PORT%
start "sglang-c cuda server :%PORT%" "%SGC%\build-cuda\bin\Release\cuda_http.exe" "%MODEL%" %PORT%
call :wait_health || goto :fail_health
echo [quickstart] /health ok -- CUDA server listening on http://127.0.0.1:%PORT%
echo [quickstart] optional: SGL_CUDA_CTX=8192 SGL_CUDA_FP16=1 SGL_CUDA_W8=1 SGL_CUDA_GRAPH=0
goto :done

rem ===========================================================================
:test
call :ensure_cpu_build || goto :fail
ctest --test-dir "%SGC%\build" --output-on-failure || goto :fail
echo [quickstart] ctest OK
goto :done

rem ===========================================================================
:fullbuild
if not defined MINGW_BIN (
    echo [quickstart] ERROR: MinGW gcc not found. Install it with:
    echo     winget install BrechtSanders.WinLibs.POSIX.UCRT
    goto :fail
)
set "PATH=%MINGW_BIN%;%PATH%"
if defined CUDA_BIN set "PATH=%CUDA_BIN%;%PATH%"
cmake -S "%SGC%" -B "%SGC%\build" -G Ninja || goto :fail
cmake --build "%SGC%\build" || goto :fail
echo [quickstart] CPU build OK: %SGC%\build\bin\sglc.exe
if not defined CUDA_BIN (
    echo [quickstart] CUDA toolkit not found - skipped the CUDA stack.
    goto :done
)
cmake -S "%SGC%" -B "%SGC%\build-cuda" -G "Visual Studio 17 2022" -A x64 -DSGL_CUDA=ON || goto :fail
cmake --build "%SGC%\build-cuda" --config Release --target cuda_test cuda_cli cuda_http || goto :fail
echo [quickstart] CUDA build OK: %SGC%\build-cuda\bin\Release\{cuda_test,cuda_cli,cuda_http}.exe
goto :done

rem ===========================================================================
rem  helpers
rem ===========================================================================
:ensure_cpu_build
if exist "%SGC%\build\bin\sglc.exe" exit /b 0
if "%SGL_NO_BUILD%"=="1" (
    echo [quickstart] ERROR: %SGC%\build\bin\sglc.exe is missing and SGL_NO_BUILD=1
    exit /b 1
)
if not defined MINGW_BIN (
    echo [quickstart] ERROR: no MinGW gcc found, cannot build. Install it with:
    echo     winget install BrechtSanders.WinLibs.POSIX.UCRT
    exit /b 1
)
echo [quickstart] first run: building the CPU stack with MinGW + Ninja ...
set "PATH=%MINGW_BIN%;%PATH%"
cmake -S "%SGC%" -B "%SGC%\build" -G Ninja || exit /b 1
cmake --build "%SGC%\build" || exit /b 1
exit /b 0

:ensure_cuda_build
if "%SGL_NO_BUILD%"=="1" exit /b 1
if not defined CUDA_BIN (
    echo [quickstart] ERROR: CUDA toolkit not found (install CUDA 13.x + VS2022 Build Tools)
    exit /b 1
)
echo [quickstart] CUDA build missing: building cuda_test / cuda_cli / cuda_http ...
set "PATH=%CUDA_BIN%;%PATH%"
cmake -S "%SGC%" -B "%SGC%\build-cuda" -G "Visual Studio 17 2022" -A x64 -DSGL_CUDA=ON || exit /b 1
cmake --build "%SGC%\build-cuda" --config Release --target cuda_test cuda_cli cuda_http || exit /b 1
exit /b 0

:pick_model
set "MODEL="
if defined SGL_MODEL if exist "%SGL_MODEL%\config.json" set "MODEL=%SGL_MODEL%"
if not defined MODEL if exist "%ROOT%\models\qwen05\config.json"    set "MODEL=%ROOT%\models\qwen05"
if not defined MODEL if exist "%ROOT%\models\qwen3-06\config.json"  set "MODEL=%ROOT%\models\qwen3-06"
if not defined MODEL if exist "%ROOT%\models\tinyllama\config.json" set "MODEL=%ROOT%\models\tinyllama"
if not defined MODEL if exist "%ROOT%\models\gpt2\config.json"      set "MODEL=%ROOT%\models\gpt2"
if not defined MODEL for /d %%D in ("%ROOT%\models\*") do (
    if not defined MODEL if exist "%%~fD\config.json" set "MODEL=%%~fD"
)
if not defined MODEL for /d %%D in ("%SGC%\models\*") do (
    if not defined MODEL if exist "%%~fD\config.json" set "MODEL=%%~fD"
)
if not defined MODEL (
    echo [quickstart] ERROR: no model found. Put an HF-format model ^(config.json,
    echo   tokenizer.json, model.safetensors^) under %ROOT%\models\ or set SGL_MODEL.
    exit /b 1
)
exit /b 0

:wait_health
set /a WH=0
:wh_loop
set /a WH+=1
curl.exe -s -o nul --max-time 2 "http://127.0.0.1:%PORT%/health" >nul 2>nul && (
    echo [quickstart] server answered on port %PORT% after %WH%s
    exit /b 0
)
if !WH! geq 120 (
    echo [quickstart] health check timed out after %WH%s ^(model loading can be slow - check the server window^)
    exit /b 1
)
>nul ping -n 2 127.0.0.1
goto :wh_loop

rem ===========================================================================
:fail_root
echo [quickstart] ERROR: cannot enter the repository directory
goto :fail
:fail_health
echo [quickstart] ERROR: server did not become healthy - see its console window
goto :fail
:fail
if "%SGL_NO_PAUSE%"=="1" exit /b 1
echo.
pause
popd
endlocal
exit /b 1

:done
popd
if "%SGL_NO_PAUSE%"=="1" endlocal & exit /b 0
echo.
echo [quickstart] done - press any key to close this window
pause >nul
endlocal
exit /b 0