@echo off
chcp 65001 >nul
echo ========================================
echo 基金预测项目 - 环境安装脚本
echo ========================================
echo.

REM 检查Python是否已安装
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [✓] Python已安装
    python --version
    goto :install_deps
)

REM 检查Python3
python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [✓] Python3已安装
    python3 --version
    set PYTHON=python3
    goto :install_deps
)

REM Python未安装，提示用户安装
echo [✗] Python未安装
echo.
echo 请按以下步骤安装Python:
echo 1. 访问 https://www.python.org/downloads/
echo 2. 下载Python 3.10或更高版本
echo 3. 安装时务必勾选 "Add Python to PATH"
echo 4. 安装完成后重新运行此脚本
echo.
echo 或者使用winget自动安装:
echo   winget install Python.Python.3.11
echo.
pause
exit /b 1

:install_deps
echo.
echo 开始安装依赖包...
echo.

REM 代理自动检测：可用则走代理，否则直连
powershell -Command "try { (New-Object Net.Sockets.TcpClient).Connect('127.0.0.1', 7897); echo [proxy] 已启用: http://127.0.0.1:7897; $env:http_proxy='http://127.0.0.1:7897'; $env:https_proxy='http://127.0.0.1:7897' } catch { echo [proxy] 不可用，使用直连 }"

REM 升级pip
python -m pip install --upgrade pip

REM 安装依赖
echo.
echo [1/4] 安装数据处理包...
python -m pip install akshare pandas numpy matplotlib seaborn -q

echo [2/4] 安装机器学习包...
python -m pip install scikit-learn lightgbm xgboost shap -q

echo [3/4] 安装PyTorch (CUDA 11.8)...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q

echo [4/4] 安装其他工具...
python -m pip install snownlp jieba tqdm joblib plotly -q

echo.
echo ========================================
echo 安装完成!
echo ========================================
echo.
echo 下一步: 运行 python run.py 开始下载数据
echo.
pause
