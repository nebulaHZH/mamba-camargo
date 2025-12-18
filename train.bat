@echo off
chcp 65001 >nul
echo ============================================================
echo          Mamba运动分类系统 - 训练脚本
echo ============================================================
echo.

echo [1/3] 检查环境...
python test_setup.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ 环境检查失败，请先安装依赖:
    echo    pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [2/3] 开始训练模型...
echo ============================================================
python main.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ 训练失败
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [3/3] 训练完成!
echo ============================================================
echo.
echo 生成的文件:
echo   - best_model.pth (最佳模型)
echo   - training_history.png (训练曲线)
echo   - confusion_matrix.png (混淆矩阵)
echo.
echo 现在可以使用 inference.py 进行预测!
echo.
pause
