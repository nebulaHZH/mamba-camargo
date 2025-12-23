@echo off
chcp 65001
echo ========================================
echo 多模型性能对比实验
echo 对比模型: Mamba, Transformer, XGBoost, SVM, LDA, MLP
echo ========================================
echo.
echo 正在运行模型对比...
echo.
python compare_models.py
echo.
echo ========================================
echo 实验完成!
echo ========================================
pause
