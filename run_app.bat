@echo off
title NEUROML-X : TRACE TOXIC AI - "AI That Detects. ML That Protects."
echo ======================================================================
echo           NEUROML-X : TRACE TOXIC AI (NVIDIA CUDA Accelerated)
echo       NVIDIA x Nebius Global AI Hackathon - Best Apps & Agents
echo ======================================================================
echo.
cd /d "%~dp0"
streamlit run app.py --server.port=8501 --server.headless=false
pause
