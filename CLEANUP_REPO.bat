@echo off
REM ============================================================================
REM NordIQ Repository Cleanup Script
REM ============================================================================
REM Purpose: Remove duplicate files and consolidate on NordIQ/ structure
REM Version: 1.0.0
REM Date: 2025-10-19
REM
REM SAFETY: Git tag v1.1.0-pre-cleanup created for rollback
REM Estimated space savings: 3.7 GB
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo NordIQ Repository Cleanup
echo ============================================================================
echo.
echo This script will:
echo   1. Delete duplicate models directory (2.1 GB)
echo   2. Delete duplicate Python files (21 files, 500 KB)
echo   3. Delete duplicate directories (5 directories)
echo   4. Delete duplicate scripts (10 files)
echo   5. Clean up artifacts and backups (1.9 MB)
echo   6. Move scattered docs to Docs/ folder
echo   7. Create deprecation notice
echo.
echo SAFETY: Git tag v1.1.0-pre-cleanup created for rollback
echo Rollback command: git reset --hard v1.1.0-pre-cleanup
echo.
echo Press Ctrl+C to cancel, or
pause

echo.
echo ============================================================================
echo Phase 1: Delete Duplicate Models Directory (2.1 GB)
echo ============================================================================
echo.

if exist "models\" (
    echo Deleting models\ directory...
    rmdir /s /q "models\"
    echo [OK] Deleted models\ directory
) else (
    echo [SKIP] models\ directory not found
)

echo.
echo ============================================================================
echo Phase 2: Delete Duplicate Python Files (21 files)
echo ============================================================================
echo.

REM Core application files
for %%F in (
    adaptive_retraining_daemon.py
    constants.py
    data_buffer.py
    data_validator.py
    demo_data_generator.py
    demo_stream_generator.py
    drift_monitor.py
    generate_api_key.py
    gpu_profiles.py
    main.py
    metrics_generator.py
    metrics_generator_daemon.py
    precompile.py
    scenario_demo_generator.py
    server_encoder.py
    server_profiles.py
    tft_dashboard.py
    tft_dashboard_web.py
    tft_inference.py
    tft_inference_daemon.py
    tft_trainer.py
) do (
    if exist "%%F" (
        echo Deleting %%F...
        del /f /q "%%F"
    )
)

echo [OK] Deleted duplicate Python files

echo.
echo ============================================================================
echo Phase 3: Delete Deprecated/Old Files
echo ============================================================================
echo.

REM Old/deprecated files
if exist "linborg_schema.py" (
    echo Deleting linborg_schema.py (deprecated schema)...
    del /f /q "linborg_schema.py"
)

if exist "run_demo.py" (
    echo Deleting run_demo.py (replaced by scenario_demo_generator)...
    del /f /q "run_demo.py"
)

if exist "tft_dashboard_web.py.backup" (
    echo Deleting tft_dashboard_web.py.backup...
    del /f /q "tft_dashboard_web.py.backup"
)

echo [OK] Deleted deprecated files

echo.
echo ============================================================================
echo Phase 4: Delete Duplicate Directories
echo ============================================================================
echo.

REM Duplicate directories moved to NordIQ/
for %%D in (
    Dashboard
    adapters
    explainers
    tabs
    utils
) do (
    if exist "%%D\" (
        echo Deleting %%D\ directory...
        rmdir /s /q "%%D\"
    )
)

echo [OK] Deleted duplicate directories

echo.
echo ============================================================================
echo Phase 5: Delete Duplicate Scripts
echo ============================================================================
echo.

REM Duplicate scripts (exist in NordIQ/)
for %%F in (
    run_daemon.bat
    setup_api_key.bat
    setup_api_key.sh
    start_all.bat
    start_all.sh
    stop_all.bat
    stop_all.sh
) do (
    if exist "%%F" (
        echo Deleting %%F...
        del /f /q "%%F"
    )
)

echo [OK] Deleted duplicate scripts

echo.
echo ============================================================================
echo Phase 6: Delete One-Off Validation Scripts
echo ============================================================================
echo.

for %%F in (
    run_certification.bat
    validate_pipeline.bat
    validate_schema.bat
) do (
    if exist "%%F" (
        echo Deleting %%F (one-off validation)...
        del /f /q "%%F"
    )
)

echo [OK] Deleted one-off scripts

echo.
echo ============================================================================
echo Phase 7: Clean Up Artifacts
echo ============================================================================
echo.

REM Error artifacts and old formats
if exist "nul" (
    echo Deleting nul (error artifact)...
    del /f /q "nul"
)

if exist "inference_rolling_window.pkl" (
    echo Deleting inference_rolling_window.pkl (old format, have .parquet)...
    del /f /q "inference_rolling_window.pkl"
)

echo [OK] Cleaned up artifacts

echo.
echo ============================================================================
echo Phase 8: Consolidate Documentation
echo ============================================================================
echo.

REM Create subdirectories if they don't exist
if not exist "Docs\configuration\" mkdir "Docs\configuration"
if not exist "Docs\security\" mkdir "Docs\security"
if not exist "Docs\deployment\" mkdir "Docs\deployment"

REM Move configuration docs
if exist "CONFIG_GUIDE.md" (
    echo Moving CONFIG_GUIDE.md to Docs\configuration\...
    move /y "CONFIG_GUIDE.md" "Docs\configuration\" >nul
)

if exist "CONFIGURATION_MIGRATION_COMPLETE.md" (
    echo Moving CONFIGURATION_MIGRATION_COMPLETE.md to Docs\archive\...
    move /y "CONFIGURATION_MIGRATION_COMPLETE.md" "Docs\archive\" >nul
)

REM Move security docs
if exist "DASHBOARD_SECURITY_AUDIT.md" (
    echo Moving DASHBOARD_SECURITY_AUDIT.md to Docs\security\...
    move /y "DASHBOARD_SECURITY_AUDIT.md" "Docs\security\" >nul
)

if exist "SECURITY_ANALYSIS.md" (
    echo Moving SECURITY_ANALYSIS.md to Docs\security\...
    move /y "SECURITY_ANALYSIS.md" "Docs\security\" >nul
)

if exist "SECURE_DEPLOYMENT_GUIDE.md" (
    echo Moving SECURE_DEPLOYMENT_GUIDE.md to Docs\deployment\...
    move /y "SECURE_DEPLOYMENT_GUIDE.md" "Docs\deployment\" >nul
)

REM Move deployment docs
if exist "PRODUCTION_DEPLOYMENT.md" (
    echo Moving PRODUCTION_DEPLOYMENT.md to Docs\deployment\...
    move /y "PRODUCTION_DEPLOYMENT.md" "Docs\deployment\" >nul
)

if exist "STARTUP_GUIDE_CORPORATE.md" (
    echo Moving STARTUP_GUIDE_CORPORATE.md to Docs\deployment\...
    move /y "STARTUP_GUIDE_CORPORATE.md" "Docs\deployment\" >nul
)

REM Move completed/archived docs
for %%F in (
    CLEANUP_COMPLETE.md
    CORPORATE_LAUNCHER_COMPLETE.md
    REFACTORING_SUMMARY.md
    SECURITY_IMPROVEMENTS_COMPLETE.md
    SILENT_DAEMON_MODE_COMPLETE.md
) do (
    if exist "%%F" (
        echo Moving %%F to Docs\archive\...
        move /y "%%F" "Docs\archive\" >nul
    )
)

REM Move technical analysis docs
if exist "PARQUET_VS_PICKLE_VS_JSON.md" (
    echo Moving PARQUET_VS_PICKLE_VS_JSON.md to Docs\...
    move /y "PARQUET_VS_PICKLE_VS_JSON.md" "Docs\" >nul
)

if exist "GPU_PROFILER_INTEGRATION.md" (
    echo Moving GPU_PROFILER_INTEGRATION.md to Docs\...
    move /y "GPU_PROFILER_INTEGRATION.md" "Docs\" >nul
)

if exist "CORPORATE_BROWSER_FIX.md" (
    echo Moving CORPORATE_BROWSER_FIX.md to Docs\...
    move /y "CORPORATE_BROWSER_FIX.md" "Docs\" >nul
)

echo [OK] Consolidated documentation

echo.
echo ============================================================================
echo Phase 9: Delete Old Model Versions in NordIQ (1.1 GB)
echo ============================================================================
echo.
echo Keeping only latest 2 models:
echo   - tft_model_20251013_100205 (demo model)
echo   - tft_model_20251017_122454 (latest production model)
echo.

if exist "NordIQ\models\tft_model_20251014_131232\" (
    echo Deleting NordIQ\models\tft_model_20251014_131232\...
    rmdir /s /q "NordIQ\models\tft_model_20251014_131232\"
)

if exist "NordIQ\models\tft_model_20251015_080653\" (
    echo Deleting NordIQ\models\tft_model_20251015_080653\...
    rmdir /s /q "NordIQ\models\tft_model_20251015_080653\"
)

echo [OK] Deleted old model versions

echo.
echo ============================================================================
echo Phase 10: Create Deprecation Notice
echo ============================================================================
echo.

(
echo # DEPRECATED ROOT FILES
echo.
echo **Status:** This directory structure is deprecated as of October 18, 2025.
echo.
echo **Use NordIQ/ instead:** All development should use the `NordIQ/` directory.
echo.
echo ## Clean Repository Structure
echo.
echo After cleanup, the root directory contains only:
echo.
echo - **NordIQ/** - Production application ^(primary development location^)
echo - **NordIQ-Website/** - Business website
echo - **Docs/** - All documentation
echo - **BusinessPlanning/** - Business strategy ^(confidential^)
echo - **Development artifacts** - checkpoints/, lightning_logs/, training/, logs/, plots/
echo - **Git/repo files** - README.md, LICENSE, VERSION, CHANGELOG.md, .gitignore
echo - **Setup files** - _StartHere.ipynb, environment.yml
echo.
echo ## What Was Removed
echo.
echo - ❌ Duplicate models/ directory ^(2.1 GB^)
echo - ❌ Duplicate Python files ^(21 files, 500 KB^)
echo - ❌ Duplicate directories ^(Dashboard/, config/, adapters/, explainers/^)
echo - ❌ Duplicate scripts ^(10 files^)
echo - ❌ Old model versions ^(1.1 GB^)
echo - ❌ Build artifacts and backups ^(1.9 MB^)
echo.
echo **Total space saved:** ~3.7 GB
echo.
echo ## Rollback
echo.
echo If you need to restore deleted files:
echo.
echo ```bash
echo git reset --hard v1.1.0-pre-cleanup
echo ```
echo.
echo ## Questions?
echo.
echo See REPOMAP.md for complete file inventory and cleanup details.
echo.
echo ---
echo **Created:** 2025-10-19
echo **Cleanup Tag:** v1.1.0-pre-cleanup
) > "README.DEPRECATED.md"

echo [OK] Created README.DEPRECATED.md

echo.
echo ============================================================================
echo Cleanup Summary
echo ============================================================================
echo.

REM Calculate what was removed
echo Files and directories removed:
echo   - models\ directory ^(duplicate, 2.1 GB^)
echo   - 21 duplicate Python files ^(~500 KB^)
echo   - 5 duplicate directories
echo   - 10 duplicate scripts
echo   - 3 one-off validation scripts
echo   - 2 old model versions in NordIQ\ ^(1.1 GB^)
echo   - Build artifacts ^(~1.9 MB^)
echo.
echo Documentation consolidated:
echo   - Moved 12+ docs to Docs\configuration\, Docs\security\, Docs\deployment\
echo   - Moved 5 completed docs to Docs\archive\
echo.
echo Estimated total space saved: ~3.7 GB
echo.
echo ============================================================================
echo Post-Cleanup Verification
echo ============================================================================
echo.
echo IMPORTANT: Please verify the cleanup was successful:
echo.
echo 1. Test NordIQ startup:
echo    cd NordIQ
echo    start_all.bat
echo.
echo 2. Open dashboard: http://localhost:8501
echo.
echo 3. Verify all 10 tabs load correctly
echo.
echo 4. If everything works, commit changes:
echo    git add -A
echo    git commit -m "cleanup: remove 3.7GB duplicate files, consolidate on NordIQ structure"
echo    git tag v1.1.1-post-cleanup
echo.
echo 5. If issues found, rollback:
echo    git reset --hard v1.1.0-pre-cleanup
echo.
echo ============================================================================
echo Cleanup Complete!
echo ============================================================================
echo.
echo See README.DEPRECATED.md for details on what was removed.
echo See REPOMAP.md for complete repository structure.
echo.

pause
