@echo off
REM filepath: c:\Users\prayh\source\repos\spacy-nlp\automation.bat

REM Step 1: Check for Anaconda/Miniconda/Miniforge
echo Checking for Anaconda, Miniconda, or Miniforge...
where conda >nul 2>nul
if %errorlevel%==0 (
    echo Conda found!
) else (
    echo Conda not found. Please install Anaconda, Miniconda, or Miniforge manually, then rerun this script.
    pause
    exit /b 1
)

REM Step 2: Check for environment
echo Checking for spacy_nlp_py310 environment...
conda info --envs | findstr /C:"spacy_nlp_py310"
if %errorlevel%==0 (
    echo Environment found!
) else (
    echo Environment not found. Creating from environment.yml...
    conda env create -f environment.yml
    if %errorlevel% neq 0 (
        echo Failed to create environment. Please check environment.yml.
        pause
        exit /b 1
    )
)

REM Step 3: Activate environment and fill config
echo Activating environment...
call conda activate spacy_nlp_py310

echo Generating config.cfg from base_config.cfg...
if not exist base_config.cfg (
    echo base_config.cfg not found! Please provide it.
    pause
    exit /b 1
)
python -m spacy init fill-config base_config.cfg config.cfg
if %errorlevel% neq 0 (
    echo Failed to generate config.cfg.
    pause
    exit /b 1
)

REM Step 4: Loop through folds and train
echo Starting training for each fold...
python run_folds.py

REM Step 5: Aggregate metrics from all folds
echo Aggregating metrics from all folds...

python aggregate_metrics.py

echo Metrics aggregation complete!
pause