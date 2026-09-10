# Nexora - Urban Flow Analytics Datathon 2026

## Project Structure
- `data/raw/` - original monthly CSV files (never modified, not committed to git)
- `data/interim/` - profiling outputs, schema reports, intermediate cleaning artifacts
- `data/processed/` - final cleaned dataset + train/val/test splits
- `notebooks/` - analysis and modeling notebooks
- `src/` - reusable cleaning/feature engineering functions
- `models/` - trained model .pkl files
- `reports/` - technical report and figures

## Setup
\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`
