# Demo Setup Guide

Quick guide to get the demo running.

## Prerequisites

Python 3.8+ with these packages:
```bash
pip install pandas numpy matplotlib pyarrow ipython
```

Or use the requirements from your existing environment since the other modules already work.

## Quick Start

### Option 1: One Command
```bash
python run_demo.py
```

This will:
1. Check for existing demo data (or generate it)
2. Launch the dashboard
3. Play through the 5-minute incident scenario

### Option 2: Manual Steps

**Step 1: Generate Demo Data**
```bash
python demo_data_generator.py
```

You should see:
```
🎬 Generating demo dataset...
   Duration: 5 minutes
   Tick interval: 5 seconds
   Fleet size: 10 servers
   Total data points: 600
✅ CSV saved: demo_data/demo_dataset.csv
✅ Parquet saved: demo_data/demo_dataset.parquet
✅ Metadata saved: demo_data/demo_dataset_metadata.json
```

**Step 2: Run Dashboard**
```bash
python tft_dashboard_refactored.py demo_data/demo_dataset.parquet
```

or for CSV:
```bash
python tft_dashboard_refactored.py demo_data/demo_dataset.csv
```

## What to Expect

### Timeline
Watch as the dashboard progresses through:
1. **0:00-1:30** - Everything healthy (green)
2. **1:30-2:30** - Warnings start appearing (yellow/orange)
3. **2:30-3:30** - Critical incident (red alerts)
4. **3:30-5:00** - Recovery back to normal

### Metrics to Watch
- **30-min probability**: Should rise from ~10% → 70-90% → back to 10%
- **Top servers**: `web-001`, `api-001`, `db-001`, `cache-001` will appear
- **CPU usage**: Watch median spike during incident
- **Latency P95**: Should increase significantly at peak

### Dashboard Figures
You'll see 5 live-updating figures:
1. Header with health status (GREEN → YELLOW → RED → GREEN)
2. Top 5 problem servers bar chart
3. Probability trend line graph
4. Fleet-wide risk heat map
5. Rolling metrics (CPU/latency/errors)

## Customization

### Different Duration
```bash
python demo_data_generator.py --duration 10
```

### More Servers
```bash
python demo_data_generator.py --fleet-size 20
```

### Faster Dashboard
```bash
python tft_dashboard_refactored.py demo_data/demo_dataset.parquet --refresh 2
```

### Different Random Seed (Different Pattern)
```bash
python demo_data_generator.py --seed 123
```

## Verification

After generation, check the metadata:
```bash
cat demo_data/demo_dataset_metadata.json
```

Should show:
- Total records: 600 (for 5min x 10 servers x 12 ticks/min)
- Affected servers list
- Phase distribution
- Incident pattern description

## Troubleshooting

### "No module named 'numpy'"
```bash
pip install numpy pandas matplotlib
```

### "No module named 'pyarrow'"
Either:
```bash
pip install pyarrow
```
Or use CSV instead:
```bash
python tft_dashboard_refactored.py demo_data/demo_dataset.csv --format csv
```

### "Data source not found"
Make sure you generated the data first:
```bash
python demo_data_generator.py
```

### Dashboard window closes immediately
You might need to run in interactive mode or from Jupyter:
```python
from tft_dashboard_refactored import run_dashboard
run_dashboard("demo_data/demo_dataset.parquet")
```

### Want to see it again
Either:
1. Run `python run_demo.py` again (will reuse existing data)
2. Or force regeneration: `python run_demo.py --regenerate`

## Jupyter Notebook Usage

```python
# Cell 1: Generate data
from demo_data_generator import generate_demo_dataset
generate_demo_dataset(output_dir="./demo_data/")

# Cell 2: View dashboard
from tft_dashboard_refactored import run_dashboard
run_dashboard(
    data_path="./demo_data/demo_dataset.parquet",
    refresh_sec=5
)
```

The dashboard figures will update inline in the notebook.

## For Presentations

### Before Your Demo
1. Generate data: `python demo_data_generator.py`
2. Review metadata to know what to expect
3. Practice the narrative (see below)

### Demo Narrative
```
"Let me show you our TFT monitoring dashboard with a realistic incident scenario.

[Start dashboard]

We're starting with a healthy environment - all servers green,
incident probability around 10%.

[1:30 mark]
Now we're entering an escalation phase. Notice how the probability
is climbing and some servers are turning yellow. The system is
detecting early warning signs.

[2:30 mark]
Now we're at the peak of the incident. Several servers have gone
critical - see the red bars? The 30-minute probability has jumped
to 70-80%, giving us advanced warning.

[3:30 mark]
The system is recovering. Probability is dropping, servers are
stabilizing. This demonstrates how the model would help you
anticipate and respond to issues.

[End]
All metrics back to normal. The entire scenario was reproducible -
same results every time - making it perfect for training and testing."
```

## Next Steps After Demo

1. **Connect Real Data**: Point dashboard to production metrics
2. **Train Model**: Use real data to train TFT model
3. **Replace Predictions**: Use trained model instead of ModelAdapter
4. **Integrate Alerts**: Connect to your alerting system

## Files Reference

Generated files:
- `demo_data/demo_dataset.csv` - CSV format (larger, slower)
- `demo_data/demo_dataset.parquet` - Parquet format (smaller, faster) ⭐
- `demo_data/demo_dataset_metadata.json` - Incident info and stats

Source files:
- [demo_data_generator.py](demo_data_generator.py) - Data generator
- [tft_dashboard_refactored.py](tft_dashboard_refactored.py) - Dashboard
- [run_demo.py](run_demo.py) - Convenience runner

Documentation:
- [DEMO_README.md](DEMO_README.md) - Complete guide
- [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) - What changed and why
- **SETUP_DEMO.md** - This quick start guide
