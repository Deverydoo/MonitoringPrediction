# Python Environment Information

## Active Environment

**Environment Name**: `py310`

This project uses a Python 3.10 environment named `py310`.

---

## Activation

### Windows
```bash
conda activate py310
# or
py310\Scripts\activate
```

### Linux/Mac
```bash
conda activate py310
# or
source py310/bin/activate
```

---

## Running Scripts

Always ensure you're in the `py310` environment before running any scripts:

```bash
# Activate environment first
conda activate py310

# Then run scripts
python demo_data_generator.py --scenario critical
python run_demo.py
python main.py train --epochs 20
python test_scenarios.py
```

---

## Jupyter Notebook

The notebook `_StartHere.ipynb` should be configured to use the `py310` kernel.

To verify/set the kernel:
1. Open the notebook
2. Click on the kernel name (top-right)
3. Select `py310` from the list

---

## Environment Dependencies

Key packages required (should already be installed in `py310`):
- PyTorch 2.0+
- Lightning 2.0+
- PyTorch Forecasting 1.0+
- Pandas 2.2+
- PyArrow (for Parquet support)
- NumPy
- Matplotlib
- Safetensors

To verify installation:
```bash
conda activate py310
python -c "from main import setup; setup()"
```

---

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'numpy'`
- **Cause**: Wrong Python environment active
- **Solution**: Activate `py310` environment first

**Issue**: Script runs but uses wrong Python version
- **Cause**: System Python being used instead of `py310`
- **Solution**:
  ```bash
  conda activate py310
  which python  # Should show py310 path
  ```

---

Last Updated: 2025-10-08
