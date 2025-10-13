# Contributing to TFT Monitoring Prediction System

Thank you for your interest in contributing! This project was built using AI-assisted development, and we welcome contributions from the community.

## 🚀 Quick Start for Contributors

### 1. Set Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/MonitoringPrediction.git
cd MonitoringPrediction

# Create conda environment
conda create -n py310 python=3.10
conda activate py310

# Install dependencies
pip install -r requirements.txt

# Verify GPU (optional but recommended)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. Generate Training Data

```bash
# Generate 30 days of test data
python metrics_generator.py --servers 20 --hours 720 --output ./training/
```

### 3. Train Model (Optional)

```bash
# Train for testing (quick)
python tft_trainer.py --epochs 5

# Full training (production)
python tft_trainer.py --epochs 20
```

### 4. Run System

```bash
# Terminal 1: Start inference daemon
python tft_inference.py --daemon --port 8000 --fleet-size 20

# Terminal 2: Launch dashboard
streamlit run tft_dashboard_web.py
```

## 📝 How to Contribute

### Types of Contributions Welcome

- 🐛 **Bug fixes** - Found an issue? Submit a fix!
- 📚 **Documentation** - Improve guides, add examples
- 🎨 **Dashboard improvements** - UI/UX enhancements
- ⚡ **Performance optimizations** - Speed up training/inference
- 🧪 **Testing** - Add tests, improve coverage
- 🔌 **Integrations** - PagerDuty, Slack, Prometheus, etc.
- 🌍 **New server profiles** - Kubernetes, Redis, message queues
- 📊 **Visualization** - Better charts, attention heatmaps

### Contribution Workflow

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Test thoroughly**
5. **Commit with clear messages**: `git commit -m "Add feature: description"`
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Open a Pull Request**

## 🎯 Code Standards

### Python Style

- **PEP 8** compliant (with 100-char line limit)
- **Type hints** for function signatures
- **Docstrings** for all public functions (Google style)
- **Clear variable names** - readability over brevity

### Example:

```python
def encode_server_name(server_name: str) -> str:
    """
    Create deterministic hash-based encoding for a server name.

    Args:
        server_name: Original hostname (e.g., 'ppvra00a0018')

    Returns:
        Consistent numeric string ID (e.g., '123456')

    Examples:
        >>> encode_server_name('ppvra00a0018')
        '849234'
    """
    hash_obj = hashlib.sha256(server_name.encode('utf-8'))
    hash_int = int(hash_obj.hexdigest()[:8], 16)
    return str(hash_int % 1_000_000)
```

### Commit Messages

Use conventional commits:

```
feat: Add Kubernetes server profile
fix: Correct encoder persistence bug
docs: Update installation instructions
perf: Optimize data loading with Parquet caching
test: Add unit tests for server encoder
refactor: Simplify prediction pipeline
```

## 🧪 Testing

### Before Submitting PR

```bash
# 1. Verify code works
python tft_trainer.py --epochs 1  # Quick smoke test

# 2. Check inference daemon
python tft_inference.py --daemon --port 8000

# 3. Test dashboard
streamlit run tft_dashboard_web.py

# 4. Run any new tests you added
pytest tests/ -v
```

### Test Coverage

- **Critical paths** must have tests
- **New features** must include test cases
- **Bug fixes** should include regression tests

## 📋 Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation updated (if applicable)
- [ ] README updated (if adding features)
- [ ] CHANGELOG updated (for notable changes)
- [ ] No breaking changes (or clearly documented)
- [ ] PR description explains what and why

## 🔒 Important Constraints

### Feature Scope

This project follows a **feature-locked** approach for the core demo. See [FEATURE_LOCK.md](Docs/FEATURE_LOCK.md) for current status.

**Currently accepting:**
- ✅ Bug fixes
- ✅ Performance optimizations
- ✅ Code quality improvements
- ✅ Documentation enhancements

**Post-demo (future):**
- See [FUTURE_ROADMAP.md](Docs/FUTURE_ROADMAP.md) for planned features

### Data Contract

All changes must respect the **Data Contract** ([DATA_CONTRACT.md](Docs/DATA_CONTRACT.md)):

- ✅ Maintain schema compatibility
- ✅ Preserve valid states
- ✅ Keep encoder determinism
- ✅ Version all breaking changes

## 🤝 Code of Conduct

### Our Standards

- **Be respectful** - Treat all contributors with respect
- **Be constructive** - Provide helpful feedback
- **Be collaborative** - Work together toward better solutions
- **Be patient** - Everyone was new once

### Not Tolerated

- Harassment or discriminatory language
- Trolling or insulting comments
- Personal attacks
- Unprofessional conduct

## 📞 Getting Help

### Questions?

- **General questions**: Open a [Discussion](https://github.com/yourusername/MonitoringPrediction/discussions)
- **Bug reports**: Open an [Issue](https://github.com/yourusername/MonitoringPrediction/issues)
- **Feature requests**: Check [FUTURE_ROADMAP.md](Docs/FUTURE_ROADMAP.md) first, then open Discussion

### Documentation

- [ESSENTIAL_RAG.md](Docs/ESSENTIAL_RAG.md) - Complete system reference
- [QUICK_START.md](Docs/QUICK_START.md) - Fast onboarding
- [PROJECT_CODEX.md](Docs/PROJECT_CODEX.md) - Architecture deep dive

## 🌟 Recognition

### Contributors

All contributors will be:
- Listed in AUTHORS.md
- Credited in release notes
- Acknowledged in documentation (for significant contributions)

### Star Contributors

Outstanding contributions may receive:
- Featured in README
- Highlighted in project showcase
- Direct collaboration opportunities

## 📜 License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

## 🎓 Learning Opportunity

This project was built using **AI-assisted development** with Claude Code. We encourage contributors to:

- Experiment with AI-assisted workflows
- Share learnings about human-AI collaboration
- Document novel approaches in your PRs

This is as much about **how we build** as **what we build**.

---

## 💡 Pro Tips

### For First-Time Contributors

1. Start small - fix typos, improve docs
2. Read [ESSENTIAL_RAG.md](Docs/ESSENTIAL_RAG.md) thoroughly
3. Run the system locally before coding
4. Ask questions in Discussions
5. Learn from existing code patterns

### For Experienced Contributors

1. Check [FUTURE_ROADMAP.md](Docs/FUTURE_ROADMAP.md) for high-priority items
2. Coordinate on large features via Discussion first
3. Consider mentoring new contributors
4. Help with code reviews

---

**Thank you for contributing!** 🙏

Your improvements help make predictive monitoring accessible to everyone.

*Built with AI, improved by humans.* 🤖🤝👨‍💻
