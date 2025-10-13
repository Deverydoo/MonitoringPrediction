# Model Training Guidelines

**Version:** 1.0.0
**Last Updated:** 2025-10-13
**Status:** Production Reference

---

## Purpose

This document provides guidance on training the TFT model for different use cases: demos, validation, and production deployment. It clarifies what performance claims can be made based on training configuration.

---

## Training Configurations

### Quick Demo (30 minutes)

**Use Case:** Proof of concept, architecture demonstration, time-constrained demos

```bash
# Generate data
python metrics_generator.py --hours 168 --out_dir ./training/

# Train model
python tft_trainer.py --epochs 1
```

**Specs:**
- **Data:** 1 week (168 hours)
- **Epochs:** 1
- **Time:** ~30 minutes on RTX 4090
- **Records:** ~33,600 (20 servers × 168 hours × 12 samples/hour)

**What You Get:**
- ✅ Working end-to-end pipeline
- ✅ Model loads and predicts
- ✅ Demo-ready system
- ✅ Initial loss metrics

**What You DON'T Get:**
- ❌ Converged model
- ❌ Validated accuracy metrics
- ❌ Production-ready predictions
- ❌ Long-term pattern learning

**Claims You CAN Make:**
> "This demonstrates the TFT architecture and prediction capability. The system is functional and ready for production training with extended data and epochs."

**Claims You CANNOT Make:**
- ❌ Any specific accuracy percentages
- ❌ "Production-ready predictions"
- ❌ "Fully trained model"

---

### Validation Training (2-6 hours)

**Use Case:** Realistic validation, initial performance metrics, pilot deployments

```bash
# Generate data
python metrics_generator.py --hours 720 --out_dir ./training/

# Train model
python tft_trainer.py --epochs 10
```

**Specs:**
- **Data:** 30 days (720 hours)
- **Epochs:** 10
- **Time:** ~4-6 hours on RTX 4090
- **Records:** ~172,800 (20 servers × 720 hours × 12 samples/hour)

**What You Get:**
- ✅ Partial convergence
- ✅ Weekly pattern learning
- ✅ Measurable validation metrics
- ✅ Honest performance baselines
- ✅ Reasonable predictions

**What You DON'T Get:**
- ❌ Full convergence (needs 20 epochs)
- ❌ Monthly/seasonal patterns
- ❌ Optimal performance

**Claims You CAN Make:**
> "Model trained on 30 days of data with 10 epochs. Initial validation shows [X]% quantile loss. Performance will improve with additional training epochs."

**What to Report:**
- Train loss (final value)
- Validation loss (final value)
- Loss convergence trend
- Time per epoch
- Prediction samples

---

### Production Training (30-40 hours)

**Use Case:** Production deployment, performance claims, maximum accuracy

```bash
# Generate data
python metrics_generator.py --hours 720 --out_dir ./training/

# Train model
python tft_trainer.py --epochs 20
```

**Specs:**
- **Data:** 30 days (720 hours)
- **Epochs:** 20
- **Time:** ~30-40 hours on RTX 4090
- **Records:** ~172,800 (20 servers × 720 hours × 12 samples/hour)

**What You Get:**
- ✅ Full convergence
- ✅ Production-quality predictions
- ✅ Validated accuracy metrics
- ✅ Weekly and monthly patterns
- ✅ Robust performance

**Claims You CAN Make:**
> "Model trained on 30 days of historical data with 20 epochs to full convergence. Validation metrics show [X]% accuracy on held-out data."

**What to Report:**
- Final train/validation loss
- Convergence curves
- Per-metric prediction quality
- Quantile loss values
- Production readiness assessment

---

## Large-Scale Production (8+ hours per epoch)

**Use Case:** Enterprise deployment, 100+ servers

```bash
# Generate data
python metrics_generator.py \
    --hours 720 \
    --num_ml_compute 40 \
    --num_database 25 \
    --num_web_api 50 \
    --num_conductor_mgmt 8 \
    --num_data_ingest 15 \
    --num_risk_analytics 12 \
    --num_generic 10 \
    --out_dir ./training/

# Train model
python tft_trainer.py --epochs 20
```

**Specs:**
- **Data:** 30 days, 160 servers
- **Epochs:** 20
- **Time:** ~8-12 hours per epoch (160-240 hours total)
- **Records:** ~1.4M records

**Recommendations:**
- Use gradient accumulation
- Enable checkpointing every 5 epochs
- Monitor GPU memory usage
- Use mixed precision (bf16-mixed)
- Consider distributed training

---

## Accuracy Claims Reference

### What the "88%" Means

The **88% accuracy** referenced in presentation documents is:
- ✅ A **target** based on TFT + transfer learning benchmarks
- ✅ Theoretical improvement from profile-based architecture
- ❌ NOT empirically validated (yet)
- ❌ NOT from actual training runs

**Source:** Literature on profile-based transfer learning showing 10-15% improvement over baseline models.

### Honest Accuracy Claims

**Without Full Training:**
> "The TFT architecture with profile-based transfer learning is designed to achieve 85-90% accuracy based on published benchmarks. Full validation will come from production pilot data."

**With Validation Training (10 epochs):**
> "Model achieves [measured validation loss] on held-out data. Preliminary results show [describe prediction quality]. Full convergence expected with 20 epochs."

**With Production Training (20 epochs):**
> "Model achieves [X]% accuracy on validation set with [Y] quantile loss. Predictions show strong alignment with historical patterns."

### Key Metrics to Track

During training, monitor and record:

1. **Loss Metrics:**
   - Training loss (per epoch)
   - Validation loss (per epoch)
   - Quantile loss (final value)

2. **Convergence:**
   - Loss curve trends
   - Early stopping triggers
   - Learning rate schedule

3. **Prediction Quality:**
   - Sample predictions vs. actuals
   - Confidence interval calibration
   - Per-metric performance (CPU, memory, disk)

4. **System Metrics:**
   - Time per epoch
   - GPU memory usage
   - Training throughput (samples/sec)

---

## Demo Strategy

### Timeline-Based Recommendations

**3 Days to Demo:**
```bash
# Run NOW (30 min)
python metrics_generator.py --hours 168 --out_dir ./training/
python tft_trainer.py --epochs 1

# Also run in background (if time allows)
python tft_trainer.py --epochs 10  # Using same data
```

Use 1-epoch for demo, mention that 10-epoch is running for validation.

**1 Week to Demo:**
```bash
# Best balance
python metrics_generator.py --hours 720 --out_dir ./training/
python tft_trainer.py --epochs 10
```

Show actual validation metrics, promise 20-epoch for production.

**1 Month to Demo:**
```bash
# Full production
python metrics_generator.py --hours 720 --out_dir ./training/
python tft_trainer.py --epochs 20
```

Make full accuracy claims with confidence.

---

## What to Say When Asked About Accuracy

### The Question:
> "What's the model's accuracy?"

### Good Answers (Based on Training Level):

**Quick Demo (1 epoch):**
> "This is a proof-of-concept model demonstrating the architecture. The TFT model with profile-based transfer learning is expected to achieve 85-90% accuracy based on benchmarks. Full validation requires completing the 20-epoch training cycle."

**Validation Training (10 epochs):**
> "The model is partially trained with 10 epochs on 30 days of data. Current validation loss is [X], showing good convergence. We expect final accuracy in the 85-90% range after completing the full 20-epoch training."

**Production Training (20 epochs):**
> "The model achieved [measured accuracy]% on held-out validation data after 20 epochs of training on 30 days of historical metrics. Quantile loss is [Y], indicating strong prediction quality."

### Bad Answers (Don't Say These):

❌ "88% accuracy" (without measurement)
❌ "Industry-leading accuracy" (unverified)
❌ "Better than existing solutions" (no comparison data)
❌ "Highly accurate" (vague and unmeasurable)

---

## Focus on Real Differentiators

When accuracy numbers are uncertain, pivot to what IS proven:

### Proven Features:
- ✅ **8-hour prediction horizon** - Demonstrated and working
- ✅ **Profile-based transfer learning** - Implemented and tested
- ✅ **Production-ready architecture** - REST API, daemon, dashboard
- ✅ **Real-time predictions** - Sub-3-second inference
- ✅ **Interactive scenarios** - Live demo capability
- ✅ **Unknown server handling** - Hash-based encoding

### Value Propositions (Independent of Accuracy):
- ✅ **Early warning time** - Predict 30min to 8hr ahead
- ✅ **Proactive operations** - Act before incidents happen
- ✅ **Reduced MTTR** - Faster incident response
- ✅ **Capacity planning** - Predict resource needs
- ✅ **Cost optimization** - Right-size infrastructure

---

## Production Deployment Checklist

Before making production accuracy claims:

- [ ] Train model with 20 epochs minimum
- [ ] Use 30+ days of historical data
- [ ] Record final train/validation loss
- [ ] Generate prediction samples vs. actuals
- [ ] Calculate quantile loss metrics
- [ ] Document convergence behavior
- [ ] Test on held-out validation set
- [ ] Validate profile-based predictions
- [ ] Test unknown server handling
- [ ] Measure inference latency

---

## Training Metrics to Capture

### During Training

Record these to a log file:

```python
{
  "epoch": 1,
  "train_loss": 0.234,
  "val_loss": 0.256,
  "learning_rate": 0.01,
  "time_seconds": 1200,
  "gpu_memory_mb": 8500
}
```

### After Training

Save to `training_info.json`:

```python
{
  "trained_at": "2025-10-13T10:30:00",
  "epochs": 20,
  "data_hours": 720,
  "num_servers": 20,
  "final_train_loss": 0.089,
  "final_val_loss": 0.102,
  "quantile_loss": 0.095,
  "training_time_hours": 38.5,
  "model_parameters": 87080,
  "convergence": "achieved",
  "early_stopping": false
}
```

---

## FAQ

### Q: Why not train for 50 or 100 epochs?

**A:** Diminishing returns and overfitting risk. TFT models typically converge by epoch 20. Beyond that, you risk memorizing training data rather than learning patterns.

### Q: Can I use less than 1 week of data?

**A:** Not recommended. TFT needs at least 7 days to learn weekly patterns (weekday vs. weekend). 30 days is ideal for capturing monthly cycles.

### Q: How do I know if my model converged?

**A:** Check if validation loss plateaus. If val_loss stops improving for 3+ epochs, you've likely converged. Early stopping will trigger automatically.

### Q: What if val_loss is much higher than train_loss?

**A:** This indicates overfitting. Solutions:
- Increase dropout (default: 0.15)
- Reduce model complexity
- Add more training data
- Use regularization

### Q: Can I claim accuracy from 1 epoch?

**A:** No. 1 epoch is not converged. You can only claim "proof of concept" or "initial validation."

### Q: What's a good quantile loss value?

**A:** Depends on your data scale, but generally:
- < 0.10: Excellent
- 0.10-0.20: Good
- 0.20-0.30: Acceptable
- > 0.30: Needs improvement

---

## Conclusion

**Key Principle:** Only claim what you can measure.

- 1 epoch = Proof of concept
- 10 epochs = Initial validation
- 20 epochs = Production ready

Be transparent about training maturity, and focus on architectural strengths when accuracy data is incomplete.

---

**Version:** 1.0.0
**Status:** Production Reference
**Maintained By:** Project Team
**Review Frequency:** After each major training run
