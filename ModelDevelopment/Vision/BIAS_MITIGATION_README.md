# Bias Mitigation Implementation

This document describes the bias mitigation functionality added to the MedScan AI vision pipeline.

## Overview

After bias is detected in model predictions, the system can automatically apply mitigation techniques and re-evaluate bias to generate comparison reports.

## Features

1. **Automatic Bias Mitigation**: When bias is detected, the system can automatically apply post-processing mitigation techniques
2. **Threshold Optimization**: Uses Fairlearn's `ThresholdOptimizer` to adjust decision thresholds per demographic group
3. **Re-evaluation**: After mitigation, bias detection is re-run to measure improvement
4. **Comparison Reports**: Generates HTML reports comparing bias metrics before and after mitigation

## How It Works

### 1. Bias Detection
- The system first runs standard bias detection using `BiasDetector.detect_bias()`
- This identifies performance disparities across demographic groups (Gender, Age_Group, etc.)

### 2. Mitigation Application
- If bias is detected and mitigation is enabled, `BiasDetector.detect_and_mitigate_bias()` is called
- For each sensitive feature with detected bias:
  - Model predictions (probabilities) are extracted
  - `ThresholdOptimizer` from Fairlearn is applied to adjust thresholds per group
  - Mitigated predictions are generated

### 3. Re-evaluation
- Bias detection is re-run using the mitigated predictions
- Fairness metrics are recomputed (demographic parity, equalized odds, etc.)

### 4. Report Generation
- A comparison report is generated showing:
  - Overall performance metrics (accuracy, precision, recall, F1) before/after
  - Fairness metrics (demographic parity, equalized odds) before/after
  - Per-group performance comparison
  - Bias detection summary

## Configuration

To enable bias mitigation, add the following to your training configuration YAML:

```yaml
bias_detection:
  enabled: true
  mitigation_enabled: true  # Enable automatic mitigation
  slicing_features: ['Gender', 'Age_Group']
  performance_threshold: 0.1  # 10% performance difference threshold
  demographic_parity_threshold: 0.1
  equalized_odds_threshold: 0.1

bias_mitigation:
  method: 'threshold_optimizer'  # Currently only threshold_optimizer is supported
  constraints: ['equalized_odds']  # or ['demographic_parity']
```

## Usage

### In Training Scripts

The bias mitigation is automatically applied when:
1. `bias_detection.enabled: true`
2. `bias_detection.mitigation_enabled: true`
3. Bias is detected in the initial analysis

### Manual Usage

You can also use the mitigation module directly:

```python
from bias_mitigation import BiasMitigator

mitigator = BiasMitigator(config=config)
mitigated_preds, mitigation_info = mitigator.mitigate_bias(
    y_true=labels,
    y_pred_proba=predictions,
    sensitive_features=gender_series,
    feature_name='Gender'
)
```

## Output Files

After mitigation, the following files are generated:

1. **Original Bias Report**: `bias_report_{dataset}_{timestamp}.json` and `.html`
2. **Mitigated Bias Report**: `mitigated/bias_report_{dataset}_{timestamp}.json` and `.html`
3. **Comparison Report**: `bias_mitigation_comparison_{timestamp}.html`

All reports are saved in: `{output_dir}/bias_reports/{dataset_name}/`

## Mitigation Techniques

### Threshold Optimization (Current Implementation)

- **Method**: Post-processing using Fairlearn's `ThresholdOptimizer`
- **How it works**: Adjusts decision thresholds per demographic group to satisfy fairness constraints
- **Constraints supported**:
  - `equalized_odds`: Equalizes true positive rate and false positive rate across groups
  - `demographic_parity`: Equalizes selection rate (positive prediction rate) across groups
- **Pros**: 
  - No retraining required
  - Fast to apply
  - Preserves model performance while improving fairness
- **Cons**:
  - Only adjusts thresholds, doesn't change model behavior
  - May slightly reduce overall accuracy to improve fairness

## Limitations

1. **Multi-class Support**: Currently optimized for binary classification. Multi-class support is limited.
2. **Single Constraint**: Applies one fairness constraint at a time (equalized_odds or demographic_parity)
3. **Post-processing Only**: Current implementation only supports post-processing mitigation. Pre-processing (data balancing) and in-processing (algorithm modifications) are not yet implemented.

## Future Enhancements

1. **Multiple Constraints**: Support for applying multiple fairness constraints simultaneously
2. **Pre-processing Mitigation**: Data balancing, resampling, and augmentation
3. **In-processing Mitigation**: Fairness-aware training algorithms
4. **Multi-class Support**: Full support for multi-class classification scenarios
5. **Custom Mitigation Strategies**: Allow users to define custom mitigation approaches

## Dependencies

- `fairlearn`: Required for threshold optimization
- `scikit-learn`: For metrics computation
- `pandas`, `numpy`: For data handling

## Example Output

The comparison report shows metrics like:

- **Before Mitigation**: 
  - Accuracy: 0.8500
  - Gender performance difference: 0.2222 (Male: 77.78%, Female: 90%)
  - Bias detected: Yes

- **After Mitigation**:
  - Accuracy: 0.8500 (or slightly different)
  - Gender performance difference: 0.1000 (improved)
  - Bias detected: No (or reduced)

## Troubleshooting

1. **Fairlearn not available**: Install with `pip install fairlearn`
2. **Mitigation not applied**: Check that `mitigation_enabled: true` in config
3. **No improvement**: Threshold optimization may not always improve all metrics. Check the comparison report for details.

