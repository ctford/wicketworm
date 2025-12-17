# WicketWorm Model Inference Equations

## Model Type
Multinomial Logistic Regression (3 classes: Win, Draw, Loss)

## Inference Steps

### 1. Standardize Features
For each feature i:
```
z[i] = (x[i] - mean[i]) / std[i]
```

### 2. Calculate Logits
For each outcome class c (0=Win, 1=Draw, 2=Loss):
```
logit[c] = intercept[c] + Σ(coef[c][i] * z[i])
```

### 3. Softmax to Probabilities
```
P(Win)  = exp(logit[0]) / (exp(logit[0]) + exp(logit[1]) + exp(logit[2]))
P(Draw) = exp(logit[1]) / (exp(logit[0]) + exp(logit[1]) + exp(logit[2]))
P(Loss) = exp(logit[2]) / (exp(logit[0]) + exp(logit[1]) + exp(logit[2]))
```

## Features (8 total)
From model.json:
1. **innings** - Current innings (1-4)
2. **wicketsDown** - Wickets fallen
3. **runRate** - Current run rate (runs/over)
4. **lead** - Lead (positive = batting team ahead)
5. **ballsUsed** - Balls bowled this innings
6. **runsPerWicket** - Runs per wicket fallen
7. **isChasing** - 1 if innings 3 or 4, else 0
8. **requiredRunRate** - Required run rate if chasing

## Normalization Parameters
From model.json:
```
featureMeans = [2.189, 3.542, 3.105, 71.089, 318.448, 38.342, 0.374, 0.239]
featureStds  = [1.034, 2.529, 0.879, 284.781, 227.575, 28.376, 0.484, 0.652]
```

## Model Coefficients
From model.json:

### Class 0 (Win)
```
intercept = -0.324
coef = [0.0812, -0.5479, -0.0976, 0.0206, 0.5971, 0.0131, -0.0742, 0.0475]
```

### Class 1 (Draw)
```
intercept = 0.085
coef = [-0.0638, 0.6500, -0.0918, -0.1522, -0.6394, -0.1417, 0.0753, 0.3359]
```

### Class 2 (Loss)
```
intercept = 0.239
coef = [-0.0175, -0.1021, 0.1894, 0.1316, 0.0422, 0.1286, -0.0011, -0.3834]
```

## Example Calculation

Given game state:
- innings=4, wicketsDown=2, runRate=5.87, lead=-0, ballsUsed=210
- runsPerWicket=102.5, isChasing=1, requiredRunRate=0

### Step 1: Standardize
```
z[0] = (4 - 2.189) / 1.034 = 1.751
z[1] = (2 - 3.542) / 2.529 = -0.610
z[2] = (5.87 - 3.105) / 0.879 = 3.146
z[3] = (-0 - 71.089) / 284.781 = -0.250
z[4] = (210 - 318.448) / 227.575 = -0.477
z[5] = (102.5 - 38.342) / 28.376 = 2.261
z[6] = (1 - 0.374) / 0.484 = 1.293
z[7] = (0 - 0.239) / 0.652 = -0.367
```

### Step 2: Calculate Logits
```
logit[Win]  = -0.324 + (0.0812*1.751 + -0.5479*-0.610 + ... + 0.0475*-0.367)
            = -0.324 + 0.949 = 0.625

logit[Draw] = 0.085 + (-0.0638*1.751 + 0.6500*-0.610 + ... + 0.3359*-0.367)
            = 0.085 + (-0.949) = -0.864

logit[Loss] = 0.239 + (-0.0175*1.751 + -0.1021*-0.610 + ... + -0.3834*-0.367)
            = 0.239 + 0.000 = 0.239
```

### Step 3: Softmax
```
exp(0.625) = 1.868
exp(-0.864) = 0.421
exp(0.239) = 1.270

sum = 1.868 + 0.421 + 1.270 = 3.559

P(Win)  = 1.868 / 3.559 = 52.5%
P(Draw) = 0.421 / 3.559 = 11.8%
P(Loss) = 1.270 / 3.559 = 35.7%
```

## Implementation

This can be implemented in any language with basic math:
- 8 multiplications + 1 addition per class (24 total ops)
- 3 exponentials + 1 division for softmax
- No external dependencies needed

Total: ~30 floating point operations per prediction.
