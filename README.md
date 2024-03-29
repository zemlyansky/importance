# Permutation feature importance package for browsers and Node.js

### Compute the relative importance of input variables of trained predictive models using feature shuffling

When called, the `importance` function shuffles each feature `n` times and computes the difference between the base score (calculated with original features `X` and target variable `y`) and permuted data. Intuitively that measures how the performance of a model decreases when we "remove" the feature.

- More info about the method: [Permutation Feature Importance](https://christophm.github.io/interpretable-ml-book/feature-importance.html)
- Permutation importance can be biased if features are highly correlated ([Hooker,  Mentch 2019](https://arxiv.org/pdf/1905.03151v1.pdf))

### Usage
```javascript
// Create and train a model first
const rf = new RandomForestRegressor({
  maxDepth: 20,
  nEstimators: 50
})
rf.train(X, y)

// Get feature importance
const imp = importance(rf, X, y, {
  kind: 'mse',
  n: 10,
  means: true,
  verbose: false
})
console.log(impsRF)
```

You can also check `example.js` in this repo that uses the [random-forest](https://github.com/zemlyansky/random-forest) package as a predictive model.

### API
```javascript
importance(model, X, y, options)
```
- `model` - trained model with `predict` method (`predictProba` if cross-entropy used as score)
- `X` - 2D array of features
- `y` - 1D array of target variables

Options:
- `kind` - scoring function (`mse`, `mae`, `rmse`, `smape`, `acc`, `ce` (cross-entropy)
- `n` - number of times each feature is shuffled. 
- `means` - if `true` returns only average importance
- `verbose` - if `true` throws some info into console

### Feature selection
Feature importance is often used for variable selection. Permutation-based importance is a good method for that goal, but if you need more robust selection method check [boruta.js](https://www.npmjs.com/package/boruta)

### Web demo
The `importance` package is used for feature selection on [StatSim Select](https://statsim.com/select) and for data visualization on [StatSim Vis](https://statsim.com/vis) 
