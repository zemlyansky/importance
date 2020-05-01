function mae (yt, yp) {
  return yt.reduce((a, v, i) => a + Math.abs(v - yp[i]) / yt.length, 0)
}

function mse (yt, yp) {
  return yt.reduce((a, v, i) => a + Math.pow(v - yp[i], 2) / yt.length, 0)
}

function rmse (yt, yp) {
  return Math.sqrt(mse(yt, yp))
}

function smape (yt, yp) {
  const sum = yt.reduce((a, v, i) => a + Math.abs(v - yp[i]) / (Math.abs(v) + Math.abs(yp[i])), 0)
  return (sum / yt.length) * 100
}

function acc (yt, yp) {
  return yt.reduce((a, v, i) => a + (v === yp[i]) / yt.length, 0)
}

function ce (yt, yp) {
  const eps = 1e-10
  const ce = []
  for (let i = 0; i < yt.length; i++) {
    // Limit probs to (eps, 1 - eps)
    const probs = yp[i].map(classProb => classProb < eps ? eps : classProb > (1 - eps) ? 1 - eps : classProb)
    const sum = probs.reduce((p, v) => p + v, 0)
    ce.push(
      probs
        .map(p => p / sum)
        .map((p, j) => Math.log(p) * (yt[i] === j))
        .reduce((a, v) => a + v, 0)
    )
  }

  const res = ce.reduce((a, v) => a + v / yt.length, 0)
  return res
}

const scoreTypes = {
  mae, mse, rmse, smape, acc, ce
}

// Score a model given input X and target y
function score (model, X, y, kind) {
  if (y.length !== X.length) {
    throw new Error('Arrays have different length')
  } else if (!y.length) {
    throw new Error('Zero length array')
  } else if (kind === 'ce') {
    return scoreTypes[kind](y, model.predictProba(X))
  } else if (kind === 'acc') {
    return scoreTypes[kind](y, model.predict(X))
  } else {
    return -scoreTypes[kind](y, model.predict(X))
  }
}

module.exports = score
