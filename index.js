var score = require('./score')

function permutationScores (model, X, y, kind, id, nRepeats) {
  const scores = []
  for (let r = 0; r < nRepeats; r++) {
    const Xclone = JSON.parse(JSON.stringify(X))
    for (let i = Xclone.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[Xclone[i][id], Xclone[j][id]] = [Xclone[j][id], Xclone[i][id]]
    }
    scores.push(score(model, Xclone, y, kind))
  }
  return scores
}

module.exports = function importance (model, X, y, opts = {}) {
  const log = opts.verbose ? console.log : () => {}
  const kind = opts.kind ? opts.kind : (Array.from(new Set(y)).length / y.length > 0.5) ? 'mae' : 'ce'
  const nRepeats = opts.n || 1
  const onlyMeans = opts.means || (nRepeats === 1)
  const baseScore = score(model, X, y, kind)
  const nFeatures = X[0].length
  log('Start feature importance')
  log('Score: %s, N repeats: %d, N features: %d', kind, nRepeats, nFeatures)
  log('Base score:', baseScore)

  const importances = []
  for (let i = 0; i < nFeatures; i++) {
    const imp = permutationScores(model, X, y, kind, i, nRepeats).map(score => baseScore - score)
    log(' - computing importance of feature: %d  ->  %f', i, imp.reduce((a, v) => a + v / imp.length, 0))
    importances.push(imp)
  }

  // if (opts.scale) { }

  const importancesMeans = importances.map(imps => imps.reduce((a, v) => a + v / nRepeats))
  const importancesStds = importances.map((imps, i) => {
    const std = Math.sqrt(imps.reduce((a, v) => a + Math.pow(v - importancesMeans[i], 2) / nRepeats, 0))
    return std
  })

  return onlyMeans ? importancesMeans : {
    importances,
    importancesMeans,
    importancesStds
  }
}
