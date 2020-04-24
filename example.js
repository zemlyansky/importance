// const { RandomForestRegression } = require('random-forest')
const { RandomForestRegressor } = require('random-forest')
const LinReg = require('ml-regression-multivariate-linear')
const fs = require('fs')
const parse = require('csv-parse/lib/sync')
const bar = require('bar-charts')

const importance = require('.') // Change to require('importance') if installed from npm

// Load data
const file = fs.readFileSync('./friedman1.csv', 'utf8')
const records = parse(file, { columns: false })
const features = records.shift().slice(1, -1)
const X = records.map(r => r.slice(1, -1))
const y = records.map(r => r[r.length - 1])

// Leave the first sample for testing if models work
const Xt = X.shift()
const yt = y.shift()

// Train a linear model
const lm = new LinReg(X, y.map(v => [v]))
console.log('Lin reg pred:', lm.predict([Xt]))

// Train a random forest
const rf = new RandomForestRegressor({
  maxDepth: 20,
  nEstimators: 50
})
rf.train(X, y)
console.log('Random forest pred:', rf.predict(Xt))

console.log('True Y:', yt)
const opts = { kind: 'mse', n: 10, means: true, verbose: false }

const impsLM = importance(lm, X, y, opts)
console.log('LM', impsLM)

const impsRF = importance(rf, X, y, opts)
console.log('RF', impsRF)

// process.exit()

