'use strict'
const TestRunner = require('test-runner')
const commandLineArgs = require('../')
const a = require('assert')

const runner = new TestRunner()

runner.test('alias: one string alias', function () {
  const optionDefinitions = [
    { name: 'verbose', alias: 'v' }
  ]
  const argv = [ '-v' ]
  a.deepStrictEqual(commandLineArgs(optionDefinitions, { argv }), {
    verbose: null
  })
})

runner.test('alias: one boolean alias', function () {
  const optionDefinitions = [
    { name: 'dry-run', alias: 'd', type: Boolean }
  ]
  const argv = [ '-d' ]
  a.deepStrictEqual(commandLineArgs(optionDefinitions, { argv }), {
    'dry-run': true
  })
})

runner.test('alias: one boolean, one string', function () {
  const optionDefinitions = [
    { name: 'verbose', alias: 'v', type: Boolean },
    { name: 'colour', alias: 'c' }
  ]
  const argv = [ '-v', '-c' ]
  a.deepStrictEqual(commandLineArgs(optionDefinitions, { argv }), {
    verbose: true,
    colour: null
  })
})
