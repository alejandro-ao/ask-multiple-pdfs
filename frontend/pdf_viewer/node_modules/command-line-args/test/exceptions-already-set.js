'use strict'
const TestRunner = require('test-runner')
const commandLineArgs = require('../')
const a = require('assert')

const runner = new TestRunner()

runner.test('exceptions-already-set: long option', function () {
  const optionDefinitions = [
    { name: 'one', type: Boolean }
  ]
  const argv = [ '--one', '--one' ]
  a.throws(
    () => commandLineArgs(optionDefinitions, { argv }),
    err => err.name === 'ALREADY_SET' && err.optionName === 'one'
  )
})

runner.test('exceptions-already-set: short option', function () {
  const optionDefinitions = [
    { name: 'one', type: Boolean, alias: 'o' }
  ]
  const argv = [ '--one', '-o' ]
  a.throws(
    () => commandLineArgs(optionDefinitions, { argv }),
    err => err.name === 'ALREADY_SET' && err.optionName === 'one'
  )
})

runner.test('exceptions-already-set: --option=value', function () {
  const optionDefinitions = [
    { name: 'one' }
  ]
  const argv = [ '--one=1', '--one=1' ]
  a.throws(
    () => commandLineArgs(optionDefinitions, { argv }),
    err => err.name === 'ALREADY_SET' && err.optionName === 'one'
  )
})

runner.test('exceptions-already-set: combined short option', function () {
  const optionDefinitions = [
    { name: 'one', type: Boolean, alias: 'o' }
  ]
  const argv = [ '-oo' ]
  a.throws(
    () => commandLineArgs(optionDefinitions, { argv }),
    err => err.name === 'ALREADY_SET' && err.optionName === 'one'
  )
})
