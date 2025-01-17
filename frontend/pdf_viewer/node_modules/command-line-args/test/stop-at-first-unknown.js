'use strict'
const TestRunner = require('test-runner')
const a = require('assert')
const commandLineArgs = require('../')

const runner = new TestRunner()

runner.test('stopAtFirstUnknown', function () {
  const optionDefinitions = [
    { name: 'one', type: Boolean },
    { name: 'two', type: Boolean }
  ]
  const argv = [ '--one', 'a', '--two' ]
  const result = commandLineArgs(optionDefinitions, { argv, stopAtFirstUnknown: true, partial: true })
  a.deepStrictEqual(result, {
    one: true,
    _unknown: [ 'a', '--two' ]
  })
})

runner.test('stopAtFirstUnknown: with a singlular defaultOption', function () {
  const optionDefinitions = [
    { name: 'one', defaultOption: true },
    { name: 'two' }
  ]
  const argv = [ '--one', '1', '--', '--two', '2' ]
  const result = commandLineArgs(optionDefinitions, { argv, stopAtFirstUnknown: true })
  a.deepStrictEqual(result, {
    one: '1',
    _unknown: [ '--', '--two', '2' ]
  })
})

runner.test('stopAtFirstUnknown: with a singlular defaultOption and partial', function () {
  const optionDefinitions = [
    { name: 'one', defaultOption: true },
    { name: 'two' }
  ]
  const argv = [ '--one', '1', '--', '--two', '2' ]
  const result = commandLineArgs(optionDefinitions, { argv, stopAtFirstUnknown: true, partial: true })
  a.deepStrictEqual(result, {
    one: '1',
    _unknown: [ '--', '--two', '2' ]
  })
})
