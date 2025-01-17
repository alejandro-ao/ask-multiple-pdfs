'use strict'
const TestRunner = require('test-runner')
const commandLineArgs = require('../')
const a = require('assert')

const runner = new TestRunner()

runner.test('multiple: empty argv', function () {
  const optionDefinitions = [
    { name: 'one', multiple: true }
  ]
  const argv = []
  const result = commandLineArgs(optionDefinitions, { argv })
  a.deepStrictEqual(result, {})
})

runner.test('multiple: boolean, empty argv', function () {
  const optionDefinitions = [
    { name: 'one', type: Boolean, multiple: true }
  ]
  const argv = []
  const result = commandLineArgs(optionDefinitions, { argv })
  a.deepStrictEqual(result, { })
})

runner.test('multiple: string unset with defaultValue', function () {
  const optionDefinitions = [
    { name: 'one', multiple: true, defaultValue: 1 }
  ]
  const argv = []
  const result = commandLineArgs(optionDefinitions, { argv })
  a.deepStrictEqual(result, { one: [ 1 ] })
})

runner.test('multiple: string', function () {
  const optionDefinitions = [
    { name: 'one', multiple: true }
  ]
  const argv = [ '--one', '1', '2' ]
  const result = commandLineArgs(optionDefinitions, { argv })
  a.deepStrictEqual(result, {
    one: [ '1', '2' ]
  })
})

runner.test('multiple: string, --option=value', function () {
  const optionDefinitions = [
    { name: 'one', multiple: true }
  ]
  const argv = [ '--one=1', '--one=2' ]
  const result = commandLineArgs(optionDefinitions, { argv })
  a.deepStrictEqual(result, {
    one: [ '1', '2' ]
  })
})

runner.test('multiple: string, --option=value mix', function () {
  const optionDefinitions = [
    { name: 'one', multiple: true }
  ]
  const argv = [ '--one=1', '--one=2', '3' ]
  const result = commandLineArgs(optionDefinitions, { argv })
  a.deepStrictEqual(result, {
    one: [ '1', '2', '3' ]
  })
})

runner.test('multiple: string, defaultOption', function () {
  const optionDefinitions = [
    { name: 'one', multiple: true, defaultOption: true }
  ]
  const argv = [ '1', '2' ]
  const result = commandLineArgs(optionDefinitions, { argv })
  a.deepStrictEqual(result, {
    one: [ '1', '2' ]
  })
})
