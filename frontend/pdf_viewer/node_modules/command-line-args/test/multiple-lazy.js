'use strict'
const TestRunner = require('test-runner')
const commandLineArgs = require('../')
const a = require('assert')

const runner = new TestRunner()

runner.test('lazy multiple: string', function () {
  const argv = ['--one', 'a', '--one', 'b', '--one', 'd']
  const optionDefinitions = [
      { name: 'one', lazyMultiple: true }
  ]
  const result = commandLineArgs(optionDefinitions, { argv })
  a.deepStrictEqual(result, {
    one: ['a', 'b', 'd']
  })
})

runner.test('lazy multiple: string unset with defaultValue', function () {
  const optionDefinitions = [
    { name: 'one', lazyMultiple: true, defaultValue: 1 }
  ]
  const argv = []
  const result = commandLineArgs(optionDefinitions, { argv })
  a.deepStrictEqual(result, { one: [ 1 ] })
})

runner.test('lazy multiple: string, --option=value', function () {
  const optionDefinitions = [
    { name: 'one', lazyMultiple: true }
  ]
  const argv = [ '--one=1', '--one=2' ]
  const result = commandLineArgs(optionDefinitions, { argv })
  a.deepStrictEqual(result, {
    one: [ '1', '2' ]
  })
})

runner.test('lazy multiple: string, --option=value mix', function () {
  const optionDefinitions = [
    { name: 'one', lazyMultiple: true }
  ]
  const argv = [ '--one=1', '--one=2', '--one', '3' ]
  const result = commandLineArgs(optionDefinitions, { argv })
  a.deepStrictEqual(result, {
    one: [ '1', '2', '3' ]
  })
})

runner.test('lazy multiple: string, defaultOption', function () {
  const optionDefinitions = [
    { name: 'one', lazyMultiple: true, defaultOption: true }
  ]
  const argv = [ '1', '2' ]
  const result = commandLineArgs(optionDefinitions, { argv })
  a.deepStrictEqual(result, {
    one: [ '1', '2' ]
  })
})

runner.test('lazy multiple: greedy style, string', function () {
  const optionDefinitions = [
    { name: 'one', lazyMultiple: true }
  ]
  const argv = [ '--one', '1', '2' ]
  a.throws(
    () => commandLineArgs(optionDefinitions, { argv }),
    err => err.name === 'UNKNOWN_VALUE' && err.value === '2'
  )
})

runner.test('lazy multiple: greedy style, string, --option=value', function () {
  const optionDefinitions = [
    { name: 'one', lazyMultiple: true }
  ]
  const argv = [ '--one=1', '--one=2' ]
  const result = commandLineArgs(optionDefinitions, { argv })
  a.deepStrictEqual(result, {
    one: [ '1', '2' ]
  })
})

runner.test('lazy multiple: greedy style, string, --option=value mix', function () {
  const optionDefinitions = [
    { name: 'one', lazyMultiple: true }
  ]
  const argv = [ '--one=1', '--one=2', '3' ]
  a.throws(
    () => commandLineArgs(optionDefinitions, { argv }),
    err => err.name === 'UNKNOWN_VALUE' && err.value === '3'
  )
})
