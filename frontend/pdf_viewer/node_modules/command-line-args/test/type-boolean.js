'use strict'
const TestRunner = require('test-runner')
const commandLineArgs = require('../')
const a = require('assert')

const runner = new TestRunner()

runner.test('type-boolean: simple', function () {
  const optionDefinitions = [
    { name: 'one', type: Boolean }
  ]

  a.deepStrictEqual(
    commandLineArgs(optionDefinitions, { argv: [ '--one' ] }),
    { one: true }
  )
})

const origBoolean = Boolean

/* test in contexts which override the standard global Boolean constructor */
runner.test('type-boolean: global Boolean overridden', function () {
  function Boolean () {
    return origBoolean.apply(origBoolean, arguments)
  }

  const optionDefinitions = [
    { name: 'one', type: Boolean }
  ]

  a.deepStrictEqual(
    commandLineArgs(optionDefinitions, { argv: [ '--one' ] }),
    { one: true }
  )
})

runner.test('type-boolean-multiple: 1', function () {
  const optionDefinitions = [
    { name: 'array', type: Boolean, multiple: true }
  ]
  const argv = [ '--array', '--array', '--array' ]
  const result = commandLineArgs(optionDefinitions, { argv })
  a.deepStrictEqual(result, {
    array: [ true, true, true ]
  })
})
