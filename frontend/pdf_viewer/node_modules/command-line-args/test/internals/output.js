'use strict'
const TestRunner = require('test-runner')
const a = require('assert')
const Output = require('../../lib/output')

const runner = new TestRunner()

runner.test('output.toObject(): no defs set', function () {
  const output = new Output([
    { name: 'one' }
  ])
  a.deepStrictEqual(output.toObject(), {})
})

runner.test('output.toObject(): one def set', function () {
  const output = new Output([
    { name: 'one' }
  ])
  const Option = require('../../lib/option')
  const option = Option.create({ name: 'one' })
  option.set('yeah')
  output.set('one', option)
  a.deepStrictEqual(output.toObject(), {
    one: 'yeah'
  })
})
