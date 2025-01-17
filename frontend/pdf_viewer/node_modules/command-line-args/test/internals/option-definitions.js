'use strict'
const TestRunner = require('test-runner')
const a = require('assert')
const Definitions = require('../../lib/option-definitions')

const runner = new TestRunner()

runner.test('.get(long option)', function () {
  const definitions = Definitions.from([ { name: 'one' } ])
  a.strictEqual(definitions.get('--one').name, 'one')
})

runner.test('.get(short option)', function () {
  const definitions = Definitions.from([ { name: 'one', alias: 'o' } ])
  a.strictEqual(definitions.get('-o').name, 'one')
})

runner.test('.get(name)', function () {
  const definitions = Definitions.from([ { name: 'one' } ])
  a.strictEqual(definitions.get('one').name, 'one')
})

runner.test('.validate()', function () {
  a.throws(function () {
    const definitions = new Definitions()
    definitions.load([ { name: 'one' }, { name: 'one' } ])
  })
})
