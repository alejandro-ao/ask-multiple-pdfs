'use strict'
const TestRunner = require('test-runner')
const FlagOption = require('../../lib/option-flag')
const a = require('assert')

const runner = new TestRunner()

runner.test('type-boolean: single set', function () {
  const option = new FlagOption({ name: 'one', type: Boolean })

  option.set(undefined)
  a.strictEqual(option.get(), true)
})

runner.test('type-boolean: single set 2', function () {
  const option = new FlagOption({ name: 'one', type: Boolean })

  option.set('true')
  a.strictEqual(option.get(), true)
})

runner.test('type-boolean: set twice', function () {
  const option = new FlagOption({ name: 'one', type: Boolean })

  option.set(undefined)
  a.strictEqual(option.get(), true)
  a.throws(
    () => option.set('true'),
    err => err.name === 'ALREADY_SET'
  )
})

const origBoolean = Boolean

/* test in contexts which override the standard global Boolean constructor */
runner.test('type-boolean: global Boolean overridden', function () {
  function Boolean () {
    return origBoolean.apply(origBoolean, arguments)
  }

  const option = new FlagOption({ name: 'one', type: Boolean })

  option.set()
  a.strictEqual(option.get(), true)
})

runner.test('type-boolean-multiple: 1', function () {
  const option = new FlagOption({ name: 'one', type: Boolean, multiple: true })

  option.set(undefined)
  option.set(undefined)
  option.set(undefined)

  a.deepStrictEqual(option.get(), [ true, true, true ])
})
