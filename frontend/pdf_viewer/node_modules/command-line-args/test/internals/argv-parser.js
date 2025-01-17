'use strict'
const TestRunner = require('test-runner')
const a = require('assert')
const runner = new TestRunner()
const ArgvParser = require('../../lib/argv-parser')

runner.test('argv-parser: long option, string', function () {
  const optionDefinitions = [
    { name: 'one' }
  ]
  const argv = [ '--one', '1' ]
  const parser = new ArgvParser(optionDefinitions, { argv })
  const result = Array.from(parser)
  a.ok(result[0].def)
  a.ok(result[1].def)
  result.forEach(r => delete r.def)
  a.deepStrictEqual(result, [
    { event: 'set', arg: '--one', name: 'one', value: null },
    { event: 'set', arg: '1', name: 'one', value: '1' }
  ])
})

runner.test('argv-parser: long option, string repeated', function () {
  const optionDefinitions = [
    { name: 'one' }
  ]
  const argv = [ '--one', '1', '--one', '2' ]
  const parser = new ArgvParser(optionDefinitions, { argv })
  const result = Array.from(parser)
  a.ok(result[0].def)
  a.ok(result[1].def)
  a.ok(result[2].def)
  a.ok(result[3].def)
  result.forEach(r => delete r.def)
  a.deepStrictEqual(result, [
    { event: 'set', arg: '--one', name: 'one', value: null },
    { event: 'set', arg: '1', name: 'one', value: '1' },
    { event: 'set', arg: '--one', name: 'one', value: null },
    { event: 'set', arg: '2', name: 'one', value: '2' }
  ])
})

runner.test('argv-parser: long option, string multiple', function () {
  const optionDefinitions = [
    { name: 'one', multiple: true }
  ]
  const argv = [ '--one', '1', '2' ]
  const parser = new ArgvParser(optionDefinitions, { argv })
  const result = Array.from(parser)
  a.ok(result[0].def)
  a.ok(result[1].def)
  a.ok(result[2].def)
  result.forEach(r => delete r.def)
  a.deepStrictEqual(result, [
    { event: 'set', arg: '--one', name: 'one', value: null },
    { event: 'set', arg: '1', name: 'one', value: '1' },
    { event: 'set', arg: '2', name: 'one', value: '2' }
  ])
})

runner.test('argv-parser: long option, string multiple then boolean', function () {
  const optionDefinitions = [
    { name: 'one', multiple: true },
    { name: 'two', type: Boolean }
  ]
  const argv = [ '--one', '1', '2', '--two' ]
  const parser = new ArgvParser(optionDefinitions, { argv })
  const result = Array.from(parser)
  a.ok(result[0].def)
  a.ok(result[1].def)
  a.ok(result[2].def)
  a.ok(result[3].def)
  result.forEach(r => delete r.def)
  a.deepStrictEqual(result, [
    { event: 'set', arg: '--one', name: 'one', value: null },
    { event: 'set', arg: '1', name: 'one', value: '1' },
    { event: 'set', arg: '2', name: 'one', value: '2' },
    { event: 'set', arg: '--two', name: 'two', value: true }
  ])
})

runner.test('argv-parser: long option, boolean', function () {
  const optionDefinitions = [
    { name: 'one', type: Boolean }
  ]
  const argv = [ '--one', '1' ]
  const parser = new ArgvParser(optionDefinitions, { argv })
  const result = Array.from(parser)
  a.ok(result[0].def)
  a.ok(!result[1].def)
  result.forEach(r => delete r.def)
  a.deepStrictEqual(result, [
    { event: 'set', arg: '--one', name: 'one', value: true },
    { event: 'unknown_value', arg: '1', name: '_unknown', value: undefined }
  ])
})

runner.test('argv-parser: simple, with unknown values', function () {
  const optionDefinitions = [
    { name: 'one', type: Number }
  ]
  const argv = [ 'clive', '--one', '1', 'yeah' ]
  const parser = new ArgvParser(optionDefinitions, { argv })
  const result = Array.from(parser)
  a.ok(!result[0].def)
  a.ok(result[1].def)
  a.ok(result[2].def)
  a.ok(!result[3].def)
  result.forEach(r => delete r.def)
  a.deepStrictEqual(result, [
    { event: 'unknown_value', arg: 'clive', name: '_unknown', value: undefined },
    { event: 'set', arg: '--one', name: 'one', value: null },
    { event: 'set', arg: '1', name: 'one', value: '1' },
    { event: 'unknown_value', arg: 'yeah', name: '_unknown', value: undefined }
  ])
})

runner.test('argv-parser: simple, with singular defaultOption', function () {
  const optionDefinitions = [
    { name: 'one', type: Number },
    { name: 'two', defaultOption: true }
  ]
  const argv = [ 'clive', '--one', '1', 'yeah' ]
  const parser = new ArgvParser(optionDefinitions, { argv })
  const result = Array.from(parser)
  a.ok(result[0].def)
  a.ok(result[1].def)
  a.ok(result[2].def)
  a.ok(!result[3].def)
  result.forEach(r => delete r.def)
  a.deepStrictEqual(result, [
    { event: 'set', arg: 'clive', name: 'two', value: 'clive' },
    { event: 'set', arg: '--one', name: 'one', value: null },
    { event: 'set', arg: '1', name: 'one', value: '1' },
    { event: 'unknown_value', arg: 'yeah', name: '_unknown', value: undefined }
  ])
})

runner.test('argv-parser: simple, with multiple defaultOption', function () {
  const optionDefinitions = [
    { name: 'one', type: Number },
    { name: 'two', defaultOption: true, multiple: true }
  ]
  const argv = [ 'clive', '--one', '1', 'yeah' ]
  const parser = new ArgvParser(optionDefinitions, { argv })
  const result = Array.from(parser)
  a.ok(result[0].def)
  a.ok(result[1].def)
  a.ok(result[2].def)
  a.ok(result[3].def)
  result.forEach(r => delete r.def)
  a.deepStrictEqual(result, [
    { event: 'set', arg: 'clive', name: 'two', value: 'clive' },
    { event: 'set', arg: '--one', name: 'one', value: null },
    { event: 'set', arg: '1', name: 'one', value: '1' },
    { event: 'set', arg: 'yeah', name: 'two', value: 'yeah' }
  ])
})

runner.test('argv-parser: long option, string lazyMultiple bad', function () {
  const optionDefinitions = [
    { name: 'one', lazyMultiple: true }
  ]
  const argv = [ '--one', '1', '2' ]
  const parser = new ArgvParser(optionDefinitions, { argv })
  const result = Array.from(parser)
  a.ok(result[0].def)
  a.ok(result[1].def)
  a.ok(!result[2].def)
  result.forEach(r => delete r.def)
  a.deepStrictEqual(result, [
    { event: 'set', arg: '--one', name: 'one', value: null },
    { event: 'set', arg: '1', name: 'one', value: '1' },
    { event: 'unknown_value', arg: '2', name: '_unknown', value: undefined }
  ])
})

runner.test('argv-parser: long option, string lazyMultiple good', function () {
  const optionDefinitions = [
    { name: 'one', lazyMultiple: true }
  ]
  const argv = [ '--one', '1', '--one', '2' ]
  const parser = new ArgvParser(optionDefinitions, { argv })
  const result = Array.from(parser)
  a.ok(result[0].def)
  a.ok(result[1].def)
  a.ok(result[2].def)
  a.ok(result[3].def)
  result.forEach(r => delete r.def)
  a.deepStrictEqual(result, [
    { event: 'set', arg: '--one', name: 'one', value: null },
    { event: 'set', arg: '1', name: 'one', value: '1' },
    { event: 'set', arg: '--one', name: 'one', value: null },
    { event: 'set', arg: '2', name: 'one', value: '2' }
  ])
})

runner.test('argv-parser: long option, stopAtFirstUnknown', function () {
  const optionDefinitions = [
    { name: 'one' },
    { name: 'two' }
  ]
  const argv = [ '--one', '1', 'asdf', '--two', '2' ]
  const parser = new ArgvParser(optionDefinitions, { argv, stopAtFirstUnknown: true })
  const result = Array.from(parser)
  a.ok(result[0].def)
  a.ok(result[1].def)
  a.ok(!result[2].def)
  a.ok(!result[3].def)
  a.ok(!result[4].def)
  result.forEach(r => delete r.def)
  a.deepStrictEqual(result, [
    { event: 'set', arg: '--one', name: 'one', value: null },
    { event: 'set', arg: '1', name: 'one', value: '1' },
    { event: 'unknown_value', arg: 'asdf', name: '_unknown', value: undefined },
    { event: 'unknown_value', arg: '--two', name: '_unknown', value: undefined },
    { event: 'unknown_value', arg: '2', name: '_unknown', value: undefined }
  ])
})

runner.test('argv-parser: long option, stopAtFirstUnknown with defaultOption', function () {
  const optionDefinitions = [
    { name: 'one', defaultOption: true },
    { name: 'two' }
  ]
  const argv = [ '1', 'asdf', '--two', '2' ]
  const parser = new ArgvParser(optionDefinitions, { argv, stopAtFirstUnknown: true })
  const result = Array.from(parser)
  a.ok(result[0].def)
  a.ok(!result[1].def)
  a.ok(!result[2].def)
  a.ok(!result[3].def)
  result.forEach(r => delete r.def)
  a.deepStrictEqual(result, [
    { event: 'set', arg: '1', name: 'one', value: '1' },
    { event: 'unknown_value', arg: 'asdf', name: '_unknown', value: undefined },
    { event: 'unknown_value', arg: '--two', name: '_unknown', value: undefined },
    { event: 'unknown_value', arg: '2', name: '_unknown', value: undefined }
  ])
})

runner.test('argv-parser: long option, stopAtFirstUnknown with defaultOption 2', function () {
  const optionDefinitions = [
    { name: 'one', defaultOption: true },
    { name: 'two' }
  ]
  const argv = [ '--one', '1', '--', '--two', '2' ]
  const parser = new ArgvParser(optionDefinitions, { argv, stopAtFirstUnknown: true })
  const result = Array.from(parser)
  a.ok(result[0].def)
  a.ok(result[1].def)
  a.ok(!result[2].def)
  a.ok(!result[3].def)
  a.ok(!result[4].def)
  result.forEach(r => delete r.def)
  a.deepStrictEqual(result, [
    { event: 'set', arg: '--one', name: 'one', value: null },
    { event: 'set', arg: '1', name: 'one', value: '1' },
    { event: 'unknown_value', arg: '--', name: '_unknown', value: undefined },
    { event: 'unknown_value', arg: '--two', name: '_unknown', value: undefined },
    { event: 'unknown_value', arg: '2', name: '_unknown', value: undefined }
  ])
})

runner.test('argv-parser: --option=value', function () {
  const optionDefinitions = [
    { name: 'one' },
    { name: 'two' }
  ]
  const argv = [ '--one=1', '--two=2', '--two=' ]
  const parser = new ArgvParser(optionDefinitions, { argv })
  const result = Array.from(parser)
  a.ok(result[0].def)
  a.ok(result[1].def)
  a.ok(result[2].def)
  result.forEach(r => delete r.def)
  a.deepStrictEqual(result, [
    { event: 'set', arg: '--one=1', name: 'one', value: '1' },
    { event: 'set', arg: '--two=2', name: 'two', value: '2' },
    { event: 'set', arg: '--two=', name: 'two', value: '' }
  ])
})

runner.test('argv-parser: --option=value, unknown option', function () {
  const optionDefinitions = [
    { name: 'one' }
  ]
  const argv = [ '--three=3' ]
  const parser = new ArgvParser(optionDefinitions, { argv })
  const result = Array.from(parser)
  a.ok(!result[0].def)
  result.forEach(r => delete r.def)
  a.deepStrictEqual(result, [
    { event: 'unknown_option', arg: '--three=3', name: '_unknown', value: undefined }
  ])
})

runner.test('argv-parser: --option=value where option is boolean', function () {
  const optionDefinitions = [
    { name: 'one', type: Boolean }
  ]
  const argv = [ '--one=1' ]
  const parser = new ArgvParser(optionDefinitions, { argv })
  const result = Array.from(parser)
  a.ok(result[0].def)
  a.ok(result[1].def)
  result.forEach(r => delete r.def)
  a.deepStrictEqual(result, [
    { event: 'unknown_value', arg: '--one=1', name: '_unknown', value: undefined },
    { event: 'set', arg: '--one=1', name: 'one', value: true }
  ])
})

runner.test('argv-parser: short option, string', function () {
  const optionDefinitions = [
    { name: 'one', alias: 'o' }
  ]
  const argv = [ '-o', '1' ]
  const parser = new ArgvParser(optionDefinitions, { argv })
  const result = Array.from(parser)
  a.ok(result[0].def)
  a.ok(result[1].def)
  result.forEach(r => delete r.def)
  a.deepStrictEqual(result, [
    { event: 'set', arg: '-o', name: 'one', value: null },
    { event: 'set', arg: '1', name: 'one', value: '1' }
  ])
})

runner.test('argv-parser: combined short option, string', function () {
  const optionDefinitions = [
    { name: 'one', alias: 'o' },
    { name: 'two', alias: 't' }
  ]
  const argv = [ '-ot', '1' ]
  const parser = new ArgvParser(optionDefinitions, { argv })
  const result = Array.from(parser)
  a.ok(result[0].def)
  a.ok(result[1].def)
  a.ok(result[2].def)
  result.forEach(r => delete r.def)
  a.deepStrictEqual(result, [
    { event: 'set', arg: '-ot', subArg: '-o', name: 'one', value: null },
    { event: 'set', arg: '-ot', subArg: '-t', name: 'two', value: null },
    { event: 'set', arg: '1', name: 'two', value: '1' }
  ])
})

runner.test('argv-parser: combined short option, one unknown', function () {
  const optionDefinitions = [
    { name: 'one', alias: 'o' },
    { name: 'two', alias: 't' }
  ]
  const argv = [ '-xt', '1' ]
  const parser = new ArgvParser(optionDefinitions, { argv })
  const result = Array.from(parser)
  a.ok(!result[0].def)
  a.ok(result[1].def)
  a.ok(result[2].def)
  result.forEach(r => delete r.def)
  a.deepStrictEqual(result, [
    { event: 'unknown_option', arg: '-xt', subArg: '-x', name: '_unknown', value: undefined },
    { event: 'set', arg: '-xt', subArg: '-t', name: 'two', value: null },
    { event: 'set', arg: '1', name: 'two', value: '1' }
  ])
})
