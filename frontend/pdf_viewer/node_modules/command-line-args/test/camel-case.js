'use strict'
const TestRunner = require('test-runner')
const commandLineArgs = require('../')
const a = require('assert')

const runner = new TestRunner()

runner.test('camel-case: regular', function () {
  const optionDefinitions = [
    { name: 'one-two' },
    { name: 'three', type: Boolean }
  ]
  const argv = [ '--one-two', '1', '--three' ]
  const result = commandLineArgs(optionDefinitions, { argv, camelCase: true })
  a.deepStrictEqual(result, {
    oneTwo: '1',
    three: true
  })
})

runner.test('camel-case: grouped', function () {
  const optionDefinitions = [
    { name: 'one-one', group: 'a' },
    { name: 'two-two', group: 'a' },
    { name: 'three-three', group: 'b', type: Boolean },
    { name: 'four-four' }
  ]
  const argv = [ '--one-one', '1', '--two-two', '2', '--three-three', '--four-four', '4' ]
  const result = commandLineArgs(optionDefinitions, { argv, camelCase: true })
  a.deepStrictEqual(result, {
    a: {
      oneOne: '1',
      twoTwo: '2'
    },
    b: {
      threeThree: true
    },
    _all: {
      oneOne: '1',
      twoTwo: '2',
      threeThree: true,
      fourFour: '4'
    },
    _none: {
      fourFour: '4'
    }
  })
})

runner.test('camel-case: grouped with unknowns', function () {
  const optionDefinitions = [
    { name: 'one-one', group: 'a' },
    { name: 'two-two', group: 'a' },
    { name: 'three-three', group: 'b', type: Boolean },
    { name: 'four-four' }
  ]
  const argv = [ '--one-one', '1', '--two-two', '2', '--three-three', '--four-four', '4', '--five' ]
  const result = commandLineArgs(optionDefinitions, { argv, camelCase: true, partial: true })
  a.deepStrictEqual(result, {
    a: {
      oneOne: '1',
      twoTwo: '2'
    },
    b: {
      threeThree: true
    },
    _all: {
      oneOne: '1',
      twoTwo: '2',
      threeThree: true,
      fourFour: '4'
    },
    _none: {
      fourFour: '4'
    },
    _unknown: [ '--five' ]
  })
})
