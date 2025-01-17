'use strict'
const arrayify = require('array-back')
const t = require('typical')

/**
 * @module test-value
 * @example
 * const testValue = require('test-value')
 */
module.exports = testValue

/**
 * @alias module:test-value
 * @param {any} - a value to test
 * @param {any} - the test query
 * @param [options] {object}
 * @param [options.strict] {boolean} - Treat an object like a value not a query.
 * @returns {boolean}
 */
function testValue (value, test, options) {
  options = options || {}
  if (test !== Object.prototype && t.isPlainObject(test) && t.isObject(value) && !options.strict) {
    return Object.keys(test).every(function (prop) {
      let queryValue = test[prop]

      /* get flags */
      let isNegated = false
      let isContains = false

      if (prop.charAt(0) === '!') {
        isNegated = true
      } else if (prop.charAt(0) === '+') {
        isContains = true
      }

      /* strip flag char */
      prop = (isNegated || isContains) ? prop.slice(1) : prop
      let objectValue = value[prop]

      if (isContains) {
        queryValue = arrayify(queryValue)
        objectValue = arrayify(objectValue)
      }

      const result = testValue(objectValue, queryValue, options)
      return isNegated ? !result : result
    })
  } else if (test !== Array.prototype && Array.isArray(test)) {
    const tests = test
    if (value === Array.prototype || !Array.isArray(value)) value = [ value ]
    return value.some(function (val) {
      return tests.some(function (test) {
        return testValue(val, test, options)
      })
    })

  /*
  regexes queries will always return `false` for `null`, `undefined`, `NaN`.
  This is to prevent a query like `/.+/` matching the string `undefined`.
  */
  } else if (test instanceof RegExp) {
    if ([ 'boolean', 'string', 'number' ].indexOf(typeof value) === -1) {
      return false
    } else {
      return test.test(value)
    }
  } else if (test !== Function.prototype && typeof test === 'function') {
    return test(value)
  } else {
    return test === value
  }
}

/**
 * Returns a callback suitable for use by `Array` methods like `some`, `filter`, `find` etc.
 * @param {any} - the test query
 * @returns {function}
 */
testValue.where = function (test) {
  return function (value) {
    return testValue(value, test)
  }
}
