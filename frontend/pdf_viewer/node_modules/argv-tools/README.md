[![view on npm](https://img.shields.io/npm/v/argv-tools.svg)](https://www.npmjs.org/package/argv-tools)
[![npm module downloads](https://img.shields.io/npm/dt/argv-tools.svg)](https://www.npmjs.org/package/argv-tools)
[![Build Status](https://travis-ci.org/75lb/argv-tools.svg?branch=master)](https://travis-ci.org/75lb/argv-tools)
[![Dependency Status](https://badgen.net/david/dep/75lb/argv-tools)](https://david-dm.org/75lb/argv-tools)
[![js-standard-style](https://img.shields.io/badge/code%20style-standard-brightgreen.svg)](https://github.com/feross/standard)

<a name="module_argv-tools"></a>

## argv-tools
Some useful tools for working with `process.argv`.

**Example**  
```js
const argvTools = require('argv-tools')
```

* [argv-tools](#module_argv-tools)
    * [.ArgvArray](#module_argv-tools.ArgvArray)
        * [.load(argv)](#module_argv-tools.ArgvArray+load)
        * [.clear()](#module_argv-tools.ArgvArray+clear)
        * [.expandOptionEqualsNotation()](#module_argv-tools.ArgvArray+expandOptionEqualsNotation)
        * [.expandGetoptNotation()](#module_argv-tools.ArgvArray+expandGetoptNotation)
        * [.hasCombinedShortOptions()](#module_argv-tools.ArgvArray+hasCombinedShortOptions) ⇒ <code>boolean</code>
    * [.re](#module_argv-tools.re)
    * [.expandCombinedShortArg(arg)](#module_argv-tools.expandCombinedShortArg) ⇒ <code>Array.&lt;string&gt;</code>
    * [.isOptionEqualsNotation(arg)](#module_argv-tools.isOptionEqualsNotation) ⇒ <code>boolean</code>
    * [.isOption(arg)](#module_argv-tools.isOption) ⇒ <code>boolean</code>
    * [.isLongOption(arg)](#module_argv-tools.isLongOption) ⇒ <code>boolean</code>
    * [.getOptionName(arg)](#module_argv-tools.getOptionName) ⇒ <code>string</code>

<a name="module_argv-tools.ArgvArray"></a>

### argvTools.ArgvArray
Array subclass encapsulating common operations on `process.argv`.

**Kind**: static class of [<code>argv-tools</code>](#module_argv-tools)  

* [.ArgvArray](#module_argv-tools.ArgvArray)
    * [.load(argv)](#module_argv-tools.ArgvArray+load)
    * [.clear()](#module_argv-tools.ArgvArray+clear)
    * [.expandOptionEqualsNotation()](#module_argv-tools.ArgvArray+expandOptionEqualsNotation)
    * [.expandGetoptNotation()](#module_argv-tools.ArgvArray+expandGetoptNotation)
    * [.hasCombinedShortOptions()](#module_argv-tools.ArgvArray+hasCombinedShortOptions) ⇒ <code>boolean</code>

<a name="module_argv-tools.ArgvArray+load"></a>

#### argvArray.load(argv)
Clears the array has loads the supplied input.

**Kind**: instance method of [<code>ArgvArray</code>](#module_argv-tools.ArgvArray)  

| Param | Type | Description |
| --- | --- | --- |
| argv | <code>Array.&lt;string&gt;</code> | The argv list to load. Defaults to `process.argv`. |

<a name="module_argv-tools.ArgvArray+clear"></a>

#### argvArray.clear()
Clear the array.

**Kind**: instance method of [<code>ArgvArray</code>](#module_argv-tools.ArgvArray)  
<a name="module_argv-tools.ArgvArray+expandOptionEqualsNotation"></a>

#### argvArray.expandOptionEqualsNotation()
expand ``--option=value` style args.

**Kind**: instance method of [<code>ArgvArray</code>](#module_argv-tools.ArgvArray)  
<a name="module_argv-tools.ArgvArray+expandGetoptNotation"></a>

#### argvArray.expandGetoptNotation()
expand getopt-style combinedShort options.

**Kind**: instance method of [<code>ArgvArray</code>](#module_argv-tools.ArgvArray)  
<a name="module_argv-tools.ArgvArray+hasCombinedShortOptions"></a>

#### argvArray.hasCombinedShortOptions() ⇒ <code>boolean</code>
Returns true if the array contains combined short options (e.g. `-ab`).

**Kind**: instance method of [<code>ArgvArray</code>](#module_argv-tools.ArgvArray)  
<a name="module_argv-tools.re"></a>

### argvTools.re
Regular expressions for matching option formats.

**Kind**: static constant of [<code>argv-tools</code>](#module_argv-tools)  
<a name="module_argv-tools.expandCombinedShortArg"></a>

### argvTools.expandCombinedShortArg(arg) ⇒ <code>Array.&lt;string&gt;</code>
Expand a combined short option.

**Kind**: static method of [<code>argv-tools</code>](#module_argv-tools)  

| Param | Type | Description |
| --- | --- | --- |
| arg | <code>string</code> | the string to expand, e.g. `-ab` |

<a name="module_argv-tools.isOptionEqualsNotation"></a>

### argvTools.isOptionEqualsNotation(arg) ⇒ <code>boolean</code>
Returns true if the supplied arg matches `--option=value` notation.

**Kind**: static method of [<code>argv-tools</code>](#module_argv-tools)  

| Param | Type | Description |
| --- | --- | --- |
| arg | <code>string</code> | the arg to test, e.g. `--one=something` |

<a name="module_argv-tools.isOption"></a>

### argvTools.isOption(arg) ⇒ <code>boolean</code>
Returns true if the supplied arg is in either long (`--one`) or short (`-o`) format.

**Kind**: static method of [<code>argv-tools</code>](#module_argv-tools)  

| Param | Type | Description |
| --- | --- | --- |
| arg | <code>string</code> | the arg to test, e.g. `--one` |

<a name="module_argv-tools.isLongOption"></a>

### argvTools.isLongOption(arg) ⇒ <code>boolean</code>
Returns true if the supplied arg is in long (`--one`) format.

**Kind**: static method of [<code>argv-tools</code>](#module_argv-tools)  

| Param | Type | Description |
| --- | --- | --- |
| arg | <code>string</code> | the arg to test, e.g. `--one` |

<a name="module_argv-tools.getOptionName"></a>

### argvTools.getOptionName(arg) ⇒ <code>string</code>
Returns the name from a long, short or `--options=value` arg.

**Kind**: static method of [<code>argv-tools</code>](#module_argv-tools)  

| Param | Type | Description |
| --- | --- | --- |
| arg | <code>string</code> | the arg to inspect, e.g. `--one` |


* * *

&copy; 2018-19 Lloyd Brookes \<75pound@gmail.com\>. Documented by [jsdoc-to-markdown](https://github.com/75lb/jsdoc-to-markdown).
