<a name="module_command-line-args"></a>

## command-line-args
<a name="exp_module_command-line-args--commandLineArgs"></a>

### commandLineArgs(optionDefinitions, [options]) ⇒ <code>object</code> ⏏
Returns an object containing all option values set on the command line. By default it parses the global  [`process.argv`](https://nodejs.org/api/process.html#process_process_argv) array.

Parsing is strict by default - an exception is thrown if the user sets a singular option more than once or sets an unknown value or option (one without a valid [definition](https://github.com/75lb/command-line-args/blob/master/doc/option-definition.md)). To be more permissive, enabling [partial](https://github.com/75lb/command-line-args/wiki/Partial-mode-example) or [stopAtFirstUnknown](https://github.com/75lb/command-line-args/wiki/stopAtFirstUnknown) modes will return known options in the usual manner while collecting unknown arguments in a separate `_unknown` property.

**Kind**: Exported function  
**Throws**:

- `UNKNOWN_OPTION` If `options.partial` is false and the user set an undefined option. The `err.optionName` property contains the arg that specified an unknown option, e.g. `--one`.
- `UNKNOWN_VALUE` If `options.partial` is false and the user set a value unaccounted for by an option definition. The `err.value` property contains the unknown value, e.g. `5`.
- `ALREADY_SET` If a user sets a singular, non-multiple option more than once. The `err.optionName` property contains the option name that has already been set, e.g. `one`.
- `INVALID_DEFINITIONS`
  - If an option definition is missing the required `name` property
  - If an option definition has a `type` value that's not a function
  - If an alias is numeric, a hyphen or a length other than 1
  - If an option definition name was used more than once
  - If an option definition alias was used more than once
  - If more than one option definition has `defaultOption: true`
  - If a `Boolean` option is also set as the `defaultOption`.


| Param | Type | Description |
| --- | --- | --- |
| optionDefinitions | <code>Array.&lt;module:definition&gt;</code> | An array of [OptionDefinition](https://github.com/75lb/command-line-args/blob/master/doc/option-definition.md) objects |
| [options] | <code>object</code> | Options. |
| [options.argv] | <code>Array.&lt;string&gt;</code> | An array of strings which, if present will be parsed instead  of `process.argv`. |
| [options.partial] | <code>boolean</code> | If `true`, an array of unknown arguments is returned in the `_unknown` property of the output. |
| [options.stopAtFirstUnknown] | <code>boolean</code> | If `true`, parsing will stop at the first unknown argument and the remaining arguments returned in `_unknown`. When set, `partial: true` is also implied. |
| [options.camelCase] | <code>boolean</code> | If `true`, options with hypenated names (e.g. `move-to`) will be returned in camel-case (e.g. `moveTo`). |

