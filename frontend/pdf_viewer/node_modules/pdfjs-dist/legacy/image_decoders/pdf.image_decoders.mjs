/**
 * @licstart The following is the entire license notice for the
 * JavaScript code in this page
 *
 * Copyright 2024 Mozilla Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @licend The above is the entire license notice for the
 * JavaScript code in this page
 */

/******/ var __webpack_modules__ = ({

/***/ 9306:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var isCallable = __webpack_require__(4901);
var tryToString = __webpack_require__(6823);

var $TypeError = TypeError;

// `Assert: IsCallable(argument) is true`
module.exports = function (argument) {
  if (isCallable(argument)) return argument;
  throw new $TypeError(tryToString(argument) + ' is not a function');
};


/***/ }),

/***/ 3506:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var isPossiblePrototype = __webpack_require__(3925);

var $String = String;
var $TypeError = TypeError;

module.exports = function (argument) {
  if (isPossiblePrototype(argument)) return argument;
  throw new $TypeError("Can't set " + $String(argument) + ' as a prototype');
};


/***/ }),

/***/ 7080:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var has = (__webpack_require__(4402).has);

// Perform ? RequireInternalSlot(M, [[SetData]])
module.exports = function (it) {
  has(it);
  return it;
};


/***/ }),

/***/ 679:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var isPrototypeOf = __webpack_require__(1625);

var $TypeError = TypeError;

module.exports = function (it, Prototype) {
  if (isPrototypeOf(Prototype, it)) return it;
  throw new $TypeError('Incorrect invocation');
};


/***/ }),

/***/ 8551:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var isObject = __webpack_require__(34);

var $String = String;
var $TypeError = TypeError;

// `Assert: Type(argument) is Object`
module.exports = function (argument) {
  if (isObject(argument)) return argument;
  throw new $TypeError($String(argument) + ' is not an object');
};


/***/ }),

/***/ 7811:
/***/ ((module) => {


// eslint-disable-next-line es/no-typed-arrays -- safe
module.exports = typeof ArrayBuffer != 'undefined' && typeof DataView != 'undefined';


/***/ }),

/***/ 7394:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var globalThis = __webpack_require__(4576);
var uncurryThisAccessor = __webpack_require__(6706);
var classof = __webpack_require__(2195);

var ArrayBuffer = globalThis.ArrayBuffer;
var TypeError = globalThis.TypeError;

// Includes
// - Perform ? RequireInternalSlot(O, [[ArrayBufferData]]).
// - If IsSharedArrayBuffer(O) is true, throw a TypeError exception.
module.exports = ArrayBuffer && uncurryThisAccessor(ArrayBuffer.prototype, 'byteLength', 'get') || function (O) {
  if (classof(O) !== 'ArrayBuffer') throw new TypeError('ArrayBuffer expected');
  return O.byteLength;
};


/***/ }),

/***/ 3238:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var globalThis = __webpack_require__(4576);
var uncurryThis = __webpack_require__(7476);
var arrayBufferByteLength = __webpack_require__(7394);

var ArrayBuffer = globalThis.ArrayBuffer;
var ArrayBufferPrototype = ArrayBuffer && ArrayBuffer.prototype;
var slice = ArrayBufferPrototype && uncurryThis(ArrayBufferPrototype.slice);

module.exports = function (O) {
  if (arrayBufferByteLength(O) !== 0) return false;
  if (!slice) return false;
  try {
    slice(O, 0, 0);
    return false;
  } catch (error) {
    return true;
  }
};


/***/ }),

/***/ 5169:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var isDetached = __webpack_require__(3238);

var $TypeError = TypeError;

module.exports = function (it) {
  if (isDetached(it)) throw new $TypeError('ArrayBuffer is detached');
  return it;
};


/***/ }),

/***/ 5636:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var globalThis = __webpack_require__(4576);
var uncurryThis = __webpack_require__(9504);
var uncurryThisAccessor = __webpack_require__(6706);
var toIndex = __webpack_require__(7696);
var notDetached = __webpack_require__(5169);
var arrayBufferByteLength = __webpack_require__(7394);
var detachTransferable = __webpack_require__(4483);
var PROPER_STRUCTURED_CLONE_TRANSFER = __webpack_require__(1548);

var structuredClone = globalThis.structuredClone;
var ArrayBuffer = globalThis.ArrayBuffer;
var DataView = globalThis.DataView;
var min = Math.min;
var ArrayBufferPrototype = ArrayBuffer.prototype;
var DataViewPrototype = DataView.prototype;
var slice = uncurryThis(ArrayBufferPrototype.slice);
var isResizable = uncurryThisAccessor(ArrayBufferPrototype, 'resizable', 'get');
var maxByteLength = uncurryThisAccessor(ArrayBufferPrototype, 'maxByteLength', 'get');
var getInt8 = uncurryThis(DataViewPrototype.getInt8);
var setInt8 = uncurryThis(DataViewPrototype.setInt8);

module.exports = (PROPER_STRUCTURED_CLONE_TRANSFER || detachTransferable) && function (arrayBuffer, newLength, preserveResizability) {
  var byteLength = arrayBufferByteLength(arrayBuffer);
  var newByteLength = newLength === undefined ? byteLength : toIndex(newLength);
  var fixedLength = !isResizable || !isResizable(arrayBuffer);
  var newBuffer;
  notDetached(arrayBuffer);
  if (PROPER_STRUCTURED_CLONE_TRANSFER) {
    arrayBuffer = structuredClone(arrayBuffer, { transfer: [arrayBuffer] });
    if (byteLength === newByteLength && (preserveResizability || fixedLength)) return arrayBuffer;
  }
  if (byteLength >= newByteLength && (!preserveResizability || fixedLength)) {
    newBuffer = slice(arrayBuffer, 0, newByteLength);
  } else {
    var options = preserveResizability && !fixedLength && maxByteLength ? { maxByteLength: maxByteLength(arrayBuffer) } : undefined;
    newBuffer = new ArrayBuffer(newByteLength, options);
    var a = new DataView(arrayBuffer);
    var b = new DataView(newBuffer);
    var copyLength = min(newByteLength, byteLength);
    for (var i = 0; i < copyLength; i++) setInt8(b, i, getInt8(a, i));
  }
  if (!PROPER_STRUCTURED_CLONE_TRANSFER) detachTransferable(arrayBuffer);
  return newBuffer;
};


/***/ }),

/***/ 4644:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var NATIVE_ARRAY_BUFFER = __webpack_require__(7811);
var DESCRIPTORS = __webpack_require__(3724);
var globalThis = __webpack_require__(4576);
var isCallable = __webpack_require__(4901);
var isObject = __webpack_require__(34);
var hasOwn = __webpack_require__(9297);
var classof = __webpack_require__(6955);
var tryToString = __webpack_require__(6823);
var createNonEnumerableProperty = __webpack_require__(6699);
var defineBuiltIn = __webpack_require__(6840);
var defineBuiltInAccessor = __webpack_require__(2106);
var isPrototypeOf = __webpack_require__(1625);
var getPrototypeOf = __webpack_require__(2787);
var setPrototypeOf = __webpack_require__(2967);
var wellKnownSymbol = __webpack_require__(8227);
var uid = __webpack_require__(3392);
var InternalStateModule = __webpack_require__(1181);

var enforceInternalState = InternalStateModule.enforce;
var getInternalState = InternalStateModule.get;
var Int8Array = globalThis.Int8Array;
var Int8ArrayPrototype = Int8Array && Int8Array.prototype;
var Uint8ClampedArray = globalThis.Uint8ClampedArray;
var Uint8ClampedArrayPrototype = Uint8ClampedArray && Uint8ClampedArray.prototype;
var TypedArray = Int8Array && getPrototypeOf(Int8Array);
var TypedArrayPrototype = Int8ArrayPrototype && getPrototypeOf(Int8ArrayPrototype);
var ObjectPrototype = Object.prototype;
var TypeError = globalThis.TypeError;

var TO_STRING_TAG = wellKnownSymbol('toStringTag');
var TYPED_ARRAY_TAG = uid('TYPED_ARRAY_TAG');
var TYPED_ARRAY_CONSTRUCTOR = 'TypedArrayConstructor';
// Fixing native typed arrays in Opera Presto crashes the browser, see #595
var NATIVE_ARRAY_BUFFER_VIEWS = NATIVE_ARRAY_BUFFER && !!setPrototypeOf && classof(globalThis.opera) !== 'Opera';
var TYPED_ARRAY_TAG_REQUIRED = false;
var NAME, Constructor, Prototype;

var TypedArrayConstructorsList = {
  Int8Array: 1,
  Uint8Array: 1,
  Uint8ClampedArray: 1,
  Int16Array: 2,
  Uint16Array: 2,
  Int32Array: 4,
  Uint32Array: 4,
  Float32Array: 4,
  Float64Array: 8
};

var BigIntArrayConstructorsList = {
  BigInt64Array: 8,
  BigUint64Array: 8
};

var isView = function isView(it) {
  if (!isObject(it)) return false;
  var klass = classof(it);
  return klass === 'DataView'
    || hasOwn(TypedArrayConstructorsList, klass)
    || hasOwn(BigIntArrayConstructorsList, klass);
};

var getTypedArrayConstructor = function (it) {
  var proto = getPrototypeOf(it);
  if (!isObject(proto)) return;
  var state = getInternalState(proto);
  return (state && hasOwn(state, TYPED_ARRAY_CONSTRUCTOR)) ? state[TYPED_ARRAY_CONSTRUCTOR] : getTypedArrayConstructor(proto);
};

var isTypedArray = function (it) {
  if (!isObject(it)) return false;
  var klass = classof(it);
  return hasOwn(TypedArrayConstructorsList, klass)
    || hasOwn(BigIntArrayConstructorsList, klass);
};

var aTypedArray = function (it) {
  if (isTypedArray(it)) return it;
  throw new TypeError('Target is not a typed array');
};

var aTypedArrayConstructor = function (C) {
  if (isCallable(C) && (!setPrototypeOf || isPrototypeOf(TypedArray, C))) return C;
  throw new TypeError(tryToString(C) + ' is not a typed array constructor');
};

var exportTypedArrayMethod = function (KEY, property, forced, options) {
  if (!DESCRIPTORS) return;
  if (forced) for (var ARRAY in TypedArrayConstructorsList) {
    var TypedArrayConstructor = globalThis[ARRAY];
    if (TypedArrayConstructor && hasOwn(TypedArrayConstructor.prototype, KEY)) try {
      delete TypedArrayConstructor.prototype[KEY];
    } catch (error) {
      // old WebKit bug - some methods are non-configurable
      try {
        TypedArrayConstructor.prototype[KEY] = property;
      } catch (error2) { /* empty */ }
    }
  }
  if (!TypedArrayPrototype[KEY] || forced) {
    defineBuiltIn(TypedArrayPrototype, KEY, forced ? property
      : NATIVE_ARRAY_BUFFER_VIEWS && Int8ArrayPrototype[KEY] || property, options);
  }
};

var exportTypedArrayStaticMethod = function (KEY, property, forced) {
  var ARRAY, TypedArrayConstructor;
  if (!DESCRIPTORS) return;
  if (setPrototypeOf) {
    if (forced) for (ARRAY in TypedArrayConstructorsList) {
      TypedArrayConstructor = globalThis[ARRAY];
      if (TypedArrayConstructor && hasOwn(TypedArrayConstructor, KEY)) try {
        delete TypedArrayConstructor[KEY];
      } catch (error) { /* empty */ }
    }
    if (!TypedArray[KEY] || forced) {
      // V8 ~ Chrome 49-50 `%TypedArray%` methods are non-writable non-configurable
      try {
        return defineBuiltIn(TypedArray, KEY, forced ? property : NATIVE_ARRAY_BUFFER_VIEWS && TypedArray[KEY] || property);
      } catch (error) { /* empty */ }
    } else return;
  }
  for (ARRAY in TypedArrayConstructorsList) {
    TypedArrayConstructor = globalThis[ARRAY];
    if (TypedArrayConstructor && (!TypedArrayConstructor[KEY] || forced)) {
      defineBuiltIn(TypedArrayConstructor, KEY, property);
    }
  }
};

for (NAME in TypedArrayConstructorsList) {
  Constructor = globalThis[NAME];
  Prototype = Constructor && Constructor.prototype;
  if (Prototype) enforceInternalState(Prototype)[TYPED_ARRAY_CONSTRUCTOR] = Constructor;
  else NATIVE_ARRAY_BUFFER_VIEWS = false;
}

for (NAME in BigIntArrayConstructorsList) {
  Constructor = globalThis[NAME];
  Prototype = Constructor && Constructor.prototype;
  if (Prototype) enforceInternalState(Prototype)[TYPED_ARRAY_CONSTRUCTOR] = Constructor;
}

// WebKit bug - typed arrays constructors prototype is Object.prototype
if (!NATIVE_ARRAY_BUFFER_VIEWS || !isCallable(TypedArray) || TypedArray === Function.prototype) {
  // eslint-disable-next-line no-shadow -- safe
  TypedArray = function TypedArray() {
    throw new TypeError('Incorrect invocation');
  };
  if (NATIVE_ARRAY_BUFFER_VIEWS) for (NAME in TypedArrayConstructorsList) {
    if (globalThis[NAME]) setPrototypeOf(globalThis[NAME], TypedArray);
  }
}

if (!NATIVE_ARRAY_BUFFER_VIEWS || !TypedArrayPrototype || TypedArrayPrototype === ObjectPrototype) {
  TypedArrayPrototype = TypedArray.prototype;
  if (NATIVE_ARRAY_BUFFER_VIEWS) for (NAME in TypedArrayConstructorsList) {
    if (globalThis[NAME]) setPrototypeOf(globalThis[NAME].prototype, TypedArrayPrototype);
  }
}

// WebKit bug - one more object in Uint8ClampedArray prototype chain
if (NATIVE_ARRAY_BUFFER_VIEWS && getPrototypeOf(Uint8ClampedArrayPrototype) !== TypedArrayPrototype) {
  setPrototypeOf(Uint8ClampedArrayPrototype, TypedArrayPrototype);
}

if (DESCRIPTORS && !hasOwn(TypedArrayPrototype, TO_STRING_TAG)) {
  TYPED_ARRAY_TAG_REQUIRED = true;
  defineBuiltInAccessor(TypedArrayPrototype, TO_STRING_TAG, {
    configurable: true,
    get: function () {
      return isObject(this) ? this[TYPED_ARRAY_TAG] : undefined;
    }
  });
  for (NAME in TypedArrayConstructorsList) if (globalThis[NAME]) {
    createNonEnumerableProperty(globalThis[NAME], TYPED_ARRAY_TAG, NAME);
  }
}

module.exports = {
  NATIVE_ARRAY_BUFFER_VIEWS: NATIVE_ARRAY_BUFFER_VIEWS,
  TYPED_ARRAY_TAG: TYPED_ARRAY_TAG_REQUIRED && TYPED_ARRAY_TAG,
  aTypedArray: aTypedArray,
  aTypedArrayConstructor: aTypedArrayConstructor,
  exportTypedArrayMethod: exportTypedArrayMethod,
  exportTypedArrayStaticMethod: exportTypedArrayStaticMethod,
  getTypedArrayConstructor: getTypedArrayConstructor,
  isView: isView,
  isTypedArray: isTypedArray,
  TypedArray: TypedArray,
  TypedArrayPrototype: TypedArrayPrototype
};


/***/ }),

/***/ 5370:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var lengthOfArrayLike = __webpack_require__(6198);

module.exports = function (Constructor, list, $length) {
  var index = 0;
  var length = arguments.length > 2 ? $length : lengthOfArrayLike(list);
  var result = new Constructor(length);
  while (length > index) result[index] = list[index++];
  return result;
};


/***/ }),

/***/ 9617:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var toIndexedObject = __webpack_require__(5397);
var toAbsoluteIndex = __webpack_require__(5610);
var lengthOfArrayLike = __webpack_require__(6198);

// `Array.prototype.{ indexOf, includes }` methods implementation
var createMethod = function (IS_INCLUDES) {
  return function ($this, el, fromIndex) {
    var O = toIndexedObject($this);
    var length = lengthOfArrayLike(O);
    if (length === 0) return !IS_INCLUDES && -1;
    var index = toAbsoluteIndex(fromIndex, length);
    var value;
    // Array#includes uses SameValueZero equality algorithm
    // eslint-disable-next-line no-self-compare -- NaN check
    if (IS_INCLUDES && el !== el) while (length > index) {
      value = O[index++];
      // eslint-disable-next-line no-self-compare -- NaN check
      if (value !== value) return true;
    // Array#indexOf ignores holes, Array#includes - not
    } else for (;length > index; index++) {
      if ((IS_INCLUDES || index in O) && O[index] === el) return IS_INCLUDES || index || 0;
    } return !IS_INCLUDES && -1;
  };
};

module.exports = {
  // `Array.prototype.includes` method
  // https://tc39.es/ecma262/#sec-array.prototype.includes
  includes: createMethod(true),
  // `Array.prototype.indexOf` method
  // https://tc39.es/ecma262/#sec-array.prototype.indexof
  indexOf: createMethod(false)
};


/***/ }),

/***/ 4527:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var DESCRIPTORS = __webpack_require__(3724);
var isArray = __webpack_require__(4376);

var $TypeError = TypeError;
// eslint-disable-next-line es/no-object-getownpropertydescriptor -- safe
var getOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;

// Safari < 13 does not throw an error in this case
var SILENT_ON_NON_WRITABLE_LENGTH_SET = DESCRIPTORS && !function () {
  // makes no sense without proper strict mode support
  if (this !== undefined) return true;
  try {
    // eslint-disable-next-line es/no-object-defineproperty -- safe
    Object.defineProperty([], 'length', { writable: false }).length = 1;
  } catch (error) {
    return error instanceof TypeError;
  }
}();

module.exports = SILENT_ON_NON_WRITABLE_LENGTH_SET ? function (O, length) {
  if (isArray(O) && !getOwnPropertyDescriptor(O, 'length').writable) {
    throw new $TypeError('Cannot set read only .length');
  } return O.length = length;
} : function (O, length) {
  return O.length = length;
};


/***/ }),

/***/ 7680:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var uncurryThis = __webpack_require__(9504);

module.exports = uncurryThis([].slice);


/***/ }),

/***/ 7628:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var lengthOfArrayLike = __webpack_require__(6198);

// https://tc39.es/proposal-change-array-by-copy/#sec-array.prototype.toReversed
// https://tc39.es/proposal-change-array-by-copy/#sec-%typedarray%.prototype.toReversed
module.exports = function (O, C) {
  var len = lengthOfArrayLike(O);
  var A = new C(len);
  var k = 0;
  for (; k < len; k++) A[k] = O[len - k - 1];
  return A;
};


/***/ }),

/***/ 9928:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var lengthOfArrayLike = __webpack_require__(6198);
var toIntegerOrInfinity = __webpack_require__(1291);

var $RangeError = RangeError;

// https://tc39.es/proposal-change-array-by-copy/#sec-array.prototype.with
// https://tc39.es/proposal-change-array-by-copy/#sec-%typedarray%.prototype.with
module.exports = function (O, C, index, value) {
  var len = lengthOfArrayLike(O);
  var relativeIndex = toIntegerOrInfinity(index);
  var actualIndex = relativeIndex < 0 ? len + relativeIndex : relativeIndex;
  if (actualIndex >= len || actualIndex < 0) throw new $RangeError('Incorrect index');
  var A = new C(len);
  var k = 0;
  for (; k < len; k++) A[k] = k === actualIndex ? value : O[k];
  return A;
};


/***/ }),

/***/ 6319:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var anObject = __webpack_require__(8551);
var iteratorClose = __webpack_require__(9539);

// call something on iterator step with safe closing on error
module.exports = function (iterator, fn, value, ENTRIES) {
  try {
    return ENTRIES ? fn(anObject(value)[0], value[1]) : fn(value);
  } catch (error) {
    iteratorClose(iterator, 'throw', error);
  }
};


/***/ }),

/***/ 2195:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var uncurryThis = __webpack_require__(9504);

var toString = uncurryThis({}.toString);
var stringSlice = uncurryThis(''.slice);

module.exports = function (it) {
  return stringSlice(toString(it), 8, -1);
};


/***/ }),

/***/ 6955:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var TO_STRING_TAG_SUPPORT = __webpack_require__(2140);
var isCallable = __webpack_require__(4901);
var classofRaw = __webpack_require__(2195);
var wellKnownSymbol = __webpack_require__(8227);

var TO_STRING_TAG = wellKnownSymbol('toStringTag');
var $Object = Object;

// ES3 wrong here
var CORRECT_ARGUMENTS = classofRaw(function () { return arguments; }()) === 'Arguments';

// fallback for IE11 Script Access Denied error
var tryGet = function (it, key) {
  try {
    return it[key];
  } catch (error) { /* empty */ }
};

// getting tag from ES6+ `Object.prototype.toString`
module.exports = TO_STRING_TAG_SUPPORT ? classofRaw : function (it) {
  var O, tag, result;
  return it === undefined ? 'Undefined' : it === null ? 'Null'
    // @@toStringTag case
    : typeof (tag = tryGet(O = $Object(it), TO_STRING_TAG)) == 'string' ? tag
    // builtinTag case
    : CORRECT_ARGUMENTS ? classofRaw(O)
    // ES3 arguments fallback
    : (result = classofRaw(O)) === 'Object' && isCallable(O.callee) ? 'Arguments' : result;
};


/***/ }),

/***/ 7740:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var hasOwn = __webpack_require__(9297);
var ownKeys = __webpack_require__(5031);
var getOwnPropertyDescriptorModule = __webpack_require__(7347);
var definePropertyModule = __webpack_require__(4913);

module.exports = function (target, source, exceptions) {
  var keys = ownKeys(source);
  var defineProperty = definePropertyModule.f;
  var getOwnPropertyDescriptor = getOwnPropertyDescriptorModule.f;
  for (var i = 0; i < keys.length; i++) {
    var key = keys[i];
    if (!hasOwn(target, key) && !(exceptions && hasOwn(exceptions, key))) {
      defineProperty(target, key, getOwnPropertyDescriptor(source, key));
    }
  }
};


/***/ }),

/***/ 2211:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var fails = __webpack_require__(9039);

module.exports = !fails(function () {
  function F() { /* empty */ }
  F.prototype.constructor = null;
  // eslint-disable-next-line es/no-object-getprototypeof -- required for testing
  return Object.getPrototypeOf(new F()) !== F.prototype;
});


/***/ }),

/***/ 2529:
/***/ ((module) => {


// `CreateIterResultObject` abstract operation
// https://tc39.es/ecma262/#sec-createiterresultobject
module.exports = function (value, done) {
  return { value: value, done: done };
};


/***/ }),

/***/ 6699:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var DESCRIPTORS = __webpack_require__(3724);
var definePropertyModule = __webpack_require__(4913);
var createPropertyDescriptor = __webpack_require__(6980);

module.exports = DESCRIPTORS ? function (object, key, value) {
  return definePropertyModule.f(object, key, createPropertyDescriptor(1, value));
} : function (object, key, value) {
  object[key] = value;
  return object;
};


/***/ }),

/***/ 6980:
/***/ ((module) => {


module.exports = function (bitmap, value) {
  return {
    enumerable: !(bitmap & 1),
    configurable: !(bitmap & 2),
    writable: !(bitmap & 4),
    value: value
  };
};


/***/ }),

/***/ 4659:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var DESCRIPTORS = __webpack_require__(3724);
var definePropertyModule = __webpack_require__(4913);
var createPropertyDescriptor = __webpack_require__(6980);

module.exports = function (object, key, value) {
  if (DESCRIPTORS) definePropertyModule.f(object, key, createPropertyDescriptor(0, value));
  else object[key] = value;
};


/***/ }),

/***/ 2106:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var makeBuiltIn = __webpack_require__(283);
var defineProperty = __webpack_require__(4913);

module.exports = function (target, name, descriptor) {
  if (descriptor.get) makeBuiltIn(descriptor.get, name, { getter: true });
  if (descriptor.set) makeBuiltIn(descriptor.set, name, { setter: true });
  return defineProperty.f(target, name, descriptor);
};


/***/ }),

/***/ 6840:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var isCallable = __webpack_require__(4901);
var definePropertyModule = __webpack_require__(4913);
var makeBuiltIn = __webpack_require__(283);
var defineGlobalProperty = __webpack_require__(9433);

module.exports = function (O, key, value, options) {
  if (!options) options = {};
  var simple = options.enumerable;
  var name = options.name !== undefined ? options.name : key;
  if (isCallable(value)) makeBuiltIn(value, name, options);
  if (options.global) {
    if (simple) O[key] = value;
    else defineGlobalProperty(key, value);
  } else {
    try {
      if (!options.unsafe) delete O[key];
      else if (O[key]) simple = true;
    } catch (error) { /* empty */ }
    if (simple) O[key] = value;
    else definePropertyModule.f(O, key, {
      value: value,
      enumerable: false,
      configurable: !options.nonConfigurable,
      writable: !options.nonWritable
    });
  } return O;
};


/***/ }),

/***/ 6279:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var defineBuiltIn = __webpack_require__(6840);

module.exports = function (target, src, options) {
  for (var key in src) defineBuiltIn(target, key, src[key], options);
  return target;
};


/***/ }),

/***/ 9433:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var globalThis = __webpack_require__(4576);

// eslint-disable-next-line es/no-object-defineproperty -- safe
var defineProperty = Object.defineProperty;

module.exports = function (key, value) {
  try {
    defineProperty(globalThis, key, { value: value, configurable: true, writable: true });
  } catch (error) {
    globalThis[key] = value;
  } return value;
};


/***/ }),

/***/ 3724:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var fails = __webpack_require__(9039);

// Detect IE8's incomplete defineProperty implementation
module.exports = !fails(function () {
  // eslint-disable-next-line es/no-object-defineproperty -- required for testing
  return Object.defineProperty({}, 1, { get: function () { return 7; } })[1] !== 7;
});


/***/ }),

/***/ 4483:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var globalThis = __webpack_require__(4576);
var getBuiltInNodeModule = __webpack_require__(9429);
var PROPER_STRUCTURED_CLONE_TRANSFER = __webpack_require__(1548);

var structuredClone = globalThis.structuredClone;
var $ArrayBuffer = globalThis.ArrayBuffer;
var $MessageChannel = globalThis.MessageChannel;
var detach = false;
var WorkerThreads, channel, buffer, $detach;

if (PROPER_STRUCTURED_CLONE_TRANSFER) {
  detach = function (transferable) {
    structuredClone(transferable, { transfer: [transferable] });
  };
} else if ($ArrayBuffer) try {
  if (!$MessageChannel) {
    WorkerThreads = getBuiltInNodeModule('worker_threads');
    if (WorkerThreads) $MessageChannel = WorkerThreads.MessageChannel;
  }

  if ($MessageChannel) {
    channel = new $MessageChannel();
    buffer = new $ArrayBuffer(2);

    $detach = function (transferable) {
      channel.port1.postMessage(null, [transferable]);
    };

    if (buffer.byteLength === 2) {
      $detach(buffer);
      if (buffer.byteLength === 0) detach = $detach;
    }
  }
} catch (error) { /* empty */ }

module.exports = detach;


/***/ }),

/***/ 4055:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var globalThis = __webpack_require__(4576);
var isObject = __webpack_require__(34);

var document = globalThis.document;
// typeof document.createElement is 'object' in old IE
var EXISTS = isObject(document) && isObject(document.createElement);

module.exports = function (it) {
  return EXISTS ? document.createElement(it) : {};
};


/***/ }),

/***/ 6837:
/***/ ((module) => {


var $TypeError = TypeError;
var MAX_SAFE_INTEGER = 0x1FFFFFFFFFFFFF; // 2 ** 53 - 1 == 9007199254740991

module.exports = function (it) {
  if (it > MAX_SAFE_INTEGER) throw $TypeError('Maximum allowed index exceeded');
  return it;
};


/***/ }),

/***/ 5002:
/***/ ((module) => {


module.exports = {
  IndexSizeError: { s: 'INDEX_SIZE_ERR', c: 1, m: 1 },
  DOMStringSizeError: { s: 'DOMSTRING_SIZE_ERR', c: 2, m: 0 },
  HierarchyRequestError: { s: 'HIERARCHY_REQUEST_ERR', c: 3, m: 1 },
  WrongDocumentError: { s: 'WRONG_DOCUMENT_ERR', c: 4, m: 1 },
  InvalidCharacterError: { s: 'INVALID_CHARACTER_ERR', c: 5, m: 1 },
  NoDataAllowedError: { s: 'NO_DATA_ALLOWED_ERR', c: 6, m: 0 },
  NoModificationAllowedError: { s: 'NO_MODIFICATION_ALLOWED_ERR', c: 7, m: 1 },
  NotFoundError: { s: 'NOT_FOUND_ERR', c: 8, m: 1 },
  NotSupportedError: { s: 'NOT_SUPPORTED_ERR', c: 9, m: 1 },
  InUseAttributeError: { s: 'INUSE_ATTRIBUTE_ERR', c: 10, m: 1 },
  InvalidStateError: { s: 'INVALID_STATE_ERR', c: 11, m: 1 },
  SyntaxError: { s: 'SYNTAX_ERR', c: 12, m: 1 },
  InvalidModificationError: { s: 'INVALID_MODIFICATION_ERR', c: 13, m: 1 },
  NamespaceError: { s: 'NAMESPACE_ERR', c: 14, m: 1 },
  InvalidAccessError: { s: 'INVALID_ACCESS_ERR', c: 15, m: 1 },
  ValidationError: { s: 'VALIDATION_ERR', c: 16, m: 0 },
  TypeMismatchError: { s: 'TYPE_MISMATCH_ERR', c: 17, m: 1 },
  SecurityError: { s: 'SECURITY_ERR', c: 18, m: 1 },
  NetworkError: { s: 'NETWORK_ERR', c: 19, m: 1 },
  AbortError: { s: 'ABORT_ERR', c: 20, m: 1 },
  URLMismatchError: { s: 'URL_MISMATCH_ERR', c: 21, m: 1 },
  QuotaExceededError: { s: 'QUOTA_EXCEEDED_ERR', c: 22, m: 1 },
  TimeoutError: { s: 'TIMEOUT_ERR', c: 23, m: 1 },
  InvalidNodeTypeError: { s: 'INVALID_NODE_TYPE_ERR', c: 24, m: 1 },
  DataCloneError: { s: 'DATA_CLONE_ERR', c: 25, m: 1 }
};


/***/ }),

/***/ 8727:
/***/ ((module) => {


// IE8- don't enum bug keys
module.exports = [
  'constructor',
  'hasOwnProperty',
  'isPrototypeOf',
  'propertyIsEnumerable',
  'toLocaleString',
  'toString',
  'valueOf'
];


/***/ }),

/***/ 6193:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var ENVIRONMENT = __webpack_require__(4215);

module.exports = ENVIRONMENT === 'NODE';


/***/ }),

/***/ 2839:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var globalThis = __webpack_require__(4576);

var navigator = globalThis.navigator;
var userAgent = navigator && navigator.userAgent;

module.exports = userAgent ? String(userAgent) : '';


/***/ }),

/***/ 9519:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var globalThis = __webpack_require__(4576);
var userAgent = __webpack_require__(2839);

var process = globalThis.process;
var Deno = globalThis.Deno;
var versions = process && process.versions || Deno && Deno.version;
var v8 = versions && versions.v8;
var match, version;

if (v8) {
  match = v8.split('.');
  // in old Chrome, versions of V8 isn't V8 = Chrome / 10
  // but their correct versions are not interesting for us
  version = match[0] > 0 && match[0] < 4 ? 1 : +(match[0] + match[1]);
}

// BrowserFS NodeJS `process` polyfill incorrectly set `.v8` to `0.0`
// so check `userAgent` even if `.v8` exists, but 0
if (!version && userAgent) {
  match = userAgent.match(/Edge\/(\d+)/);
  if (!match || match[1] >= 74) {
    match = userAgent.match(/Chrome\/(\d+)/);
    if (match) version = +match[1];
  }
}

module.exports = version;


/***/ }),

/***/ 4215:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


/* global Bun, Deno -- detection */
var globalThis = __webpack_require__(4576);
var userAgent = __webpack_require__(2839);
var classof = __webpack_require__(2195);

var userAgentStartsWith = function (string) {
  return userAgent.slice(0, string.length) === string;
};

module.exports = (function () {
  if (userAgentStartsWith('Bun/')) return 'BUN';
  if (userAgentStartsWith('Cloudflare-Workers')) return 'CLOUDFLARE';
  if (userAgentStartsWith('Deno/')) return 'DENO';
  if (userAgentStartsWith('Node.js/')) return 'NODE';
  if (globalThis.Bun && typeof Bun.version == 'string') return 'BUN';
  if (globalThis.Deno && typeof Deno.version == 'object') return 'DENO';
  if (classof(globalThis.process) === 'process') return 'NODE';
  if (globalThis.window && globalThis.document) return 'BROWSER';
  return 'REST';
})();


/***/ }),

/***/ 8574:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var uncurryThis = __webpack_require__(9504);

var $Error = Error;
var replace = uncurryThis(''.replace);

var TEST = (function (arg) { return String(new $Error(arg).stack); })('zxcasd');
// eslint-disable-next-line redos/no-vulnerable, sonarjs/slow-regex -- safe
var V8_OR_CHAKRA_STACK_ENTRY = /\n\s*at [^:]*:[^\n]*/;
var IS_V8_OR_CHAKRA_STACK = V8_OR_CHAKRA_STACK_ENTRY.test(TEST);

module.exports = function (stack, dropEntries) {
  if (IS_V8_OR_CHAKRA_STACK && typeof stack == 'string' && !$Error.prepareStackTrace) {
    while (dropEntries--) stack = replace(stack, V8_OR_CHAKRA_STACK_ENTRY, '');
  } return stack;
};


/***/ }),

/***/ 6518:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var globalThis = __webpack_require__(4576);
var getOwnPropertyDescriptor = (__webpack_require__(7347).f);
var createNonEnumerableProperty = __webpack_require__(6699);
var defineBuiltIn = __webpack_require__(6840);
var defineGlobalProperty = __webpack_require__(9433);
var copyConstructorProperties = __webpack_require__(7740);
var isForced = __webpack_require__(2796);

/*
  options.target         - name of the target object
  options.global         - target is the global object
  options.stat           - export as static methods of target
  options.proto          - export as prototype methods of target
  options.real           - real prototype method for the `pure` version
  options.forced         - export even if the native feature is available
  options.bind           - bind methods to the target, required for the `pure` version
  options.wrap           - wrap constructors to preventing global pollution, required for the `pure` version
  options.unsafe         - use the simple assignment of property instead of delete + defineProperty
  options.sham           - add a flag to not completely full polyfills
  options.enumerable     - export as enumerable property
  options.dontCallGetSet - prevent calling a getter on target
  options.name           - the .name of the function if it does not match the key
*/
module.exports = function (options, source) {
  var TARGET = options.target;
  var GLOBAL = options.global;
  var STATIC = options.stat;
  var FORCED, target, key, targetProperty, sourceProperty, descriptor;
  if (GLOBAL) {
    target = globalThis;
  } else if (STATIC) {
    target = globalThis[TARGET] || defineGlobalProperty(TARGET, {});
  } else {
    target = globalThis[TARGET] && globalThis[TARGET].prototype;
  }
  if (target) for (key in source) {
    sourceProperty = source[key];
    if (options.dontCallGetSet) {
      descriptor = getOwnPropertyDescriptor(target, key);
      targetProperty = descriptor && descriptor.value;
    } else targetProperty = target[key];
    FORCED = isForced(GLOBAL ? key : TARGET + (STATIC ? '.' : '#') + key, options.forced);
    // contained in target
    if (!FORCED && targetProperty !== undefined) {
      if (typeof sourceProperty == typeof targetProperty) continue;
      copyConstructorProperties(sourceProperty, targetProperty);
    }
    // add a flag to not completely full polyfills
    if (options.sham || (targetProperty && targetProperty.sham)) {
      createNonEnumerableProperty(sourceProperty, 'sham', true);
    }
    defineBuiltIn(target, key, sourceProperty, options);
  }
};


/***/ }),

/***/ 9039:
/***/ ((module) => {


module.exports = function (exec) {
  try {
    return !!exec();
  } catch (error) {
    return true;
  }
};


/***/ }),

/***/ 8745:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var NATIVE_BIND = __webpack_require__(616);

var FunctionPrototype = Function.prototype;
var apply = FunctionPrototype.apply;
var call = FunctionPrototype.call;

// eslint-disable-next-line es/no-reflect -- safe
module.exports = typeof Reflect == 'object' && Reflect.apply || (NATIVE_BIND ? call.bind(apply) : function () {
  return call.apply(apply, arguments);
});


/***/ }),

/***/ 6080:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var uncurryThis = __webpack_require__(7476);
var aCallable = __webpack_require__(9306);
var NATIVE_BIND = __webpack_require__(616);

var bind = uncurryThis(uncurryThis.bind);

// optional / simple context binding
module.exports = function (fn, that) {
  aCallable(fn);
  return that === undefined ? fn : NATIVE_BIND ? bind(fn, that) : function (/* ...args */) {
    return fn.apply(that, arguments);
  };
};


/***/ }),

/***/ 616:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var fails = __webpack_require__(9039);

module.exports = !fails(function () {
  // eslint-disable-next-line es/no-function-prototype-bind -- safe
  var test = (function () { /* empty */ }).bind();
  // eslint-disable-next-line no-prototype-builtins -- safe
  return typeof test != 'function' || test.hasOwnProperty('prototype');
});


/***/ }),

/***/ 9565:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var NATIVE_BIND = __webpack_require__(616);

var call = Function.prototype.call;

module.exports = NATIVE_BIND ? call.bind(call) : function () {
  return call.apply(call, arguments);
};


/***/ }),

/***/ 350:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var DESCRIPTORS = __webpack_require__(3724);
var hasOwn = __webpack_require__(9297);

var FunctionPrototype = Function.prototype;
// eslint-disable-next-line es/no-object-getownpropertydescriptor -- safe
var getDescriptor = DESCRIPTORS && Object.getOwnPropertyDescriptor;

var EXISTS = hasOwn(FunctionPrototype, 'name');
// additional protection from minified / mangled / dropped function names
var PROPER = EXISTS && (function something() { /* empty */ }).name === 'something';
var CONFIGURABLE = EXISTS && (!DESCRIPTORS || (DESCRIPTORS && getDescriptor(FunctionPrototype, 'name').configurable));

module.exports = {
  EXISTS: EXISTS,
  PROPER: PROPER,
  CONFIGURABLE: CONFIGURABLE
};


/***/ }),

/***/ 6706:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var uncurryThis = __webpack_require__(9504);
var aCallable = __webpack_require__(9306);

module.exports = function (object, key, method) {
  try {
    // eslint-disable-next-line es/no-object-getownpropertydescriptor -- safe
    return uncurryThis(aCallable(Object.getOwnPropertyDescriptor(object, key)[method]));
  } catch (error) { /* empty */ }
};


/***/ }),

/***/ 7476:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var classofRaw = __webpack_require__(2195);
var uncurryThis = __webpack_require__(9504);

module.exports = function (fn) {
  // Nashorn bug:
  //   https://github.com/zloirock/core-js/issues/1128
  //   https://github.com/zloirock/core-js/issues/1130
  if (classofRaw(fn) === 'Function') return uncurryThis(fn);
};


/***/ }),

/***/ 9504:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var NATIVE_BIND = __webpack_require__(616);

var FunctionPrototype = Function.prototype;
var call = FunctionPrototype.call;
var uncurryThisWithBind = NATIVE_BIND && FunctionPrototype.bind.bind(call, call);

module.exports = NATIVE_BIND ? uncurryThisWithBind : function (fn) {
  return function () {
    return call.apply(fn, arguments);
  };
};


/***/ }),

/***/ 9429:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var globalThis = __webpack_require__(4576);
var IS_NODE = __webpack_require__(6193);

module.exports = function (name) {
  if (IS_NODE) {
    try {
      return globalThis.process.getBuiltinModule(name);
    } catch (error) { /* empty */ }
    try {
      // eslint-disable-next-line no-new-func -- safe
      return Function('return require("' + name + '")')();
    } catch (error) { /* empty */ }
  }
};


/***/ }),

/***/ 7751:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var globalThis = __webpack_require__(4576);
var isCallable = __webpack_require__(4901);

var aFunction = function (argument) {
  return isCallable(argument) ? argument : undefined;
};

module.exports = function (namespace, method) {
  return arguments.length < 2 ? aFunction(globalThis[namespace]) : globalThis[namespace] && globalThis[namespace][method];
};


/***/ }),

/***/ 1767:
/***/ ((module) => {


// `GetIteratorDirect(obj)` abstract operation
// https://tc39.es/proposal-iterator-helpers/#sec-getiteratordirect
module.exports = function (obj) {
  return {
    iterator: obj,
    next: obj.next,
    done: false
  };
};


/***/ }),

/***/ 851:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var classof = __webpack_require__(6955);
var getMethod = __webpack_require__(5966);
var isNullOrUndefined = __webpack_require__(4117);
var Iterators = __webpack_require__(6269);
var wellKnownSymbol = __webpack_require__(8227);

var ITERATOR = wellKnownSymbol('iterator');

module.exports = function (it) {
  if (!isNullOrUndefined(it)) return getMethod(it, ITERATOR)
    || getMethod(it, '@@iterator')
    || Iterators[classof(it)];
};


/***/ }),

/***/ 81:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var call = __webpack_require__(9565);
var aCallable = __webpack_require__(9306);
var anObject = __webpack_require__(8551);
var tryToString = __webpack_require__(6823);
var getIteratorMethod = __webpack_require__(851);

var $TypeError = TypeError;

module.exports = function (argument, usingIterator) {
  var iteratorMethod = arguments.length < 2 ? getIteratorMethod(argument) : usingIterator;
  if (aCallable(iteratorMethod)) return anObject(call(iteratorMethod, argument));
  throw new $TypeError(tryToString(argument) + ' is not iterable');
};


/***/ }),

/***/ 5966:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var aCallable = __webpack_require__(9306);
var isNullOrUndefined = __webpack_require__(4117);

// `GetMethod` abstract operation
// https://tc39.es/ecma262/#sec-getmethod
module.exports = function (V, P) {
  var func = V[P];
  return isNullOrUndefined(func) ? undefined : aCallable(func);
};


/***/ }),

/***/ 3789:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var aCallable = __webpack_require__(9306);
var anObject = __webpack_require__(8551);
var call = __webpack_require__(9565);
var toIntegerOrInfinity = __webpack_require__(1291);
var getIteratorDirect = __webpack_require__(1767);

var INVALID_SIZE = 'Invalid size';
var $RangeError = RangeError;
var $TypeError = TypeError;
var max = Math.max;

var SetRecord = function (set, intSize) {
  this.set = set;
  this.size = max(intSize, 0);
  this.has = aCallable(set.has);
  this.keys = aCallable(set.keys);
};

SetRecord.prototype = {
  getIterator: function () {
    return getIteratorDirect(anObject(call(this.keys, this.set)));
  },
  includes: function (it) {
    return call(this.has, this.set, it);
  }
};

// `GetSetRecord` abstract operation
// https://tc39.es/proposal-set-methods/#sec-getsetrecord
module.exports = function (obj) {
  anObject(obj);
  var numSize = +obj.size;
  // NOTE: If size is undefined, then numSize will be NaN
  // eslint-disable-next-line no-self-compare -- NaN check
  if (numSize !== numSize) throw new $TypeError(INVALID_SIZE);
  var intSize = toIntegerOrInfinity(numSize);
  if (intSize < 0) throw new $RangeError(INVALID_SIZE);
  return new SetRecord(obj, intSize);
};


/***/ }),

/***/ 4576:
/***/ (function(module) {


var check = function (it) {
  return it && it.Math === Math && it;
};

// https://github.com/zloirock/core-js/issues/86#issuecomment-115759028
module.exports =
  // eslint-disable-next-line es/no-global-this -- safe
  check(typeof globalThis == 'object' && globalThis) ||
  check(typeof window == 'object' && window) ||
  // eslint-disable-next-line no-restricted-globals -- safe
  check(typeof self == 'object' && self) ||
  check(typeof global == 'object' && global) ||
  check(typeof this == 'object' && this) ||
  // eslint-disable-next-line no-new-func -- fallback
  (function () { return this; })() || Function('return this')();


/***/ }),

/***/ 9297:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var uncurryThis = __webpack_require__(9504);
var toObject = __webpack_require__(8981);

var hasOwnProperty = uncurryThis({}.hasOwnProperty);

// `HasOwnProperty` abstract operation
// https://tc39.es/ecma262/#sec-hasownproperty
// eslint-disable-next-line es/no-object-hasown -- safe
module.exports = Object.hasOwn || function hasOwn(it, key) {
  return hasOwnProperty(toObject(it), key);
};


/***/ }),

/***/ 421:
/***/ ((module) => {


module.exports = {};


/***/ }),

/***/ 397:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var getBuiltIn = __webpack_require__(7751);

module.exports = getBuiltIn('document', 'documentElement');


/***/ }),

/***/ 5917:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var DESCRIPTORS = __webpack_require__(3724);
var fails = __webpack_require__(9039);
var createElement = __webpack_require__(4055);

// Thanks to IE8 for its funny defineProperty
module.exports = !DESCRIPTORS && !fails(function () {
  // eslint-disable-next-line es/no-object-defineproperty -- required for testing
  return Object.defineProperty(createElement('div'), 'a', {
    get: function () { return 7; }
  }).a !== 7;
});


/***/ }),

/***/ 7055:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var uncurryThis = __webpack_require__(9504);
var fails = __webpack_require__(9039);
var classof = __webpack_require__(2195);

var $Object = Object;
var split = uncurryThis(''.split);

// fallback for non-array-like ES3 and non-enumerable old V8 strings
module.exports = fails(function () {
  // throws an error in rhino, see https://github.com/mozilla/rhino/issues/346
  // eslint-disable-next-line no-prototype-builtins -- safe
  return !$Object('z').propertyIsEnumerable(0);
}) ? function (it) {
  return classof(it) === 'String' ? split(it, '') : $Object(it);
} : $Object;


/***/ }),

/***/ 3167:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var isCallable = __webpack_require__(4901);
var isObject = __webpack_require__(34);
var setPrototypeOf = __webpack_require__(2967);

// makes subclassing work correct for wrapped built-ins
module.exports = function ($this, dummy, Wrapper) {
  var NewTarget, NewTargetPrototype;
  if (
    // it can work only with native `setPrototypeOf`
    setPrototypeOf &&
    // we haven't completely correct pre-ES6 way for getting `new.target`, so use this
    isCallable(NewTarget = dummy.constructor) &&
    NewTarget !== Wrapper &&
    isObject(NewTargetPrototype = NewTarget.prototype) &&
    NewTargetPrototype !== Wrapper.prototype
  ) setPrototypeOf($this, NewTargetPrototype);
  return $this;
};


/***/ }),

/***/ 3706:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var uncurryThis = __webpack_require__(9504);
var isCallable = __webpack_require__(4901);
var store = __webpack_require__(7629);

var functionToString = uncurryThis(Function.toString);

// this helper broken in `core-js@3.4.1-3.4.4`, so we can't use `shared` helper
if (!isCallable(store.inspectSource)) {
  store.inspectSource = function (it) {
    return functionToString(it);
  };
}

module.exports = store.inspectSource;


/***/ }),

/***/ 1181:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var NATIVE_WEAK_MAP = __webpack_require__(8622);
var globalThis = __webpack_require__(4576);
var isObject = __webpack_require__(34);
var createNonEnumerableProperty = __webpack_require__(6699);
var hasOwn = __webpack_require__(9297);
var shared = __webpack_require__(7629);
var sharedKey = __webpack_require__(6119);
var hiddenKeys = __webpack_require__(421);

var OBJECT_ALREADY_INITIALIZED = 'Object already initialized';
var TypeError = globalThis.TypeError;
var WeakMap = globalThis.WeakMap;
var set, get, has;

var enforce = function (it) {
  return has(it) ? get(it) : set(it, {});
};

var getterFor = function (TYPE) {
  return function (it) {
    var state;
    if (!isObject(it) || (state = get(it)).type !== TYPE) {
      throw new TypeError('Incompatible receiver, ' + TYPE + ' required');
    } return state;
  };
};

if (NATIVE_WEAK_MAP || shared.state) {
  var store = shared.state || (shared.state = new WeakMap());
  /* eslint-disable no-self-assign -- prototype methods protection */
  store.get = store.get;
  store.has = store.has;
  store.set = store.set;
  /* eslint-enable no-self-assign -- prototype methods protection */
  set = function (it, metadata) {
    if (store.has(it)) throw new TypeError(OBJECT_ALREADY_INITIALIZED);
    metadata.facade = it;
    store.set(it, metadata);
    return metadata;
  };
  get = function (it) {
    return store.get(it) || {};
  };
  has = function (it) {
    return store.has(it);
  };
} else {
  var STATE = sharedKey('state');
  hiddenKeys[STATE] = true;
  set = function (it, metadata) {
    if (hasOwn(it, STATE)) throw new TypeError(OBJECT_ALREADY_INITIALIZED);
    metadata.facade = it;
    createNonEnumerableProperty(it, STATE, metadata);
    return metadata;
  };
  get = function (it) {
    return hasOwn(it, STATE) ? it[STATE] : {};
  };
  has = function (it) {
    return hasOwn(it, STATE);
  };
}

module.exports = {
  set: set,
  get: get,
  has: has,
  enforce: enforce,
  getterFor: getterFor
};


/***/ }),

/***/ 4209:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var wellKnownSymbol = __webpack_require__(8227);
var Iterators = __webpack_require__(6269);

var ITERATOR = wellKnownSymbol('iterator');
var ArrayPrototype = Array.prototype;

// check on default Array iterator
module.exports = function (it) {
  return it !== undefined && (Iterators.Array === it || ArrayPrototype[ITERATOR] === it);
};


/***/ }),

/***/ 4376:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var classof = __webpack_require__(2195);

// `IsArray` abstract operation
// https://tc39.es/ecma262/#sec-isarray
// eslint-disable-next-line es/no-array-isarray -- safe
module.exports = Array.isArray || function isArray(argument) {
  return classof(argument) === 'Array';
};


/***/ }),

/***/ 1108:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var classof = __webpack_require__(6955);

module.exports = function (it) {
  var klass = classof(it);
  return klass === 'BigInt64Array' || klass === 'BigUint64Array';
};


/***/ }),

/***/ 4901:
/***/ ((module) => {


// https://tc39.es/ecma262/#sec-IsHTMLDDA-internal-slot
var documentAll = typeof document == 'object' && document.all;

// `IsCallable` abstract operation
// https://tc39.es/ecma262/#sec-iscallable
// eslint-disable-next-line unicorn/no-typeof-undefined -- required for testing
module.exports = typeof documentAll == 'undefined' && documentAll !== undefined ? function (argument) {
  return typeof argument == 'function' || argument === documentAll;
} : function (argument) {
  return typeof argument == 'function';
};


/***/ }),

/***/ 2796:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var fails = __webpack_require__(9039);
var isCallable = __webpack_require__(4901);

var replacement = /#|\.prototype\./;

var isForced = function (feature, detection) {
  var value = data[normalize(feature)];
  return value === POLYFILL ? true
    : value === NATIVE ? false
    : isCallable(detection) ? fails(detection)
    : !!detection;
};

var normalize = isForced.normalize = function (string) {
  return String(string).replace(replacement, '.').toLowerCase();
};

var data = isForced.data = {};
var NATIVE = isForced.NATIVE = 'N';
var POLYFILL = isForced.POLYFILL = 'P';

module.exports = isForced;


/***/ }),

/***/ 4117:
/***/ ((module) => {


// we can't use just `it == null` since of `document.all` special case
// https://tc39.es/ecma262/#sec-IsHTMLDDA-internal-slot-aec
module.exports = function (it) {
  return it === null || it === undefined;
};


/***/ }),

/***/ 34:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var isCallable = __webpack_require__(4901);

module.exports = function (it) {
  return typeof it == 'object' ? it !== null : isCallable(it);
};


/***/ }),

/***/ 3925:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var isObject = __webpack_require__(34);

module.exports = function (argument) {
  return isObject(argument) || argument === null;
};


/***/ }),

/***/ 6395:
/***/ ((module) => {


module.exports = false;


/***/ }),

/***/ 757:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var getBuiltIn = __webpack_require__(7751);
var isCallable = __webpack_require__(4901);
var isPrototypeOf = __webpack_require__(1625);
var USE_SYMBOL_AS_UID = __webpack_require__(7040);

var $Object = Object;

module.exports = USE_SYMBOL_AS_UID ? function (it) {
  return typeof it == 'symbol';
} : function (it) {
  var $Symbol = getBuiltIn('Symbol');
  return isCallable($Symbol) && isPrototypeOf($Symbol.prototype, $Object(it));
};


/***/ }),

/***/ 507:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var call = __webpack_require__(9565);

module.exports = function (record, fn, ITERATOR_INSTEAD_OF_RECORD) {
  var iterator = ITERATOR_INSTEAD_OF_RECORD ? record : record.iterator;
  var next = record.next;
  var step, result;
  while (!(step = call(next, iterator)).done) {
    result = fn(step.value);
    if (result !== undefined) return result;
  }
};


/***/ }),

/***/ 2652:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var bind = __webpack_require__(6080);
var call = __webpack_require__(9565);
var anObject = __webpack_require__(8551);
var tryToString = __webpack_require__(6823);
var isArrayIteratorMethod = __webpack_require__(4209);
var lengthOfArrayLike = __webpack_require__(6198);
var isPrototypeOf = __webpack_require__(1625);
var getIterator = __webpack_require__(81);
var getIteratorMethod = __webpack_require__(851);
var iteratorClose = __webpack_require__(9539);

var $TypeError = TypeError;

var Result = function (stopped, result) {
  this.stopped = stopped;
  this.result = result;
};

var ResultPrototype = Result.prototype;

module.exports = function (iterable, unboundFunction, options) {
  var that = options && options.that;
  var AS_ENTRIES = !!(options && options.AS_ENTRIES);
  var IS_RECORD = !!(options && options.IS_RECORD);
  var IS_ITERATOR = !!(options && options.IS_ITERATOR);
  var INTERRUPTED = !!(options && options.INTERRUPTED);
  var fn = bind(unboundFunction, that);
  var iterator, iterFn, index, length, result, next, step;

  var stop = function (condition) {
    if (iterator) iteratorClose(iterator, 'normal', condition);
    return new Result(true, condition);
  };

  var callFn = function (value) {
    if (AS_ENTRIES) {
      anObject(value);
      return INTERRUPTED ? fn(value[0], value[1], stop) : fn(value[0], value[1]);
    } return INTERRUPTED ? fn(value, stop) : fn(value);
  };

  if (IS_RECORD) {
    iterator = iterable.iterator;
  } else if (IS_ITERATOR) {
    iterator = iterable;
  } else {
    iterFn = getIteratorMethod(iterable);
    if (!iterFn) throw new $TypeError(tryToString(iterable) + ' is not iterable');
    // optimisation for array iterators
    if (isArrayIteratorMethod(iterFn)) {
      for (index = 0, length = lengthOfArrayLike(iterable); length > index; index++) {
        result = callFn(iterable[index]);
        if (result && isPrototypeOf(ResultPrototype, result)) return result;
      } return new Result(false);
    }
    iterator = getIterator(iterable, iterFn);
  }

  next = IS_RECORD ? iterable.next : iterator.next;
  while (!(step = call(next, iterator)).done) {
    try {
      result = callFn(step.value);
    } catch (error) {
      iteratorClose(iterator, 'throw', error);
    }
    if (typeof result == 'object' && result && isPrototypeOf(ResultPrototype, result)) return result;
  } return new Result(false);
};


/***/ }),

/***/ 9539:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var call = __webpack_require__(9565);
var anObject = __webpack_require__(8551);
var getMethod = __webpack_require__(5966);

module.exports = function (iterator, kind, value) {
  var innerResult, innerError;
  anObject(iterator);
  try {
    innerResult = getMethod(iterator, 'return');
    if (!innerResult) {
      if (kind === 'throw') throw value;
      return value;
    }
    innerResult = call(innerResult, iterator);
  } catch (error) {
    innerError = true;
    innerResult = error;
  }
  if (kind === 'throw') throw value;
  if (innerError) throw innerResult;
  anObject(innerResult);
  return value;
};


/***/ }),

/***/ 9462:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var call = __webpack_require__(9565);
var create = __webpack_require__(2360);
var createNonEnumerableProperty = __webpack_require__(6699);
var defineBuiltIns = __webpack_require__(6279);
var wellKnownSymbol = __webpack_require__(8227);
var InternalStateModule = __webpack_require__(1181);
var getMethod = __webpack_require__(5966);
var IteratorPrototype = (__webpack_require__(7657).IteratorPrototype);
var createIterResultObject = __webpack_require__(2529);
var iteratorClose = __webpack_require__(9539);

var TO_STRING_TAG = wellKnownSymbol('toStringTag');
var ITERATOR_HELPER = 'IteratorHelper';
var WRAP_FOR_VALID_ITERATOR = 'WrapForValidIterator';
var setInternalState = InternalStateModule.set;

var createIteratorProxyPrototype = function (IS_ITERATOR) {
  var getInternalState = InternalStateModule.getterFor(IS_ITERATOR ? WRAP_FOR_VALID_ITERATOR : ITERATOR_HELPER);

  return defineBuiltIns(create(IteratorPrototype), {
    next: function next() {
      var state = getInternalState(this);
      // for simplification:
      //   for `%WrapForValidIteratorPrototype%.next` our `nextHandler` returns `IterResultObject`
      //   for `%IteratorHelperPrototype%.next` - just a value
      if (IS_ITERATOR) return state.nextHandler();
      try {
        var result = state.done ? undefined : state.nextHandler();
        return createIterResultObject(result, state.done);
      } catch (error) {
        state.done = true;
        throw error;
      }
    },
    'return': function () {
      var state = getInternalState(this);
      var iterator = state.iterator;
      state.done = true;
      if (IS_ITERATOR) {
        var returnMethod = getMethod(iterator, 'return');
        return returnMethod ? call(returnMethod, iterator) : createIterResultObject(undefined, true);
      }
      if (state.inner) try {
        iteratorClose(state.inner.iterator, 'normal');
      } catch (error) {
        return iteratorClose(iterator, 'throw', error);
      }
      if (iterator) iteratorClose(iterator, 'normal');
      return createIterResultObject(undefined, true);
    }
  });
};

var WrapForValidIteratorPrototype = createIteratorProxyPrototype(true);
var IteratorHelperPrototype = createIteratorProxyPrototype(false);

createNonEnumerableProperty(IteratorHelperPrototype, TO_STRING_TAG, 'Iterator Helper');

module.exports = function (nextHandler, IS_ITERATOR) {
  var IteratorProxy = function Iterator(record, state) {
    if (state) {
      state.iterator = record.iterator;
      state.next = record.next;
    } else state = record;
    state.type = IS_ITERATOR ? WRAP_FOR_VALID_ITERATOR : ITERATOR_HELPER;
    state.nextHandler = nextHandler;
    state.counter = 0;
    state.done = false;
    setInternalState(this, state);
  };

  IteratorProxy.prototype = IS_ITERATOR ? WrapForValidIteratorPrototype : IteratorHelperPrototype;

  return IteratorProxy;
};


/***/ }),

/***/ 713:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var call = __webpack_require__(9565);
var aCallable = __webpack_require__(9306);
var anObject = __webpack_require__(8551);
var getIteratorDirect = __webpack_require__(1767);
var createIteratorProxy = __webpack_require__(9462);
var callWithSafeIterationClosing = __webpack_require__(6319);

var IteratorProxy = createIteratorProxy(function () {
  var iterator = this.iterator;
  var result = anObject(call(this.next, iterator));
  var done = this.done = !!result.done;
  if (!done) return callWithSafeIterationClosing(iterator, this.mapper, [result.value, this.counter++], true);
});

// `Iterator.prototype.map` method
// https://github.com/tc39/proposal-iterator-helpers
module.exports = function map(mapper) {
  anObject(this);
  aCallable(mapper);
  return new IteratorProxy(getIteratorDirect(this), {
    mapper: mapper
  });
};


/***/ }),

/***/ 7657:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var fails = __webpack_require__(9039);
var isCallable = __webpack_require__(4901);
var isObject = __webpack_require__(34);
var create = __webpack_require__(2360);
var getPrototypeOf = __webpack_require__(2787);
var defineBuiltIn = __webpack_require__(6840);
var wellKnownSymbol = __webpack_require__(8227);
var IS_PURE = __webpack_require__(6395);

var ITERATOR = wellKnownSymbol('iterator');
var BUGGY_SAFARI_ITERATORS = false;

// `%IteratorPrototype%` object
// https://tc39.es/ecma262/#sec-%iteratorprototype%-object
var IteratorPrototype, PrototypeOfArrayIteratorPrototype, arrayIterator;

/* eslint-disable es/no-array-prototype-keys -- safe */
if ([].keys) {
  arrayIterator = [].keys();
  // Safari 8 has buggy iterators w/o `next`
  if (!('next' in arrayIterator)) BUGGY_SAFARI_ITERATORS = true;
  else {
    PrototypeOfArrayIteratorPrototype = getPrototypeOf(getPrototypeOf(arrayIterator));
    if (PrototypeOfArrayIteratorPrototype !== Object.prototype) IteratorPrototype = PrototypeOfArrayIteratorPrototype;
  }
}

var NEW_ITERATOR_PROTOTYPE = !isObject(IteratorPrototype) || fails(function () {
  var test = {};
  // FF44- legacy iterators case
  return IteratorPrototype[ITERATOR].call(test) !== test;
});

if (NEW_ITERATOR_PROTOTYPE) IteratorPrototype = {};
else if (IS_PURE) IteratorPrototype = create(IteratorPrototype);

// `%IteratorPrototype%[@@iterator]()` method
// https://tc39.es/ecma262/#sec-%iteratorprototype%-@@iterator
if (!isCallable(IteratorPrototype[ITERATOR])) {
  defineBuiltIn(IteratorPrototype, ITERATOR, function () {
    return this;
  });
}

module.exports = {
  IteratorPrototype: IteratorPrototype,
  BUGGY_SAFARI_ITERATORS: BUGGY_SAFARI_ITERATORS
};


/***/ }),

/***/ 6269:
/***/ ((module) => {


module.exports = {};


/***/ }),

/***/ 6198:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var toLength = __webpack_require__(8014);

// `LengthOfArrayLike` abstract operation
// https://tc39.es/ecma262/#sec-lengthofarraylike
module.exports = function (obj) {
  return toLength(obj.length);
};


/***/ }),

/***/ 283:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var uncurryThis = __webpack_require__(9504);
var fails = __webpack_require__(9039);
var isCallable = __webpack_require__(4901);
var hasOwn = __webpack_require__(9297);
var DESCRIPTORS = __webpack_require__(3724);
var CONFIGURABLE_FUNCTION_NAME = (__webpack_require__(350).CONFIGURABLE);
var inspectSource = __webpack_require__(3706);
var InternalStateModule = __webpack_require__(1181);

var enforceInternalState = InternalStateModule.enforce;
var getInternalState = InternalStateModule.get;
var $String = String;
// eslint-disable-next-line es/no-object-defineproperty -- safe
var defineProperty = Object.defineProperty;
var stringSlice = uncurryThis(''.slice);
var replace = uncurryThis(''.replace);
var join = uncurryThis([].join);

var CONFIGURABLE_LENGTH = DESCRIPTORS && !fails(function () {
  return defineProperty(function () { /* empty */ }, 'length', { value: 8 }).length !== 8;
});

var TEMPLATE = String(String).split('String');

var makeBuiltIn = module.exports = function (value, name, options) {
  if (stringSlice($String(name), 0, 7) === 'Symbol(') {
    name = '[' + replace($String(name), /^Symbol\(([^)]*)\).*$/, '$1') + ']';
  }
  if (options && options.getter) name = 'get ' + name;
  if (options && options.setter) name = 'set ' + name;
  if (!hasOwn(value, 'name') || (CONFIGURABLE_FUNCTION_NAME && value.name !== name)) {
    if (DESCRIPTORS) defineProperty(value, 'name', { value: name, configurable: true });
    else value.name = name;
  }
  if (CONFIGURABLE_LENGTH && options && hasOwn(options, 'arity') && value.length !== options.arity) {
    defineProperty(value, 'length', { value: options.arity });
  }
  try {
    if (options && hasOwn(options, 'constructor') && options.constructor) {
      if (DESCRIPTORS) defineProperty(value, 'prototype', { writable: false });
    // in V8 ~ Chrome 53, prototypes of some methods, like `Array.prototype.values`, are non-writable
    } else if (value.prototype) value.prototype = undefined;
  } catch (error) { /* empty */ }
  var state = enforceInternalState(value);
  if (!hasOwn(state, 'source')) {
    state.source = join(TEMPLATE, typeof name == 'string' ? name : '');
  } return value;
};

// add fake Function#toString for correct work wrapped methods / constructors with methods like LoDash isNative
// eslint-disable-next-line no-extend-native -- required
Function.prototype.toString = makeBuiltIn(function toString() {
  return isCallable(this) && getInternalState(this).source || inspectSource(this);
}, 'toString');


/***/ }),

/***/ 741:
/***/ ((module) => {


var ceil = Math.ceil;
var floor = Math.floor;

// `Math.trunc` method
// https://tc39.es/ecma262/#sec-math.trunc
// eslint-disable-next-line es/no-math-trunc -- safe
module.exports = Math.trunc || function trunc(x) {
  var n = +x;
  return (n > 0 ? floor : ceil)(n);
};


/***/ }),

/***/ 6043:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var aCallable = __webpack_require__(9306);

var $TypeError = TypeError;

var PromiseCapability = function (C) {
  var resolve, reject;
  this.promise = new C(function ($$resolve, $$reject) {
    if (resolve !== undefined || reject !== undefined) throw new $TypeError('Bad Promise constructor');
    resolve = $$resolve;
    reject = $$reject;
  });
  this.resolve = aCallable(resolve);
  this.reject = aCallable(reject);
};

// `NewPromiseCapability` abstract operation
// https://tc39.es/ecma262/#sec-newpromisecapability
module.exports.f = function (C) {
  return new PromiseCapability(C);
};


/***/ }),

/***/ 2603:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var toString = __webpack_require__(655);

module.exports = function (argument, $default) {
  return argument === undefined ? arguments.length < 2 ? '' : $default : toString(argument);
};


/***/ }),

/***/ 2360:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


/* global ActiveXObject -- old IE, WSH */
var anObject = __webpack_require__(8551);
var definePropertiesModule = __webpack_require__(6801);
var enumBugKeys = __webpack_require__(8727);
var hiddenKeys = __webpack_require__(421);
var html = __webpack_require__(397);
var documentCreateElement = __webpack_require__(4055);
var sharedKey = __webpack_require__(6119);

var GT = '>';
var LT = '<';
var PROTOTYPE = 'prototype';
var SCRIPT = 'script';
var IE_PROTO = sharedKey('IE_PROTO');

var EmptyConstructor = function () { /* empty */ };

var scriptTag = function (content) {
  return LT + SCRIPT + GT + content + LT + '/' + SCRIPT + GT;
};

// Create object with fake `null` prototype: use ActiveX Object with cleared prototype
var NullProtoObjectViaActiveX = function (activeXDocument) {
  activeXDocument.write(scriptTag(''));
  activeXDocument.close();
  var temp = activeXDocument.parentWindow.Object;
  // eslint-disable-next-line no-useless-assignment -- avoid memory leak
  activeXDocument = null;
  return temp;
};

// Create object with fake `null` prototype: use iframe Object with cleared prototype
var NullProtoObjectViaIFrame = function () {
  // Thrash, waste and sodomy: IE GC bug
  var iframe = documentCreateElement('iframe');
  var JS = 'java' + SCRIPT + ':';
  var iframeDocument;
  iframe.style.display = 'none';
  html.appendChild(iframe);
  // https://github.com/zloirock/core-js/issues/475
  iframe.src = String(JS);
  iframeDocument = iframe.contentWindow.document;
  iframeDocument.open();
  iframeDocument.write(scriptTag('document.F=Object'));
  iframeDocument.close();
  return iframeDocument.F;
};

// Check for document.domain and active x support
// No need to use active x approach when document.domain is not set
// see https://github.com/es-shims/es5-shim/issues/150
// variation of https://github.com/kitcambridge/es5-shim/commit/4f738ac066346
// avoid IE GC bug
var activeXDocument;
var NullProtoObject = function () {
  try {
    activeXDocument = new ActiveXObject('htmlfile');
  } catch (error) { /* ignore */ }
  NullProtoObject = typeof document != 'undefined'
    ? document.domain && activeXDocument
      ? NullProtoObjectViaActiveX(activeXDocument) // old IE
      : NullProtoObjectViaIFrame()
    : NullProtoObjectViaActiveX(activeXDocument); // WSH
  var length = enumBugKeys.length;
  while (length--) delete NullProtoObject[PROTOTYPE][enumBugKeys[length]];
  return NullProtoObject();
};

hiddenKeys[IE_PROTO] = true;

// `Object.create` method
// https://tc39.es/ecma262/#sec-object.create
// eslint-disable-next-line es/no-object-create -- safe
module.exports = Object.create || function create(O, Properties) {
  var result;
  if (O !== null) {
    EmptyConstructor[PROTOTYPE] = anObject(O);
    result = new EmptyConstructor();
    EmptyConstructor[PROTOTYPE] = null;
    // add "__proto__" for Object.getPrototypeOf polyfill
    result[IE_PROTO] = O;
  } else result = NullProtoObject();
  return Properties === undefined ? result : definePropertiesModule.f(result, Properties);
};


/***/ }),

/***/ 6801:
/***/ ((__unused_webpack_module, exports, __webpack_require__) => {


var DESCRIPTORS = __webpack_require__(3724);
var V8_PROTOTYPE_DEFINE_BUG = __webpack_require__(8686);
var definePropertyModule = __webpack_require__(4913);
var anObject = __webpack_require__(8551);
var toIndexedObject = __webpack_require__(5397);
var objectKeys = __webpack_require__(1072);

// `Object.defineProperties` method
// https://tc39.es/ecma262/#sec-object.defineproperties
// eslint-disable-next-line es/no-object-defineproperties -- safe
exports.f = DESCRIPTORS && !V8_PROTOTYPE_DEFINE_BUG ? Object.defineProperties : function defineProperties(O, Properties) {
  anObject(O);
  var props = toIndexedObject(Properties);
  var keys = objectKeys(Properties);
  var length = keys.length;
  var index = 0;
  var key;
  while (length > index) definePropertyModule.f(O, key = keys[index++], props[key]);
  return O;
};


/***/ }),

/***/ 4913:
/***/ ((__unused_webpack_module, exports, __webpack_require__) => {


var DESCRIPTORS = __webpack_require__(3724);
var IE8_DOM_DEFINE = __webpack_require__(5917);
var V8_PROTOTYPE_DEFINE_BUG = __webpack_require__(8686);
var anObject = __webpack_require__(8551);
var toPropertyKey = __webpack_require__(6969);

var $TypeError = TypeError;
// eslint-disable-next-line es/no-object-defineproperty -- safe
var $defineProperty = Object.defineProperty;
// eslint-disable-next-line es/no-object-getownpropertydescriptor -- safe
var $getOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;
var ENUMERABLE = 'enumerable';
var CONFIGURABLE = 'configurable';
var WRITABLE = 'writable';

// `Object.defineProperty` method
// https://tc39.es/ecma262/#sec-object.defineproperty
exports.f = DESCRIPTORS ? V8_PROTOTYPE_DEFINE_BUG ? function defineProperty(O, P, Attributes) {
  anObject(O);
  P = toPropertyKey(P);
  anObject(Attributes);
  if (typeof O === 'function' && P === 'prototype' && 'value' in Attributes && WRITABLE in Attributes && !Attributes[WRITABLE]) {
    var current = $getOwnPropertyDescriptor(O, P);
    if (current && current[WRITABLE]) {
      O[P] = Attributes.value;
      Attributes = {
        configurable: CONFIGURABLE in Attributes ? Attributes[CONFIGURABLE] : current[CONFIGURABLE],
        enumerable: ENUMERABLE in Attributes ? Attributes[ENUMERABLE] : current[ENUMERABLE],
        writable: false
      };
    }
  } return $defineProperty(O, P, Attributes);
} : $defineProperty : function defineProperty(O, P, Attributes) {
  anObject(O);
  P = toPropertyKey(P);
  anObject(Attributes);
  if (IE8_DOM_DEFINE) try {
    return $defineProperty(O, P, Attributes);
  } catch (error) { /* empty */ }
  if ('get' in Attributes || 'set' in Attributes) throw new $TypeError('Accessors not supported');
  if ('value' in Attributes) O[P] = Attributes.value;
  return O;
};


/***/ }),

/***/ 7347:
/***/ ((__unused_webpack_module, exports, __webpack_require__) => {


var DESCRIPTORS = __webpack_require__(3724);
var call = __webpack_require__(9565);
var propertyIsEnumerableModule = __webpack_require__(8773);
var createPropertyDescriptor = __webpack_require__(6980);
var toIndexedObject = __webpack_require__(5397);
var toPropertyKey = __webpack_require__(6969);
var hasOwn = __webpack_require__(9297);
var IE8_DOM_DEFINE = __webpack_require__(5917);

// eslint-disable-next-line es/no-object-getownpropertydescriptor -- safe
var $getOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;

// `Object.getOwnPropertyDescriptor` method
// https://tc39.es/ecma262/#sec-object.getownpropertydescriptor
exports.f = DESCRIPTORS ? $getOwnPropertyDescriptor : function getOwnPropertyDescriptor(O, P) {
  O = toIndexedObject(O);
  P = toPropertyKey(P);
  if (IE8_DOM_DEFINE) try {
    return $getOwnPropertyDescriptor(O, P);
  } catch (error) { /* empty */ }
  if (hasOwn(O, P)) return createPropertyDescriptor(!call(propertyIsEnumerableModule.f, O, P), O[P]);
};


/***/ }),

/***/ 8480:
/***/ ((__unused_webpack_module, exports, __webpack_require__) => {


var internalObjectKeys = __webpack_require__(1828);
var enumBugKeys = __webpack_require__(8727);

var hiddenKeys = enumBugKeys.concat('length', 'prototype');

// `Object.getOwnPropertyNames` method
// https://tc39.es/ecma262/#sec-object.getownpropertynames
// eslint-disable-next-line es/no-object-getownpropertynames -- safe
exports.f = Object.getOwnPropertyNames || function getOwnPropertyNames(O) {
  return internalObjectKeys(O, hiddenKeys);
};


/***/ }),

/***/ 3717:
/***/ ((__unused_webpack_module, exports) => {


// eslint-disable-next-line es/no-object-getownpropertysymbols -- safe
exports.f = Object.getOwnPropertySymbols;


/***/ }),

/***/ 2787:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var hasOwn = __webpack_require__(9297);
var isCallable = __webpack_require__(4901);
var toObject = __webpack_require__(8981);
var sharedKey = __webpack_require__(6119);
var CORRECT_PROTOTYPE_GETTER = __webpack_require__(2211);

var IE_PROTO = sharedKey('IE_PROTO');
var $Object = Object;
var ObjectPrototype = $Object.prototype;

// `Object.getPrototypeOf` method
// https://tc39.es/ecma262/#sec-object.getprototypeof
// eslint-disable-next-line es/no-object-getprototypeof -- safe
module.exports = CORRECT_PROTOTYPE_GETTER ? $Object.getPrototypeOf : function (O) {
  var object = toObject(O);
  if (hasOwn(object, IE_PROTO)) return object[IE_PROTO];
  var constructor = object.constructor;
  if (isCallable(constructor) && object instanceof constructor) {
    return constructor.prototype;
  } return object instanceof $Object ? ObjectPrototype : null;
};


/***/ }),

/***/ 1625:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var uncurryThis = __webpack_require__(9504);

module.exports = uncurryThis({}.isPrototypeOf);


/***/ }),

/***/ 1828:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var uncurryThis = __webpack_require__(9504);
var hasOwn = __webpack_require__(9297);
var toIndexedObject = __webpack_require__(5397);
var indexOf = (__webpack_require__(9617).indexOf);
var hiddenKeys = __webpack_require__(421);

var push = uncurryThis([].push);

module.exports = function (object, names) {
  var O = toIndexedObject(object);
  var i = 0;
  var result = [];
  var key;
  for (key in O) !hasOwn(hiddenKeys, key) && hasOwn(O, key) && push(result, key);
  // Don't enum bug & hidden keys
  while (names.length > i) if (hasOwn(O, key = names[i++])) {
    ~indexOf(result, key) || push(result, key);
  }
  return result;
};


/***/ }),

/***/ 1072:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var internalObjectKeys = __webpack_require__(1828);
var enumBugKeys = __webpack_require__(8727);

// `Object.keys` method
// https://tc39.es/ecma262/#sec-object.keys
// eslint-disable-next-line es/no-object-keys -- safe
module.exports = Object.keys || function keys(O) {
  return internalObjectKeys(O, enumBugKeys);
};


/***/ }),

/***/ 8773:
/***/ ((__unused_webpack_module, exports) => {


var $propertyIsEnumerable = {}.propertyIsEnumerable;
// eslint-disable-next-line es/no-object-getownpropertydescriptor -- safe
var getOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;

// Nashorn ~ JDK8 bug
var NASHORN_BUG = getOwnPropertyDescriptor && !$propertyIsEnumerable.call({ 1: 2 }, 1);

// `Object.prototype.propertyIsEnumerable` method implementation
// https://tc39.es/ecma262/#sec-object.prototype.propertyisenumerable
exports.f = NASHORN_BUG ? function propertyIsEnumerable(V) {
  var descriptor = getOwnPropertyDescriptor(this, V);
  return !!descriptor && descriptor.enumerable;
} : $propertyIsEnumerable;


/***/ }),

/***/ 2967:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


/* eslint-disable no-proto -- safe */
var uncurryThisAccessor = __webpack_require__(6706);
var isObject = __webpack_require__(34);
var requireObjectCoercible = __webpack_require__(7750);
var aPossiblePrototype = __webpack_require__(3506);

// `Object.setPrototypeOf` method
// https://tc39.es/ecma262/#sec-object.setprototypeof
// Works with __proto__ only. Old v8 can't work with null proto objects.
// eslint-disable-next-line es/no-object-setprototypeof -- safe
module.exports = Object.setPrototypeOf || ('__proto__' in {} ? function () {
  var CORRECT_SETTER = false;
  var test = {};
  var setter;
  try {
    setter = uncurryThisAccessor(Object.prototype, '__proto__', 'set');
    setter(test, []);
    CORRECT_SETTER = test instanceof Array;
  } catch (error) { /* empty */ }
  return function setPrototypeOf(O, proto) {
    requireObjectCoercible(O);
    aPossiblePrototype(proto);
    if (!isObject(O)) return O;
    if (CORRECT_SETTER) setter(O, proto);
    else O.__proto__ = proto;
    return O;
  };
}() : undefined);


/***/ }),

/***/ 4270:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var call = __webpack_require__(9565);
var isCallable = __webpack_require__(4901);
var isObject = __webpack_require__(34);

var $TypeError = TypeError;

// `OrdinaryToPrimitive` abstract operation
// https://tc39.es/ecma262/#sec-ordinarytoprimitive
module.exports = function (input, pref) {
  var fn, val;
  if (pref === 'string' && isCallable(fn = input.toString) && !isObject(val = call(fn, input))) return val;
  if (isCallable(fn = input.valueOf) && !isObject(val = call(fn, input))) return val;
  if (pref !== 'string' && isCallable(fn = input.toString) && !isObject(val = call(fn, input))) return val;
  throw new $TypeError("Can't convert object to primitive value");
};


/***/ }),

/***/ 5031:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var getBuiltIn = __webpack_require__(7751);
var uncurryThis = __webpack_require__(9504);
var getOwnPropertyNamesModule = __webpack_require__(8480);
var getOwnPropertySymbolsModule = __webpack_require__(3717);
var anObject = __webpack_require__(8551);

var concat = uncurryThis([].concat);

// all object keys, includes non-enumerable and symbols
module.exports = getBuiltIn('Reflect', 'ownKeys') || function ownKeys(it) {
  var keys = getOwnPropertyNamesModule.f(anObject(it));
  var getOwnPropertySymbols = getOwnPropertySymbolsModule.f;
  return getOwnPropertySymbols ? concat(keys, getOwnPropertySymbols(it)) : keys;
};


/***/ }),

/***/ 1103:
/***/ ((module) => {


module.exports = function (exec) {
  try {
    return { error: false, value: exec() };
  } catch (error) {
    return { error: true, value: error };
  }
};


/***/ }),

/***/ 7750:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var isNullOrUndefined = __webpack_require__(4117);

var $TypeError = TypeError;

// `RequireObjectCoercible` abstract operation
// https://tc39.es/ecma262/#sec-requireobjectcoercible
module.exports = function (it) {
  if (isNullOrUndefined(it)) throw new $TypeError("Can't call method on " + it);
  return it;
};


/***/ }),

/***/ 9286:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var SetHelpers = __webpack_require__(4402);
var iterate = __webpack_require__(8469);

var Set = SetHelpers.Set;
var add = SetHelpers.add;

module.exports = function (set) {
  var result = new Set();
  iterate(set, function (it) {
    add(result, it);
  });
  return result;
};


/***/ }),

/***/ 3440:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var aSet = __webpack_require__(7080);
var SetHelpers = __webpack_require__(4402);
var clone = __webpack_require__(9286);
var size = __webpack_require__(5170);
var getSetRecord = __webpack_require__(3789);
var iterateSet = __webpack_require__(8469);
var iterateSimple = __webpack_require__(507);

var has = SetHelpers.has;
var remove = SetHelpers.remove;

// `Set.prototype.difference` method
// https://github.com/tc39/proposal-set-methods
module.exports = function difference(other) {
  var O = aSet(this);
  var otherRec = getSetRecord(other);
  var result = clone(O);
  if (size(O) <= otherRec.size) iterateSet(O, function (e) {
    if (otherRec.includes(e)) remove(result, e);
  });
  else iterateSimple(otherRec.getIterator(), function (e) {
    if (has(O, e)) remove(result, e);
  });
  return result;
};


/***/ }),

/***/ 4402:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var uncurryThis = __webpack_require__(9504);

// eslint-disable-next-line es/no-set -- safe
var SetPrototype = Set.prototype;

module.exports = {
  // eslint-disable-next-line es/no-set -- safe
  Set: Set,
  add: uncurryThis(SetPrototype.add),
  has: uncurryThis(SetPrototype.has),
  remove: uncurryThis(SetPrototype['delete']),
  proto: SetPrototype
};


/***/ }),

/***/ 8750:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var aSet = __webpack_require__(7080);
var SetHelpers = __webpack_require__(4402);
var size = __webpack_require__(5170);
var getSetRecord = __webpack_require__(3789);
var iterateSet = __webpack_require__(8469);
var iterateSimple = __webpack_require__(507);

var Set = SetHelpers.Set;
var add = SetHelpers.add;
var has = SetHelpers.has;

// `Set.prototype.intersection` method
// https://github.com/tc39/proposal-set-methods
module.exports = function intersection(other) {
  var O = aSet(this);
  var otherRec = getSetRecord(other);
  var result = new Set();

  if (size(O) > otherRec.size) {
    iterateSimple(otherRec.getIterator(), function (e) {
      if (has(O, e)) add(result, e);
    });
  } else {
    iterateSet(O, function (e) {
      if (otherRec.includes(e)) add(result, e);
    });
  }

  return result;
};


/***/ }),

/***/ 4449:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var aSet = __webpack_require__(7080);
var has = (__webpack_require__(4402).has);
var size = __webpack_require__(5170);
var getSetRecord = __webpack_require__(3789);
var iterateSet = __webpack_require__(8469);
var iterateSimple = __webpack_require__(507);
var iteratorClose = __webpack_require__(9539);

// `Set.prototype.isDisjointFrom` method
// https://tc39.github.io/proposal-set-methods/#Set.prototype.isDisjointFrom
module.exports = function isDisjointFrom(other) {
  var O = aSet(this);
  var otherRec = getSetRecord(other);
  if (size(O) <= otherRec.size) return iterateSet(O, function (e) {
    if (otherRec.includes(e)) return false;
  }, true) !== false;
  var iterator = otherRec.getIterator();
  return iterateSimple(iterator, function (e) {
    if (has(O, e)) return iteratorClose(iterator, 'normal', false);
  }) !== false;
};


/***/ }),

/***/ 3838:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var aSet = __webpack_require__(7080);
var size = __webpack_require__(5170);
var iterate = __webpack_require__(8469);
var getSetRecord = __webpack_require__(3789);

// `Set.prototype.isSubsetOf` method
// https://tc39.github.io/proposal-set-methods/#Set.prototype.isSubsetOf
module.exports = function isSubsetOf(other) {
  var O = aSet(this);
  var otherRec = getSetRecord(other);
  if (size(O) > otherRec.size) return false;
  return iterate(O, function (e) {
    if (!otherRec.includes(e)) return false;
  }, true) !== false;
};


/***/ }),

/***/ 8527:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var aSet = __webpack_require__(7080);
var has = (__webpack_require__(4402).has);
var size = __webpack_require__(5170);
var getSetRecord = __webpack_require__(3789);
var iterateSimple = __webpack_require__(507);
var iteratorClose = __webpack_require__(9539);

// `Set.prototype.isSupersetOf` method
// https://tc39.github.io/proposal-set-methods/#Set.prototype.isSupersetOf
module.exports = function isSupersetOf(other) {
  var O = aSet(this);
  var otherRec = getSetRecord(other);
  if (size(O) < otherRec.size) return false;
  var iterator = otherRec.getIterator();
  return iterateSimple(iterator, function (e) {
    if (!has(O, e)) return iteratorClose(iterator, 'normal', false);
  }) !== false;
};


/***/ }),

/***/ 8469:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var uncurryThis = __webpack_require__(9504);
var iterateSimple = __webpack_require__(507);
var SetHelpers = __webpack_require__(4402);

var Set = SetHelpers.Set;
var SetPrototype = SetHelpers.proto;
var forEach = uncurryThis(SetPrototype.forEach);
var keys = uncurryThis(SetPrototype.keys);
var next = keys(new Set()).next;

module.exports = function (set, fn, interruptible) {
  return interruptible ? iterateSimple({ iterator: keys(set), next: next }, fn) : forEach(set, fn);
};


/***/ }),

/***/ 4916:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var getBuiltIn = __webpack_require__(7751);

var createSetLike = function (size) {
  return {
    size: size,
    has: function () {
      return false;
    },
    keys: function () {
      return {
        next: function () {
          return { done: true };
        }
      };
    }
  };
};

module.exports = function (name) {
  var Set = getBuiltIn('Set');
  try {
    new Set()[name](createSetLike(0));
    try {
      // late spec change, early WebKit ~ Safari 17.0 beta implementation does not pass it
      // https://github.com/tc39/proposal-set-methods/pull/88
      new Set()[name](createSetLike(-1));
      return false;
    } catch (error2) {
      return true;
    }
  } catch (error) {
    return false;
  }
};


/***/ }),

/***/ 5170:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var uncurryThisAccessor = __webpack_require__(6706);
var SetHelpers = __webpack_require__(4402);

module.exports = uncurryThisAccessor(SetHelpers.proto, 'size', 'get') || function (set) {
  return set.size;
};


/***/ }),

/***/ 3650:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var aSet = __webpack_require__(7080);
var SetHelpers = __webpack_require__(4402);
var clone = __webpack_require__(9286);
var getSetRecord = __webpack_require__(3789);
var iterateSimple = __webpack_require__(507);

var add = SetHelpers.add;
var has = SetHelpers.has;
var remove = SetHelpers.remove;

// `Set.prototype.symmetricDifference` method
// https://github.com/tc39/proposal-set-methods
module.exports = function symmetricDifference(other) {
  var O = aSet(this);
  var keysIter = getSetRecord(other).getIterator();
  var result = clone(O);
  iterateSimple(keysIter, function (e) {
    if (has(O, e)) remove(result, e);
    else add(result, e);
  });
  return result;
};


/***/ }),

/***/ 4204:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var aSet = __webpack_require__(7080);
var add = (__webpack_require__(4402).add);
var clone = __webpack_require__(9286);
var getSetRecord = __webpack_require__(3789);
var iterateSimple = __webpack_require__(507);

// `Set.prototype.union` method
// https://github.com/tc39/proposal-set-methods
module.exports = function union(other) {
  var O = aSet(this);
  var keysIter = getSetRecord(other).getIterator();
  var result = clone(O);
  iterateSimple(keysIter, function (it) {
    add(result, it);
  });
  return result;
};


/***/ }),

/***/ 6119:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var shared = __webpack_require__(5745);
var uid = __webpack_require__(3392);

var keys = shared('keys');

module.exports = function (key) {
  return keys[key] || (keys[key] = uid(key));
};


/***/ }),

/***/ 7629:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var IS_PURE = __webpack_require__(6395);
var globalThis = __webpack_require__(4576);
var defineGlobalProperty = __webpack_require__(9433);

var SHARED = '__core-js_shared__';
var store = module.exports = globalThis[SHARED] || defineGlobalProperty(SHARED, {});

(store.versions || (store.versions = [])).push({
  version: '3.39.0',
  mode: IS_PURE ? 'pure' : 'global',
  copyright: '© 2014-2024 Denis Pushkarev (zloirock.ru)',
  license: 'https://github.com/zloirock/core-js/blob/v3.39.0/LICENSE',
  source: 'https://github.com/zloirock/core-js'
});


/***/ }),

/***/ 5745:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var store = __webpack_require__(7629);

module.exports = function (key, value) {
  return store[key] || (store[key] = value || {});
};


/***/ }),

/***/ 1548:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var globalThis = __webpack_require__(4576);
var fails = __webpack_require__(9039);
var V8 = __webpack_require__(9519);
var ENVIRONMENT = __webpack_require__(4215);

var structuredClone = globalThis.structuredClone;

module.exports = !!structuredClone && !fails(function () {
  // prevent V8 ArrayBufferDetaching protector cell invalidation and performance degradation
  // https://github.com/zloirock/core-js/issues/679
  if ((ENVIRONMENT === 'DENO' && V8 > 92) || (ENVIRONMENT === 'NODE' && V8 > 94) || (ENVIRONMENT === 'BROWSER' && V8 > 97)) return false;
  var buffer = new ArrayBuffer(8);
  var clone = structuredClone(buffer, { transfer: [buffer] });
  return buffer.byteLength !== 0 || clone.byteLength !== 8;
});


/***/ }),

/***/ 4495:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


/* eslint-disable es/no-symbol -- required for testing */
var V8_VERSION = __webpack_require__(9519);
var fails = __webpack_require__(9039);
var globalThis = __webpack_require__(4576);

var $String = globalThis.String;

// eslint-disable-next-line es/no-object-getownpropertysymbols -- required for testing
module.exports = !!Object.getOwnPropertySymbols && !fails(function () {
  var symbol = Symbol('symbol detection');
  // Chrome 38 Symbol has incorrect toString conversion
  // `get-own-property-symbols` polyfill symbols converted to object are not Symbol instances
  // nb: Do not call `String` directly to avoid this being optimized out to `symbol+''` which will,
  // of course, fail.
  return !$String(symbol) || !(Object(symbol) instanceof Symbol) ||
    // Chrome 38-40 symbols are not inherited from DOM collections prototypes to instances
    !Symbol.sham && V8_VERSION && V8_VERSION < 41;
});


/***/ }),

/***/ 5610:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var toIntegerOrInfinity = __webpack_require__(1291);

var max = Math.max;
var min = Math.min;

// Helper for a popular repeating case of the spec:
// Let integer be ? ToInteger(index).
// If integer < 0, let result be max((length + integer), 0); else let result be min(integer, length).
module.exports = function (index, length) {
  var integer = toIntegerOrInfinity(index);
  return integer < 0 ? max(integer + length, 0) : min(integer, length);
};


/***/ }),

/***/ 5854:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var toPrimitive = __webpack_require__(2777);

var $TypeError = TypeError;

// `ToBigInt` abstract operation
// https://tc39.es/ecma262/#sec-tobigint
module.exports = function (argument) {
  var prim = toPrimitive(argument, 'number');
  if (typeof prim == 'number') throw new $TypeError("Can't convert number to bigint");
  // eslint-disable-next-line es/no-bigint -- safe
  return BigInt(prim);
};


/***/ }),

/***/ 7696:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var toIntegerOrInfinity = __webpack_require__(1291);
var toLength = __webpack_require__(8014);

var $RangeError = RangeError;

// `ToIndex` abstract operation
// https://tc39.es/ecma262/#sec-toindex
module.exports = function (it) {
  if (it === undefined) return 0;
  var number = toIntegerOrInfinity(it);
  var length = toLength(number);
  if (number !== length) throw new $RangeError('Wrong length or index');
  return length;
};


/***/ }),

/***/ 5397:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


// toObject with fallback for non-array-like ES3 strings
var IndexedObject = __webpack_require__(7055);
var requireObjectCoercible = __webpack_require__(7750);

module.exports = function (it) {
  return IndexedObject(requireObjectCoercible(it));
};


/***/ }),

/***/ 1291:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var trunc = __webpack_require__(741);

// `ToIntegerOrInfinity` abstract operation
// https://tc39.es/ecma262/#sec-tointegerorinfinity
module.exports = function (argument) {
  var number = +argument;
  // eslint-disable-next-line no-self-compare -- NaN check
  return number !== number || number === 0 ? 0 : trunc(number);
};


/***/ }),

/***/ 8014:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var toIntegerOrInfinity = __webpack_require__(1291);

var min = Math.min;

// `ToLength` abstract operation
// https://tc39.es/ecma262/#sec-tolength
module.exports = function (argument) {
  var len = toIntegerOrInfinity(argument);
  return len > 0 ? min(len, 0x1FFFFFFFFFFFFF) : 0; // 2 ** 53 - 1 == 9007199254740991
};


/***/ }),

/***/ 8981:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var requireObjectCoercible = __webpack_require__(7750);

var $Object = Object;

// `ToObject` abstract operation
// https://tc39.es/ecma262/#sec-toobject
module.exports = function (argument) {
  return $Object(requireObjectCoercible(argument));
};


/***/ }),

/***/ 2777:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var call = __webpack_require__(9565);
var isObject = __webpack_require__(34);
var isSymbol = __webpack_require__(757);
var getMethod = __webpack_require__(5966);
var ordinaryToPrimitive = __webpack_require__(4270);
var wellKnownSymbol = __webpack_require__(8227);

var $TypeError = TypeError;
var TO_PRIMITIVE = wellKnownSymbol('toPrimitive');

// `ToPrimitive` abstract operation
// https://tc39.es/ecma262/#sec-toprimitive
module.exports = function (input, pref) {
  if (!isObject(input) || isSymbol(input)) return input;
  var exoticToPrim = getMethod(input, TO_PRIMITIVE);
  var result;
  if (exoticToPrim) {
    if (pref === undefined) pref = 'default';
    result = call(exoticToPrim, input, pref);
    if (!isObject(result) || isSymbol(result)) return result;
    throw new $TypeError("Can't convert object to primitive value");
  }
  if (pref === undefined) pref = 'number';
  return ordinaryToPrimitive(input, pref);
};


/***/ }),

/***/ 6969:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var toPrimitive = __webpack_require__(2777);
var isSymbol = __webpack_require__(757);

// `ToPropertyKey` abstract operation
// https://tc39.es/ecma262/#sec-topropertykey
module.exports = function (argument) {
  var key = toPrimitive(argument, 'string');
  return isSymbol(key) ? key : key + '';
};


/***/ }),

/***/ 2140:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var wellKnownSymbol = __webpack_require__(8227);

var TO_STRING_TAG = wellKnownSymbol('toStringTag');
var test = {};

test[TO_STRING_TAG] = 'z';

module.exports = String(test) === '[object z]';


/***/ }),

/***/ 655:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var classof = __webpack_require__(6955);

var $String = String;

module.exports = function (argument) {
  if (classof(argument) === 'Symbol') throw new TypeError('Cannot convert a Symbol value to a string');
  return $String(argument);
};


/***/ }),

/***/ 6823:
/***/ ((module) => {


var $String = String;

module.exports = function (argument) {
  try {
    return $String(argument);
  } catch (error) {
    return 'Object';
  }
};


/***/ }),

/***/ 3392:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var uncurryThis = __webpack_require__(9504);

var id = 0;
var postfix = Math.random();
var toString = uncurryThis(1.0.toString);

module.exports = function (key) {
  return 'Symbol(' + (key === undefined ? '' : key) + ')_' + toString(++id + postfix, 36);
};


/***/ }),

/***/ 7040:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


/* eslint-disable es/no-symbol -- required for testing */
var NATIVE_SYMBOL = __webpack_require__(4495);

module.exports = NATIVE_SYMBOL &&
  !Symbol.sham &&
  typeof Symbol.iterator == 'symbol';


/***/ }),

/***/ 8686:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var DESCRIPTORS = __webpack_require__(3724);
var fails = __webpack_require__(9039);

// V8 ~ Chrome 36-
// https://bugs.chromium.org/p/v8/issues/detail?id=3334
module.exports = DESCRIPTORS && fails(function () {
  // eslint-disable-next-line es/no-object-defineproperty -- required for testing
  return Object.defineProperty(function () { /* empty */ }, 'prototype', {
    value: 42,
    writable: false
  }).prototype !== 42;
});


/***/ }),

/***/ 2812:
/***/ ((module) => {


var $TypeError = TypeError;

module.exports = function (passed, required) {
  if (passed < required) throw new $TypeError('Not enough arguments');
  return passed;
};


/***/ }),

/***/ 8622:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var globalThis = __webpack_require__(4576);
var isCallable = __webpack_require__(4901);

var WeakMap = globalThis.WeakMap;

module.exports = isCallable(WeakMap) && /native code/.test(String(WeakMap));


/***/ }),

/***/ 8227:
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {


var globalThis = __webpack_require__(4576);
var shared = __webpack_require__(5745);
var hasOwn = __webpack_require__(9297);
var uid = __webpack_require__(3392);
var NATIVE_SYMBOL = __webpack_require__(4495);
var USE_SYMBOL_AS_UID = __webpack_require__(7040);

var Symbol = globalThis.Symbol;
var WellKnownSymbolsStore = shared('wks');
var createWellKnownSymbol = USE_SYMBOL_AS_UID ? Symbol['for'] || Symbol : Symbol && Symbol.withoutSetter || uid;

module.exports = function (name) {
  if (!hasOwn(WellKnownSymbolsStore, name)) {
    WellKnownSymbolsStore[name] = NATIVE_SYMBOL && hasOwn(Symbol, name)
      ? Symbol[name]
      : createWellKnownSymbol('Symbol.' + name);
  } return WellKnownSymbolsStore[name];
};


/***/ }),

/***/ 6573:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var DESCRIPTORS = __webpack_require__(3724);
var defineBuiltInAccessor = __webpack_require__(2106);
var isDetached = __webpack_require__(3238);

var ArrayBufferPrototype = ArrayBuffer.prototype;

// `ArrayBuffer.prototype.detached` getter
// https://tc39.es/ecma262/#sec-get-arraybuffer.prototype.detached
if (DESCRIPTORS && !('detached' in ArrayBufferPrototype)) {
  defineBuiltInAccessor(ArrayBufferPrototype, 'detached', {
    configurable: true,
    get: function detached() {
      return isDetached(this);
    }
  });
}


/***/ }),

/***/ 7936:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var $ = __webpack_require__(6518);
var $transfer = __webpack_require__(5636);

// `ArrayBuffer.prototype.transferToFixedLength` method
// https://tc39.es/proposal-arraybuffer-transfer/#sec-arraybuffer.prototype.transfertofixedlength
if ($transfer) $({ target: 'ArrayBuffer', proto: true }, {
  transferToFixedLength: function transferToFixedLength() {
    return $transfer(this, arguments.length ? arguments[0] : undefined, false);
  }
});


/***/ }),

/***/ 8100:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var $ = __webpack_require__(6518);
var $transfer = __webpack_require__(5636);

// `ArrayBuffer.prototype.transfer` method
// https://tc39.es/proposal-arraybuffer-transfer/#sec-arraybuffer.prototype.transfer
if ($transfer) $({ target: 'ArrayBuffer', proto: true }, {
  transfer: function transfer() {
    return $transfer(this, arguments.length ? arguments[0] : undefined, true);
  }
});


/***/ }),

/***/ 4114:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var $ = __webpack_require__(6518);
var toObject = __webpack_require__(8981);
var lengthOfArrayLike = __webpack_require__(6198);
var setArrayLength = __webpack_require__(4527);
var doesNotExceedSafeInteger = __webpack_require__(6837);
var fails = __webpack_require__(9039);

var INCORRECT_TO_LENGTH = fails(function () {
  return [].push.call({ length: 0x100000000 }, 1) !== 4294967297;
});

// V8 <= 121 and Safari <= 15.4; FF < 23 throws InternalError
// https://bugs.chromium.org/p/v8/issues/detail?id=12681
var properErrorOnNonWritableLength = function () {
  try {
    // eslint-disable-next-line es/no-object-defineproperty -- safe
    Object.defineProperty([], 'length', { writable: false }).push();
  } catch (error) {
    return error instanceof TypeError;
  }
};

var FORCED = INCORRECT_TO_LENGTH || !properErrorOnNonWritableLength();

// `Array.prototype.push` method
// https://tc39.es/ecma262/#sec-array.prototype.push
$({ target: 'Array', proto: true, arity: 1, forced: FORCED }, {
  // eslint-disable-next-line no-unused-vars -- required for `.length`
  push: function push(item) {
    var O = toObject(this);
    var len = lengthOfArrayLike(O);
    var argCount = arguments.length;
    doesNotExceedSafeInteger(len + argCount);
    for (var i = 0; i < argCount; i++) {
      O[len] = arguments[i];
      len++;
    }
    setArrayLength(O, len);
    return len;
  }
});


/***/ }),

/***/ 8111:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var $ = __webpack_require__(6518);
var globalThis = __webpack_require__(4576);
var anInstance = __webpack_require__(679);
var anObject = __webpack_require__(8551);
var isCallable = __webpack_require__(4901);
var getPrototypeOf = __webpack_require__(2787);
var defineBuiltInAccessor = __webpack_require__(2106);
var createProperty = __webpack_require__(4659);
var fails = __webpack_require__(9039);
var hasOwn = __webpack_require__(9297);
var wellKnownSymbol = __webpack_require__(8227);
var IteratorPrototype = (__webpack_require__(7657).IteratorPrototype);
var DESCRIPTORS = __webpack_require__(3724);
var IS_PURE = __webpack_require__(6395);

var CONSTRUCTOR = 'constructor';
var ITERATOR = 'Iterator';
var TO_STRING_TAG = wellKnownSymbol('toStringTag');

var $TypeError = TypeError;
var NativeIterator = globalThis[ITERATOR];

// FF56- have non-standard global helper `Iterator`
var FORCED = IS_PURE
  || !isCallable(NativeIterator)
  || NativeIterator.prototype !== IteratorPrototype
  // FF44- non-standard `Iterator` passes previous tests
  || !fails(function () { NativeIterator({}); });

var IteratorConstructor = function Iterator() {
  anInstance(this, IteratorPrototype);
  if (getPrototypeOf(this) === IteratorPrototype) throw new $TypeError('Abstract class Iterator not directly constructable');
};

var defineIteratorPrototypeAccessor = function (key, value) {
  if (DESCRIPTORS) {
    defineBuiltInAccessor(IteratorPrototype, key, {
      configurable: true,
      get: function () {
        return value;
      },
      set: function (replacement) {
        anObject(this);
        if (this === IteratorPrototype) throw new $TypeError("You can't redefine this property");
        if (hasOwn(this, key)) this[key] = replacement;
        else createProperty(this, key, replacement);
      }
    });
  } else IteratorPrototype[key] = value;
};

if (!hasOwn(IteratorPrototype, TO_STRING_TAG)) defineIteratorPrototypeAccessor(TO_STRING_TAG, ITERATOR);

if (FORCED || !hasOwn(IteratorPrototype, CONSTRUCTOR) || IteratorPrototype[CONSTRUCTOR] === Object) {
  defineIteratorPrototypeAccessor(CONSTRUCTOR, IteratorConstructor);
}

IteratorConstructor.prototype = IteratorPrototype;

// `Iterator` constructor
// https://tc39.es/ecma262/#sec-iterator
$({ global: true, constructor: true, forced: FORCED }, {
  Iterator: IteratorConstructor
});


/***/ }),

/***/ 1148:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var $ = __webpack_require__(6518);
var iterate = __webpack_require__(2652);
var aCallable = __webpack_require__(9306);
var anObject = __webpack_require__(8551);
var getIteratorDirect = __webpack_require__(1767);

// `Iterator.prototype.every` method
// https://tc39.es/ecma262/#sec-iterator.prototype.every
$({ target: 'Iterator', proto: true, real: true }, {
  every: function every(predicate) {
    anObject(this);
    aCallable(predicate);
    var record = getIteratorDirect(this);
    var counter = 0;
    return !iterate(record, function (value, stop) {
      if (!predicate(value, counter++)) return stop();
    }, { IS_RECORD: true, INTERRUPTED: true }).stopped;
  }
});


/***/ }),

/***/ 7588:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var $ = __webpack_require__(6518);
var iterate = __webpack_require__(2652);
var aCallable = __webpack_require__(9306);
var anObject = __webpack_require__(8551);
var getIteratorDirect = __webpack_require__(1767);

// `Iterator.prototype.forEach` method
// https://tc39.es/ecma262/#sec-iterator.prototype.foreach
$({ target: 'Iterator', proto: true, real: true }, {
  forEach: function forEach(fn) {
    anObject(this);
    aCallable(fn);
    var record = getIteratorDirect(this);
    var counter = 0;
    iterate(record, function (value) {
      fn(value, counter++);
    }, { IS_RECORD: true });
  }
});


/***/ }),

/***/ 1701:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var $ = __webpack_require__(6518);
var map = __webpack_require__(713);
var IS_PURE = __webpack_require__(6395);

// `Iterator.prototype.map` method
// https://tc39.es/ecma262/#sec-iterator.prototype.map
$({ target: 'Iterator', proto: true, real: true, forced: IS_PURE }, {
  map: map
});


/***/ }),

/***/ 1689:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var $ = __webpack_require__(6518);
var globalThis = __webpack_require__(4576);
var apply = __webpack_require__(8745);
var slice = __webpack_require__(7680);
var newPromiseCapabilityModule = __webpack_require__(6043);
var aCallable = __webpack_require__(9306);
var perform = __webpack_require__(1103);

var Promise = globalThis.Promise;

var ACCEPT_ARGUMENTS = false;
// Avoiding the use of polyfills of the previous iteration of this proposal
// that does not accept arguments of the callback
var FORCED = !Promise || !Promise['try'] || perform(function () {
  Promise['try'](function (argument) {
    ACCEPT_ARGUMENTS = argument === 8;
  }, 8);
}).error || !ACCEPT_ARGUMENTS;

// `Promise.try` method
// https://tc39.es/ecma262/#sec-promise.try
$({ target: 'Promise', stat: true, forced: FORCED }, {
  'try': function (callbackfn /* , ...args */) {
    var args = arguments.length > 1 ? slice(arguments, 1) : [];
    var promiseCapability = newPromiseCapabilityModule.f(this);
    var result = perform(function () {
      return apply(aCallable(callbackfn), undefined, args);
    });
    (result.error ? promiseCapability.reject : promiseCapability.resolve)(result.value);
    return promiseCapability.promise;
  }
});


/***/ }),

/***/ 7642:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var $ = __webpack_require__(6518);
var difference = __webpack_require__(3440);
var setMethodAcceptSetLike = __webpack_require__(4916);

// `Set.prototype.difference` method
// https://tc39.es/ecma262/#sec-set.prototype.difference
$({ target: 'Set', proto: true, real: true, forced: !setMethodAcceptSetLike('difference') }, {
  difference: difference
});


/***/ }),

/***/ 8004:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var $ = __webpack_require__(6518);
var fails = __webpack_require__(9039);
var intersection = __webpack_require__(8750);
var setMethodAcceptSetLike = __webpack_require__(4916);

var INCORRECT = !setMethodAcceptSetLike('intersection') || fails(function () {
  // eslint-disable-next-line es/no-array-from, es/no-set -- testing
  return String(Array.from(new Set([1, 2, 3]).intersection(new Set([3, 2])))) !== '3,2';
});

// `Set.prototype.intersection` method
// https://tc39.es/ecma262/#sec-set.prototype.intersection
$({ target: 'Set', proto: true, real: true, forced: INCORRECT }, {
  intersection: intersection
});


/***/ }),

/***/ 3853:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var $ = __webpack_require__(6518);
var isDisjointFrom = __webpack_require__(4449);
var setMethodAcceptSetLike = __webpack_require__(4916);

// `Set.prototype.isDisjointFrom` method
// https://tc39.es/ecma262/#sec-set.prototype.isdisjointfrom
$({ target: 'Set', proto: true, real: true, forced: !setMethodAcceptSetLike('isDisjointFrom') }, {
  isDisjointFrom: isDisjointFrom
});


/***/ }),

/***/ 5876:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var $ = __webpack_require__(6518);
var isSubsetOf = __webpack_require__(3838);
var setMethodAcceptSetLike = __webpack_require__(4916);

// `Set.prototype.isSubsetOf` method
// https://tc39.es/ecma262/#sec-set.prototype.issubsetof
$({ target: 'Set', proto: true, real: true, forced: !setMethodAcceptSetLike('isSubsetOf') }, {
  isSubsetOf: isSubsetOf
});


/***/ }),

/***/ 2475:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var $ = __webpack_require__(6518);
var isSupersetOf = __webpack_require__(8527);
var setMethodAcceptSetLike = __webpack_require__(4916);

// `Set.prototype.isSupersetOf` method
// https://tc39.es/ecma262/#sec-set.prototype.issupersetof
$({ target: 'Set', proto: true, real: true, forced: !setMethodAcceptSetLike('isSupersetOf') }, {
  isSupersetOf: isSupersetOf
});


/***/ }),

/***/ 5024:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var $ = __webpack_require__(6518);
var symmetricDifference = __webpack_require__(3650);
var setMethodAcceptSetLike = __webpack_require__(4916);

// `Set.prototype.symmetricDifference` method
// https://tc39.es/ecma262/#sec-set.prototype.symmetricdifference
$({ target: 'Set', proto: true, real: true, forced: !setMethodAcceptSetLike('symmetricDifference') }, {
  symmetricDifference: symmetricDifference
});


/***/ }),

/***/ 1698:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var $ = __webpack_require__(6518);
var union = __webpack_require__(4204);
var setMethodAcceptSetLike = __webpack_require__(4916);

// `Set.prototype.union` method
// https://tc39.es/ecma262/#sec-set.prototype.union
$({ target: 'Set', proto: true, real: true, forced: !setMethodAcceptSetLike('union') }, {
  union: union
});


/***/ }),

/***/ 7467:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var arrayToReversed = __webpack_require__(7628);
var ArrayBufferViewCore = __webpack_require__(4644);

var aTypedArray = ArrayBufferViewCore.aTypedArray;
var exportTypedArrayMethod = ArrayBufferViewCore.exportTypedArrayMethod;
var getTypedArrayConstructor = ArrayBufferViewCore.getTypedArrayConstructor;

// `%TypedArray%.prototype.toReversed` method
// https://tc39.es/ecma262/#sec-%typedarray%.prototype.toreversed
exportTypedArrayMethod('toReversed', function toReversed() {
  return arrayToReversed(aTypedArray(this), getTypedArrayConstructor(this));
});


/***/ }),

/***/ 4732:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var ArrayBufferViewCore = __webpack_require__(4644);
var uncurryThis = __webpack_require__(9504);
var aCallable = __webpack_require__(9306);
var arrayFromConstructorAndList = __webpack_require__(5370);

var aTypedArray = ArrayBufferViewCore.aTypedArray;
var getTypedArrayConstructor = ArrayBufferViewCore.getTypedArrayConstructor;
var exportTypedArrayMethod = ArrayBufferViewCore.exportTypedArrayMethod;
var sort = uncurryThis(ArrayBufferViewCore.TypedArrayPrototype.sort);

// `%TypedArray%.prototype.toSorted` method
// https://tc39.es/ecma262/#sec-%typedarray%.prototype.tosorted
exportTypedArrayMethod('toSorted', function toSorted(compareFn) {
  if (compareFn !== undefined) aCallable(compareFn);
  var O = aTypedArray(this);
  var A = arrayFromConstructorAndList(getTypedArrayConstructor(O), O);
  return sort(A, compareFn);
});


/***/ }),

/***/ 9577:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var arrayWith = __webpack_require__(9928);
var ArrayBufferViewCore = __webpack_require__(4644);
var isBigIntArray = __webpack_require__(1108);
var toIntegerOrInfinity = __webpack_require__(1291);
var toBigInt = __webpack_require__(5854);

var aTypedArray = ArrayBufferViewCore.aTypedArray;
var getTypedArrayConstructor = ArrayBufferViewCore.getTypedArrayConstructor;
var exportTypedArrayMethod = ArrayBufferViewCore.exportTypedArrayMethod;

var PROPER_ORDER = !!function () {
  try {
    // eslint-disable-next-line no-throw-literal, es/no-typed-arrays, es/no-array-prototype-with -- required for testing
    new Int8Array(1)['with'](2, { valueOf: function () { throw 8; } });
  } catch (error) {
    // some early implementations, like WebKit, does not follow the final semantic
    // https://github.com/tc39/proposal-change-array-by-copy/pull/86
    return error === 8;
  }
}();

// `%TypedArray%.prototype.with` method
// https://tc39.es/ecma262/#sec-%typedarray%.prototype.with
exportTypedArrayMethod('with', { 'with': function (index, value) {
  var O = aTypedArray(this);
  var relativeIndex = toIntegerOrInfinity(index);
  var actualValue = isBigIntArray(O) ? toBigInt(value) : +value;
  return arrayWith(O, getTypedArrayConstructor(O), relativeIndex, actualValue);
} }['with'], !PROPER_ORDER);


/***/ }),

/***/ 8992:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


// TODO: Remove from `core-js@4`
__webpack_require__(8111);


/***/ }),

/***/ 3215:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


// TODO: Remove from `core-js@4`
__webpack_require__(1148);


/***/ }),

/***/ 3949:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


// TODO: Remove from `core-js@4`
__webpack_require__(7588);


/***/ }),

/***/ 1454:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


// TODO: Remove from `core-js@4`
__webpack_require__(1701);


/***/ }),

/***/ 5247:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


// TODO: Remove from `core-js@4`
__webpack_require__(1689);


/***/ }),

/***/ 4979:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var $ = __webpack_require__(6518);
var globalThis = __webpack_require__(4576);
var getBuiltIn = __webpack_require__(7751);
var createPropertyDescriptor = __webpack_require__(6980);
var defineProperty = (__webpack_require__(4913).f);
var hasOwn = __webpack_require__(9297);
var anInstance = __webpack_require__(679);
var inheritIfRequired = __webpack_require__(3167);
var normalizeStringArgument = __webpack_require__(2603);
var DOMExceptionConstants = __webpack_require__(5002);
var clearErrorStack = __webpack_require__(8574);
var DESCRIPTORS = __webpack_require__(3724);
var IS_PURE = __webpack_require__(6395);

var DOM_EXCEPTION = 'DOMException';
var Error = getBuiltIn('Error');
var NativeDOMException = getBuiltIn(DOM_EXCEPTION);

var $DOMException = function DOMException() {
  anInstance(this, DOMExceptionPrototype);
  var argumentsLength = arguments.length;
  var message = normalizeStringArgument(argumentsLength < 1 ? undefined : arguments[0]);
  var name = normalizeStringArgument(argumentsLength < 2 ? undefined : arguments[1], 'Error');
  var that = new NativeDOMException(message, name);
  var error = new Error(message);
  error.name = DOM_EXCEPTION;
  defineProperty(that, 'stack', createPropertyDescriptor(1, clearErrorStack(error.stack, 1)));
  inheritIfRequired(that, this, $DOMException);
  return that;
};

var DOMExceptionPrototype = $DOMException.prototype = NativeDOMException.prototype;

var ERROR_HAS_STACK = 'stack' in new Error(DOM_EXCEPTION);
var DOM_EXCEPTION_HAS_STACK = 'stack' in new NativeDOMException(1, 2);

// eslint-disable-next-line es/no-object-getownpropertydescriptor -- safe
var descriptor = NativeDOMException && DESCRIPTORS && Object.getOwnPropertyDescriptor(globalThis, DOM_EXCEPTION);

// Bun ~ 0.1.1 DOMException have incorrect descriptor and we can't redefine it
// https://github.com/Jarred-Sumner/bun/issues/399
var BUGGY_DESCRIPTOR = !!descriptor && !(descriptor.writable && descriptor.configurable);

var FORCED_CONSTRUCTOR = ERROR_HAS_STACK && !BUGGY_DESCRIPTOR && !DOM_EXCEPTION_HAS_STACK;

// `DOMException` constructor patch for `.stack` where it's required
// https://webidl.spec.whatwg.org/#es-DOMException-specialness
$({ global: true, constructor: true, forced: IS_PURE || FORCED_CONSTRUCTOR }, { // TODO: fix export logic
  DOMException: FORCED_CONSTRUCTOR ? $DOMException : NativeDOMException
});

var PolyfilledDOMException = getBuiltIn(DOM_EXCEPTION);
var PolyfilledDOMExceptionPrototype = PolyfilledDOMException.prototype;

if (PolyfilledDOMExceptionPrototype.constructor !== PolyfilledDOMException) {
  if (!IS_PURE) {
    defineProperty(PolyfilledDOMExceptionPrototype, 'constructor', createPropertyDescriptor(1, PolyfilledDOMException));
  }

  for (var key in DOMExceptionConstants) if (hasOwn(DOMExceptionConstants, key)) {
    var constant = DOMExceptionConstants[key];
    var constantName = constant.s;
    if (!hasOwn(PolyfilledDOMException, constantName)) {
      defineProperty(PolyfilledDOMException, constantName, createPropertyDescriptor(6, constant.c));
    }
  }
}


/***/ }),

/***/ 3611:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var $ = __webpack_require__(6518);
var globalThis = __webpack_require__(4576);
var defineBuiltInAccessor = __webpack_require__(2106);
var DESCRIPTORS = __webpack_require__(3724);

var $TypeError = TypeError;
// eslint-disable-next-line es/no-object-defineproperty -- safe
var defineProperty = Object.defineProperty;
var INCORRECT_VALUE = globalThis.self !== globalThis;

// `self` getter
// https://html.spec.whatwg.org/multipage/window-object.html#dom-self
try {
  if (DESCRIPTORS) {
    // eslint-disable-next-line es/no-object-getownpropertydescriptor -- safe
    var descriptor = Object.getOwnPropertyDescriptor(globalThis, 'self');
    // some engines have `self`, but with incorrect descriptor
    // https://github.com/denoland/deno/issues/15765
    if (INCORRECT_VALUE || !descriptor || !descriptor.get || !descriptor.enumerable) {
      defineBuiltInAccessor(globalThis, 'self', {
        get: function self() {
          return globalThis;
        },
        set: function self(value) {
          if (this !== globalThis) throw new $TypeError('Illegal invocation');
          defineProperty(globalThis, 'self', {
            value: value,
            writable: true,
            configurable: true,
            enumerable: true
          });
        },
        configurable: true,
        enumerable: true
      });
    }
  } else $({ global: true, simple: true, forced: INCORRECT_VALUE }, {
    self: globalThis
  });
} catch (error) { /* empty */ }


/***/ }),

/***/ 4603:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var defineBuiltIn = __webpack_require__(6840);
var uncurryThis = __webpack_require__(9504);
var toString = __webpack_require__(655);
var validateArgumentsLength = __webpack_require__(2812);

var $URLSearchParams = URLSearchParams;
var URLSearchParamsPrototype = $URLSearchParams.prototype;
var append = uncurryThis(URLSearchParamsPrototype.append);
var $delete = uncurryThis(URLSearchParamsPrototype['delete']);
var forEach = uncurryThis(URLSearchParamsPrototype.forEach);
var push = uncurryThis([].push);
var params = new $URLSearchParams('a=1&a=2&b=3');

params['delete']('a', 1);
// `undefined` case is a Chromium 117 bug
// https://bugs.chromium.org/p/v8/issues/detail?id=14222
params['delete']('b', undefined);

if (params + '' !== 'a=2') {
  defineBuiltIn(URLSearchParamsPrototype, 'delete', function (name /* , value */) {
    var length = arguments.length;
    var $value = length < 2 ? undefined : arguments[1];
    if (length && $value === undefined) return $delete(this, name);
    var entries = [];
    forEach(this, function (v, k) { // also validates `this`
      push(entries, { key: k, value: v });
    });
    validateArgumentsLength(length, 1);
    var key = toString(name);
    var value = toString($value);
    var index = 0;
    var dindex = 0;
    var found = false;
    var entriesLength = entries.length;
    var entry;
    while (index < entriesLength) {
      entry = entries[index++];
      if (found || entry.key === key) {
        found = true;
        $delete(this, entry.key);
      } else dindex++;
    }
    while (dindex < entriesLength) {
      entry = entries[dindex++];
      if (!(entry.key === key && entry.value === value)) append(this, entry.key, entry.value);
    }
  }, { enumerable: true, unsafe: true });
}


/***/ }),

/***/ 7566:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var defineBuiltIn = __webpack_require__(6840);
var uncurryThis = __webpack_require__(9504);
var toString = __webpack_require__(655);
var validateArgumentsLength = __webpack_require__(2812);

var $URLSearchParams = URLSearchParams;
var URLSearchParamsPrototype = $URLSearchParams.prototype;
var getAll = uncurryThis(URLSearchParamsPrototype.getAll);
var $has = uncurryThis(URLSearchParamsPrototype.has);
var params = new $URLSearchParams('a=1');

// `undefined` case is a Chromium 117 bug
// https://bugs.chromium.org/p/v8/issues/detail?id=14222
if (params.has('a', 2) || !params.has('a', undefined)) {
  defineBuiltIn(URLSearchParamsPrototype, 'has', function has(name /* , value */) {
    var length = arguments.length;
    var $value = length < 2 ? undefined : arguments[1];
    if (length && $value === undefined) return $has(this, name);
    var values = getAll(this, name); // also validates `this`
    validateArgumentsLength(length, 1);
    var value = toString($value);
    var index = 0;
    while (index < values.length) {
      if (values[index++] === value) return true;
    } return false;
  }, { enumerable: true, unsafe: true });
}


/***/ }),

/***/ 8721:
/***/ ((__unused_webpack_module, __unused_webpack_exports, __webpack_require__) => {


var DESCRIPTORS = __webpack_require__(3724);
var uncurryThis = __webpack_require__(9504);
var defineBuiltInAccessor = __webpack_require__(2106);

var URLSearchParamsPrototype = URLSearchParams.prototype;
var forEach = uncurryThis(URLSearchParamsPrototype.forEach);

// `URLSearchParams.prototype.size` getter
// https://github.com/whatwg/url/pull/734
if (DESCRIPTORS && !('size' in URLSearchParamsPrototype)) {
  defineBuiltInAccessor(URLSearchParamsPrototype, 'size', {
    get: function size() {
      var count = 0;
      forEach(this, function () { count++; });
      return count;
    },
    configurable: true,
    enumerable: true
  });
}


/***/ })

/******/ });
/************************************************************************/
/******/ // The module cache
/******/ var __webpack_module_cache__ = {};
/******/ 
/******/ // The require function
/******/ function __webpack_require__(moduleId) {
/******/ 	// Check if module is in cache
/******/ 	var cachedModule = __webpack_module_cache__[moduleId];
/******/ 	if (cachedModule !== undefined) {
/******/ 		return cachedModule.exports;
/******/ 	}
/******/ 	// Create a new module (and put it into the cache)
/******/ 	var module = __webpack_module_cache__[moduleId] = {
/******/ 		// no module.id needed
/******/ 		// no module.loaded needed
/******/ 		exports: {}
/******/ 	};
/******/ 
/******/ 	// Execute the module function
/******/ 	__webpack_modules__[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/ 
/******/ 	// Return the exports of the module
/******/ 	return module.exports;
/******/ }
/******/ 
/************************************************************************/
/******/ /* webpack/runtime/define property getters */
/******/ (() => {
/******/ 	// define getter functions for harmony exports
/******/ 	__webpack_require__.d = (exports, definition) => {
/******/ 		for(var key in definition) {
/******/ 			if(__webpack_require__.o(definition, key) && !__webpack_require__.o(exports, key)) {
/******/ 				Object.defineProperty(exports, key, { enumerable: true, get: definition[key] });
/******/ 			}
/******/ 		}
/******/ 	};
/******/ })();
/******/ 
/******/ /* webpack/runtime/hasOwnProperty shorthand */
/******/ (() => {
/******/ 	__webpack_require__.o = (obj, prop) => (Object.prototype.hasOwnProperty.call(obj, prop))
/******/ })();
/******/ 
/************************************************************************/
var __webpack_exports__ = globalThis.pdfjsImageDecoders = {};

// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  Jbig2Error: () => (/* reexport */ Jbig2Error),
  Jbig2Image: () => (/* reexport */ Jbig2Image),
  JpegError: () => (/* reexport */ JpegError),
  JpegImage: () => (/* reexport */ JpegImage),
  JpxError: () => (/* reexport */ JpxError),
  JpxImage: () => (/* reexport */ JpxImage),
  VerbosityLevel: () => (/* reexport */ VerbosityLevel),
  getVerbosityLevel: () => (/* reexport */ getVerbosityLevel),
  setVerbosityLevel: () => (/* reexport */ setVerbosityLevel)
});

// EXTERNAL MODULE: ./node_modules/core-js/modules/es.array.push.js
var es_array_push = __webpack_require__(4114);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.array-buffer.detached.js
var es_array_buffer_detached = __webpack_require__(6573);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.array-buffer.transfer.js
var es_array_buffer_transfer = __webpack_require__(8100);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.array-buffer.transfer-to-fixed-length.js
var es_array_buffer_transfer_to_fixed_length = __webpack_require__(7936);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.typed-array.to-reversed.js
var es_typed_array_to_reversed = __webpack_require__(7467);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.typed-array.to-sorted.js
var es_typed_array_to_sorted = __webpack_require__(4732);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.typed-array.with.js
var es_typed_array_with = __webpack_require__(9577);
// EXTERNAL MODULE: ./node_modules/core-js/modules/esnext.promise.try.js
var esnext_promise_try = __webpack_require__(5247);
// EXTERNAL MODULE: ./node_modules/core-js/modules/web.dom-exception.stack.js
var web_dom_exception_stack = __webpack_require__(4979);
// EXTERNAL MODULE: ./node_modules/core-js/modules/web.url-search-params.delete.js
var web_url_search_params_delete = __webpack_require__(4603);
// EXTERNAL MODULE: ./node_modules/core-js/modules/web.url-search-params.has.js
var web_url_search_params_has = __webpack_require__(7566);
// EXTERNAL MODULE: ./node_modules/core-js/modules/web.url-search-params.size.js
var web_url_search_params_size = __webpack_require__(8721);
;// ./src/shared/util.js












const isNodeJS = typeof process === "object" && process + "" === "[object process]" && !process.versions.nw && !(process.versions.electron && process.type && process.type !== "browser");
const IDENTITY_MATRIX = (/* unused pure expression or super */ null && ([1, 0, 0, 1, 0, 0]));
const FONT_IDENTITY_MATRIX = (/* unused pure expression or super */ null && ([0.001, 0, 0, 0.001, 0, 0]));
const MAX_IMAGE_SIZE_TO_CACHE = 10e6;
const LINE_FACTOR = 1.35;
const LINE_DESCENT_FACTOR = 0.35;
const BASELINE_FACTOR = LINE_DESCENT_FACTOR / LINE_FACTOR;
const RenderingIntentFlag = {
  ANY: 0x01,
  DISPLAY: 0x02,
  PRINT: 0x04,
  SAVE: 0x08,
  ANNOTATIONS_FORMS: 0x10,
  ANNOTATIONS_STORAGE: 0x20,
  ANNOTATIONS_DISABLE: 0x40,
  IS_EDITING: 0x80,
  OPLIST: 0x100
};
const AnnotationMode = {
  DISABLE: 0,
  ENABLE: 1,
  ENABLE_FORMS: 2,
  ENABLE_STORAGE: 3
};
const util_AnnotationEditorPrefix = "pdfjs_internal_editor_";
const AnnotationEditorType = {
  DISABLE: -1,
  NONE: 0,
  FREETEXT: 3,
  HIGHLIGHT: 9,
  STAMP: 13,
  INK: 15
};
const AnnotationEditorParamsType = {
  RESIZE: 1,
  CREATE: 2,
  FREETEXT_SIZE: 11,
  FREETEXT_COLOR: 12,
  FREETEXT_OPACITY: 13,
  INK_COLOR: 21,
  INK_THICKNESS: 22,
  INK_OPACITY: 23,
  HIGHLIGHT_COLOR: 31,
  HIGHLIGHT_DEFAULT_COLOR: 32,
  HIGHLIGHT_THICKNESS: 33,
  HIGHLIGHT_FREE: 34,
  HIGHLIGHT_SHOW_ALL: 35,
  DRAW_STEP: 41
};
const PermissionFlag = {
  PRINT: 0x04,
  MODIFY_CONTENTS: 0x08,
  COPY: 0x10,
  MODIFY_ANNOTATIONS: 0x20,
  FILL_INTERACTIVE_FORMS: 0x100,
  COPY_FOR_ACCESSIBILITY: 0x200,
  ASSEMBLE: 0x400,
  PRINT_HIGH_QUALITY: 0x800
};
const TextRenderingMode = {
  FILL: 0,
  STROKE: 1,
  FILL_STROKE: 2,
  INVISIBLE: 3,
  FILL_ADD_TO_PATH: 4,
  STROKE_ADD_TO_PATH: 5,
  FILL_STROKE_ADD_TO_PATH: 6,
  ADD_TO_PATH: 7,
  FILL_STROKE_MASK: 3,
  ADD_TO_PATH_FLAG: 4
};
const util_ImageKind = {
  GRAYSCALE_1BPP: 1,
  RGB_24BPP: 2,
  RGBA_32BPP: 3
};
const AnnotationType = {
  TEXT: 1,
  LINK: 2,
  FREETEXT: 3,
  LINE: 4,
  SQUARE: 5,
  CIRCLE: 6,
  POLYGON: 7,
  POLYLINE: 8,
  HIGHLIGHT: 9,
  UNDERLINE: 10,
  SQUIGGLY: 11,
  STRIKEOUT: 12,
  STAMP: 13,
  CARET: 14,
  INK: 15,
  POPUP: 16,
  FILEATTACHMENT: 17,
  SOUND: 18,
  MOVIE: 19,
  WIDGET: 20,
  SCREEN: 21,
  PRINTERMARK: 22,
  TRAPNET: 23,
  WATERMARK: 24,
  THREED: 25,
  REDACT: 26
};
const AnnotationReplyType = {
  GROUP: "Group",
  REPLY: "R"
};
const AnnotationFlag = {
  INVISIBLE: 0x01,
  HIDDEN: 0x02,
  PRINT: 0x04,
  NOZOOM: 0x08,
  NOROTATE: 0x10,
  NOVIEW: 0x20,
  READONLY: 0x40,
  LOCKED: 0x80,
  TOGGLENOVIEW: 0x100,
  LOCKEDCONTENTS: 0x200
};
const AnnotationFieldFlag = {
  READONLY: 0x0000001,
  REQUIRED: 0x0000002,
  NOEXPORT: 0x0000004,
  MULTILINE: 0x0001000,
  PASSWORD: 0x0002000,
  NOTOGGLETOOFF: 0x0004000,
  RADIO: 0x0008000,
  PUSHBUTTON: 0x0010000,
  COMBO: 0x0020000,
  EDIT: 0x0040000,
  SORT: 0x0080000,
  FILESELECT: 0x0100000,
  MULTISELECT: 0x0200000,
  DONOTSPELLCHECK: 0x0400000,
  DONOTSCROLL: 0x0800000,
  COMB: 0x1000000,
  RICHTEXT: 0x2000000,
  RADIOSINUNISON: 0x2000000,
  COMMITONSELCHANGE: 0x4000000
};
const AnnotationBorderStyleType = {
  SOLID: 1,
  DASHED: 2,
  BEVELED: 3,
  INSET: 4,
  UNDERLINE: 5
};
const AnnotationActionEventType = {
  E: "Mouse Enter",
  X: "Mouse Exit",
  D: "Mouse Down",
  U: "Mouse Up",
  Fo: "Focus",
  Bl: "Blur",
  PO: "PageOpen",
  PC: "PageClose",
  PV: "PageVisible",
  PI: "PageInvisible",
  K: "Keystroke",
  F: "Format",
  V: "Validate",
  C: "Calculate"
};
const DocumentActionEventType = {
  WC: "WillClose",
  WS: "WillSave",
  DS: "DidSave",
  WP: "WillPrint",
  DP: "DidPrint"
};
const PageActionEventType = {
  O: "PageOpen",
  C: "PageClose"
};
const VerbosityLevel = {
  ERRORS: 0,
  WARNINGS: 1,
  INFOS: 5
};
const OPS = {
  dependency: 1,
  setLineWidth: 2,
  setLineCap: 3,
  setLineJoin: 4,
  setMiterLimit: 5,
  setDash: 6,
  setRenderingIntent: 7,
  setFlatness: 8,
  setGState: 9,
  save: 10,
  restore: 11,
  transform: 12,
  moveTo: 13,
  lineTo: 14,
  curveTo: 15,
  curveTo2: 16,
  curveTo3: 17,
  closePath: 18,
  rectangle: 19,
  stroke: 20,
  closeStroke: 21,
  fill: 22,
  eoFill: 23,
  fillStroke: 24,
  eoFillStroke: 25,
  closeFillStroke: 26,
  closeEOFillStroke: 27,
  endPath: 28,
  clip: 29,
  eoClip: 30,
  beginText: 31,
  endText: 32,
  setCharSpacing: 33,
  setWordSpacing: 34,
  setHScale: 35,
  setLeading: 36,
  setFont: 37,
  setTextRenderingMode: 38,
  setTextRise: 39,
  moveText: 40,
  setLeadingMoveText: 41,
  setTextMatrix: 42,
  nextLine: 43,
  showText: 44,
  showSpacedText: 45,
  nextLineShowText: 46,
  nextLineSetSpacingShowText: 47,
  setCharWidth: 48,
  setCharWidthAndBounds: 49,
  setStrokeColorSpace: 50,
  setFillColorSpace: 51,
  setStrokeColor: 52,
  setStrokeColorN: 53,
  setFillColor: 54,
  setFillColorN: 55,
  setStrokeGray: 56,
  setFillGray: 57,
  setStrokeRGBColor: 58,
  setFillRGBColor: 59,
  setStrokeCMYKColor: 60,
  setFillCMYKColor: 61,
  shadingFill: 62,
  beginInlineImage: 63,
  beginImageData: 64,
  endInlineImage: 65,
  paintXObject: 66,
  markPoint: 67,
  markPointProps: 68,
  beginMarkedContent: 69,
  beginMarkedContentProps: 70,
  endMarkedContent: 71,
  beginCompat: 72,
  endCompat: 73,
  paintFormXObjectBegin: 74,
  paintFormXObjectEnd: 75,
  beginGroup: 76,
  endGroup: 77,
  beginAnnotation: 80,
  endAnnotation: 81,
  paintImageMaskXObject: 83,
  paintImageMaskXObjectGroup: 84,
  paintImageXObject: 85,
  paintInlineImageXObject: 86,
  paintInlineImageXObjectGroup: 87,
  paintImageXObjectRepeat: 88,
  paintImageMaskXObjectRepeat: 89,
  paintSolidColorImageMask: 90,
  constructPath: 91,
  setStrokeTransparent: 92,
  setFillTransparent: 93
};
const PasswordResponses = {
  NEED_PASSWORD: 1,
  INCORRECT_PASSWORD: 2
};
let verbosity = VerbosityLevel.WARNINGS;
function setVerbosityLevel(level) {
  if (Number.isInteger(level)) {
    verbosity = level;
  }
}
function getVerbosityLevel() {
  return verbosity;
}
function info(msg) {
  if (verbosity >= VerbosityLevel.INFOS) {
    console.log(`Info: ${msg}`);
  }
}
function util_warn(msg) {
  if (verbosity >= VerbosityLevel.WARNINGS) {
    console.log(`Warning: ${msg}`);
  }
}
function unreachable(msg) {
  throw new Error(msg);
}
function util_assert(cond, msg) {
  if (!cond) {
    unreachable(msg);
  }
}
function _isValidProtocol(url) {
  switch (url?.protocol) {
    case "http:":
    case "https:":
    case "ftp:":
    case "mailto:":
    case "tel:":
      return true;
    default:
      return false;
  }
}
function createValidAbsoluteUrl(url, baseUrl = null, options = null) {
  if (!url) {
    return null;
  }
  try {
    if (options && typeof url === "string") {
      if (options.addDefaultProtocol && url.startsWith("www.")) {
        const dots = url.match(/\./g);
        if (dots?.length >= 2) {
          url = `http://${url}`;
        }
      }
      if (options.tryConvertEncoding) {
        try {
          url = stringToUTF8String(url);
        } catch {}
      }
    }
    const absoluteUrl = baseUrl ? new URL(url, baseUrl) : new URL(url);
    if (_isValidProtocol(absoluteUrl)) {
      return absoluteUrl;
    }
  } catch {}
  return null;
}
function shadow(obj, prop, value, nonSerializable = false) {
  Object.defineProperty(obj, prop, {
    value,
    enumerable: !nonSerializable,
    configurable: true,
    writable: false
  });
  return value;
}
const BaseException = function BaseExceptionClosure() {
  function BaseException(message, name) {
    this.message = message;
    this.name = name;
  }
  BaseException.prototype = new Error();
  BaseException.constructor = BaseException;
  return BaseException;
}();
class PasswordException extends BaseException {
  constructor(msg, code) {
    super(msg, "PasswordException");
    this.code = code;
  }
}
class UnknownErrorException extends BaseException {
  constructor(msg, details) {
    super(msg, "UnknownErrorException");
    this.details = details;
  }
}
class InvalidPDFException extends BaseException {
  constructor(msg) {
    super(msg, "InvalidPDFException");
  }
}
class MissingPDFException extends BaseException {
  constructor(msg) {
    super(msg, "MissingPDFException");
  }
}
class UnexpectedResponseException extends BaseException {
  constructor(msg, status) {
    super(msg, "UnexpectedResponseException");
    this.status = status;
  }
}
class FormatError extends BaseException {
  constructor(msg) {
    super(msg, "FormatError");
  }
}
class AbortException extends BaseException {
  constructor(msg) {
    super(msg, "AbortException");
  }
}
function bytesToString(bytes) {
  if (typeof bytes !== "object" || bytes?.length === undefined) {
    unreachable("Invalid argument for bytesToString");
  }
  const length = bytes.length;
  const MAX_ARGUMENT_COUNT = 8192;
  if (length < MAX_ARGUMENT_COUNT) {
    return String.fromCharCode.apply(null, bytes);
  }
  const strBuf = [];
  for (let i = 0; i < length; i += MAX_ARGUMENT_COUNT) {
    const chunkEnd = Math.min(i + MAX_ARGUMENT_COUNT, length);
    const chunk = bytes.subarray(i, chunkEnd);
    strBuf.push(String.fromCharCode.apply(null, chunk));
  }
  return strBuf.join("");
}
function stringToBytes(str) {
  if (typeof str !== "string") {
    unreachable("Invalid argument for stringToBytes");
  }
  const length = str.length;
  const bytes = new Uint8Array(length);
  for (let i = 0; i < length; ++i) {
    bytes[i] = str.charCodeAt(i) & 0xff;
  }
  return bytes;
}
function string32(value) {
  return String.fromCharCode(value >> 24 & 0xff, value >> 16 & 0xff, value >> 8 & 0xff, value & 0xff);
}
function util_objectSize(obj) {
  return Object.keys(obj).length;
}
function objectFromMap(map) {
  const obj = Object.create(null);
  for (const [key, value] of map) {
    obj[key] = value;
  }
  return obj;
}
function isLittleEndian() {
  const buffer8 = new Uint8Array(4);
  buffer8[0] = 1;
  const view32 = new Uint32Array(buffer8.buffer, 0, 1);
  return view32[0] === 1;
}
function isEvalSupported() {
  try {
    new Function("");
    return true;
  } catch {
    return false;
  }
}
class util_FeatureTest {
  static get isLittleEndian() {
    return shadow(this, "isLittleEndian", isLittleEndian());
  }
  static get isEvalSupported() {
    return shadow(this, "isEvalSupported", isEvalSupported());
  }
  static get isOffscreenCanvasSupported() {
    return shadow(this, "isOffscreenCanvasSupported", typeof OffscreenCanvas !== "undefined");
  }
  static get isImageDecoderSupported() {
    return shadow(this, "isImageDecoderSupported", typeof ImageDecoder !== "undefined");
  }
  static get platform() {
    if (typeof navigator !== "undefined" && typeof navigator?.platform === "string") {
      return shadow(this, "platform", {
        isMac: navigator.platform.includes("Mac"),
        isWindows: navigator.platform.includes("Win"),
        isFirefox: typeof navigator?.userAgent === "string" && navigator.userAgent.includes("Firefox")
      });
    }
    return shadow(this, "platform", {
      isMac: false,
      isWindows: false,
      isFirefox: false
    });
  }
  static get isCSSRoundSupported() {
    return shadow(this, "isCSSRoundSupported", globalThis.CSS?.supports?.("width: round(1.5px, 1px)"));
  }
}
const util_hexNumbers = Array.from(Array(256).keys(), n => n.toString(16).padStart(2, "0"));
class util_Util {
  static makeHexColor(r, g, b) {
    return `#${util_hexNumbers[r]}${util_hexNumbers[g]}${util_hexNumbers[b]}`;
  }
  static scaleMinMax(transform, minMax) {
    let temp;
    if (transform[0]) {
      if (transform[0] < 0) {
        temp = minMax[0];
        minMax[0] = minMax[2];
        minMax[2] = temp;
      }
      minMax[0] *= transform[0];
      minMax[2] *= transform[0];
      if (transform[3] < 0) {
        temp = minMax[1];
        minMax[1] = minMax[3];
        minMax[3] = temp;
      }
      minMax[1] *= transform[3];
      minMax[3] *= transform[3];
    } else {
      temp = minMax[0];
      minMax[0] = minMax[1];
      minMax[1] = temp;
      temp = minMax[2];
      minMax[2] = minMax[3];
      minMax[3] = temp;
      if (transform[1] < 0) {
        temp = minMax[1];
        minMax[1] = minMax[3];
        minMax[3] = temp;
      }
      minMax[1] *= transform[1];
      minMax[3] *= transform[1];
      if (transform[2] < 0) {
        temp = minMax[0];
        minMax[0] = minMax[2];
        minMax[2] = temp;
      }
      minMax[0] *= transform[2];
      minMax[2] *= transform[2];
    }
    minMax[0] += transform[4];
    minMax[1] += transform[5];
    minMax[2] += transform[4];
    minMax[3] += transform[5];
  }
  static transform(m1, m2) {
    return [m1[0] * m2[0] + m1[2] * m2[1], m1[1] * m2[0] + m1[3] * m2[1], m1[0] * m2[2] + m1[2] * m2[3], m1[1] * m2[2] + m1[3] * m2[3], m1[0] * m2[4] + m1[2] * m2[5] + m1[4], m1[1] * m2[4] + m1[3] * m2[5] + m1[5]];
  }
  static applyTransform(p, m) {
    const xt = p[0] * m[0] + p[1] * m[2] + m[4];
    const yt = p[0] * m[1] + p[1] * m[3] + m[5];
    return [xt, yt];
  }
  static applyInverseTransform(p, m) {
    const d = m[0] * m[3] - m[1] * m[2];
    const xt = (p[0] * m[3] - p[1] * m[2] + m[2] * m[5] - m[4] * m[3]) / d;
    const yt = (-p[0] * m[1] + p[1] * m[0] + m[4] * m[1] - m[5] * m[0]) / d;
    return [xt, yt];
  }
  static getAxialAlignedBoundingBox(r, m) {
    const p1 = this.applyTransform(r, m);
    const p2 = this.applyTransform(r.slice(2, 4), m);
    const p3 = this.applyTransform([r[0], r[3]], m);
    const p4 = this.applyTransform([r[2], r[1]], m);
    return [Math.min(p1[0], p2[0], p3[0], p4[0]), Math.min(p1[1], p2[1], p3[1], p4[1]), Math.max(p1[0], p2[0], p3[0], p4[0]), Math.max(p1[1], p2[1], p3[1], p4[1])];
  }
  static inverseTransform(m) {
    const d = m[0] * m[3] - m[1] * m[2];
    return [m[3] / d, -m[1] / d, -m[2] / d, m[0] / d, (m[2] * m[5] - m[4] * m[3]) / d, (m[4] * m[1] - m[5] * m[0]) / d];
  }
  static singularValueDecompose2dScale(m) {
    const transpose = [m[0], m[2], m[1], m[3]];
    const a = m[0] * transpose[0] + m[1] * transpose[2];
    const b = m[0] * transpose[1] + m[1] * transpose[3];
    const c = m[2] * transpose[0] + m[3] * transpose[2];
    const d = m[2] * transpose[1] + m[3] * transpose[3];
    const first = (a + d) / 2;
    const second = Math.sqrt((a + d) ** 2 - 4 * (a * d - c * b)) / 2;
    const sx = first + second || 1;
    const sy = first - second || 1;
    return [Math.sqrt(sx), Math.sqrt(sy)];
  }
  static normalizeRect(rect) {
    const r = rect.slice(0);
    if (rect[0] > rect[2]) {
      r[0] = rect[2];
      r[2] = rect[0];
    }
    if (rect[1] > rect[3]) {
      r[1] = rect[3];
      r[3] = rect[1];
    }
    return r;
  }
  static intersect(rect1, rect2) {
    const xLow = Math.max(Math.min(rect1[0], rect1[2]), Math.min(rect2[0], rect2[2]));
    const xHigh = Math.min(Math.max(rect1[0], rect1[2]), Math.max(rect2[0], rect2[2]));
    if (xLow > xHigh) {
      return null;
    }
    const yLow = Math.max(Math.min(rect1[1], rect1[3]), Math.min(rect2[1], rect2[3]));
    const yHigh = Math.min(Math.max(rect1[1], rect1[3]), Math.max(rect2[1], rect2[3]));
    if (yLow > yHigh) {
      return null;
    }
    return [xLow, yLow, xHigh, yHigh];
  }
  static #getExtremumOnCurve(x0, x1, x2, x3, y0, y1, y2, y3, t, minMax) {
    if (t <= 0 || t >= 1) {
      return;
    }
    const mt = 1 - t;
    const tt = t * t;
    const ttt = tt * t;
    const x = mt * (mt * (mt * x0 + 3 * t * x1) + 3 * tt * x2) + ttt * x3;
    const y = mt * (mt * (mt * y0 + 3 * t * y1) + 3 * tt * y2) + ttt * y3;
    minMax[0] = Math.min(minMax[0], x);
    minMax[1] = Math.min(minMax[1], y);
    minMax[2] = Math.max(minMax[2], x);
    minMax[3] = Math.max(minMax[3], y);
  }
  static #getExtremum(x0, x1, x2, x3, y0, y1, y2, y3, a, b, c, minMax) {
    if (Math.abs(a) < 1e-12) {
      if (Math.abs(b) >= 1e-12) {
        this.#getExtremumOnCurve(x0, x1, x2, x3, y0, y1, y2, y3, -c / b, minMax);
      }
      return;
    }
    const delta = b ** 2 - 4 * c * a;
    if (delta < 0) {
      return;
    }
    const sqrtDelta = Math.sqrt(delta);
    const a2 = 2 * a;
    this.#getExtremumOnCurve(x0, x1, x2, x3, y0, y1, y2, y3, (-b + sqrtDelta) / a2, minMax);
    this.#getExtremumOnCurve(x0, x1, x2, x3, y0, y1, y2, y3, (-b - sqrtDelta) / a2, minMax);
  }
  static bezierBoundingBox(x0, y0, x1, y1, x2, y2, x3, y3, minMax) {
    if (minMax) {
      minMax[0] = Math.min(minMax[0], x0, x3);
      minMax[1] = Math.min(minMax[1], y0, y3);
      minMax[2] = Math.max(minMax[2], x0, x3);
      minMax[3] = Math.max(minMax[3], y0, y3);
    } else {
      minMax = [Math.min(x0, x3), Math.min(y0, y3), Math.max(x0, x3), Math.max(y0, y3)];
    }
    this.#getExtremum(x0, x1, x2, x3, y0, y1, y2, y3, 3 * (-x0 + 3 * (x1 - x2) + x3), 6 * (x0 - 2 * x1 + x2), 3 * (x1 - x0), minMax);
    this.#getExtremum(x0, x1, x2, x3, y0, y1, y2, y3, 3 * (-y0 + 3 * (y1 - y2) + y3), 6 * (y0 - 2 * y1 + y2), 3 * (y1 - y0), minMax);
    return minMax;
  }
}
const PDFStringTranslateTable = (/* unused pure expression or super */ null && ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x2d8, 0x2c7, 0x2c6, 0x2d9, 0x2dd, 0x2db, 0x2da, 0x2dc, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x2022, 0x2020, 0x2021, 0x2026, 0x2014, 0x2013, 0x192, 0x2044, 0x2039, 0x203a, 0x2212, 0x2030, 0x201e, 0x201c, 0x201d, 0x2018, 0x2019, 0x201a, 0x2122, 0xfb01, 0xfb02, 0x141, 0x152, 0x160, 0x178, 0x17d, 0x131, 0x142, 0x153, 0x161, 0x17e, 0, 0x20ac]));
function util_stringToPDFString(str) {
  if (str[0] >= "\xEF") {
    let encoding;
    if (str[0] === "\xFE" && str[1] === "\xFF") {
      encoding = "utf-16be";
      if (str.length % 2 === 1) {
        str = str.slice(0, -1);
      }
    } else if (str[0] === "\xFF" && str[1] === "\xFE") {
      encoding = "utf-16le";
      if (str.length % 2 === 1) {
        str = str.slice(0, -1);
      }
    } else if (str[0] === "\xEF" && str[1] === "\xBB" && str[2] === "\xBF") {
      encoding = "utf-8";
    }
    if (encoding) {
      try {
        const decoder = new TextDecoder(encoding, {
          fatal: true
        });
        const buffer = stringToBytes(str);
        const decoded = decoder.decode(buffer);
        if (!decoded.includes("\x1b")) {
          return decoded;
        }
        return decoded.replaceAll(/\x1b[^\x1b]*(?:\x1b|$)/g, "");
      } catch (ex) {
        util_warn(`stringToPDFString: "${ex}".`);
      }
    }
  }
  const strBuf = [];
  for (let i = 0, ii = str.length; i < ii; i++) {
    const charCode = str.charCodeAt(i);
    if (charCode === 0x1b) {
      while (++i < ii && str.charCodeAt(i) !== 0x1b) {}
      continue;
    }
    const code = PDFStringTranslateTable[charCode];
    strBuf.push(code ? String.fromCharCode(code) : str.charAt(i));
  }
  return strBuf.join("");
}
function stringToUTF8String(str) {
  return decodeURIComponent(escape(str));
}
function utf8StringToString(str) {
  return unescape(encodeURIComponent(str));
}
function isArrayEqual(arr1, arr2) {
  if (arr1.length !== arr2.length) {
    return false;
  }
  for (let i = 0, ii = arr1.length; i < ii; i++) {
    if (arr1[i] !== arr2[i]) {
      return false;
    }
  }
  return true;
}
function getModificationDate(date = new Date()) {
  const buffer = [date.getUTCFullYear().toString(), (date.getUTCMonth() + 1).toString().padStart(2, "0"), date.getUTCDate().toString().padStart(2, "0"), date.getUTCHours().toString().padStart(2, "0"), date.getUTCMinutes().toString().padStart(2, "0"), date.getUTCSeconds().toString().padStart(2, "0")];
  return buffer.join("");
}
let NormalizeRegex = null;
let NormalizationMap = null;
function normalizeUnicode(str) {
  if (!NormalizeRegex) {
    NormalizeRegex = /([\u00a0\u00b5\u037e\u0eb3\u2000-\u200a\u202f\u2126\ufb00-\ufb04\ufb06\ufb20-\ufb36\ufb38-\ufb3c\ufb3e\ufb40-\ufb41\ufb43-\ufb44\ufb46-\ufba1\ufba4-\ufba9\ufbae-\ufbb1\ufbd3-\ufbdc\ufbde-\ufbe7\ufbea-\ufbf8\ufbfc-\ufbfd\ufc00-\ufc5d\ufc64-\ufcf1\ufcf5-\ufd3d\ufd88\ufdf4\ufdfa-\ufdfb\ufe71\ufe77\ufe79\ufe7b\ufe7d]+)|(\ufb05+)/gu;
    NormalizationMap = new Map([["ﬅ", "ſt"]]);
  }
  return str.replaceAll(NormalizeRegex, (_, p1, p2) => p1 ? p1.normalize("NFKC") : NormalizationMap.get(p2));
}
function getUuid() {
  if (typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  const buf = new Uint8Array(32);
  crypto.getRandomValues(buf);
  return bytesToString(buf);
}
const AnnotationPrefix = "pdfjs_internal_id_";
function toHexUtil(arr) {
  if (Uint8Array.prototype.toHex) {
    return arr.toHex();
  }
  return Array.from(arr, num => util_hexNumbers[num]).join("");
}
function toBase64Util(arr) {
  if (Uint8Array.prototype.toBase64) {
    return arr.toBase64();
  }
  return btoa(bytesToString(arr));
}
function fromBase64Util(str) {
  if (Uint8Array.fromBase64) {
    return Uint8Array.fromBase64(str);
  }
  return stringToBytes(atob(str));
}

// EXTERNAL MODULE: ./node_modules/core-js/modules/es.set.difference.v2.js
var es_set_difference_v2 = __webpack_require__(7642);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.set.intersection.v2.js
var es_set_intersection_v2 = __webpack_require__(8004);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.set.is-disjoint-from.v2.js
var es_set_is_disjoint_from_v2 = __webpack_require__(3853);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.set.is-subset-of.v2.js
var es_set_is_subset_of_v2 = __webpack_require__(5876);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.set.is-superset-of.v2.js
var es_set_is_superset_of_v2 = __webpack_require__(2475);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.set.symmetric-difference.v2.js
var es_set_symmetric_difference_v2 = __webpack_require__(5024);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.set.union.v2.js
var es_set_union_v2 = __webpack_require__(1698);
// EXTERNAL MODULE: ./node_modules/core-js/modules/esnext.iterator.constructor.js
var esnext_iterator_constructor = __webpack_require__(8992);
// EXTERNAL MODULE: ./node_modules/core-js/modules/esnext.iterator.every.js
var esnext_iterator_every = __webpack_require__(3215);
// EXTERNAL MODULE: ./node_modules/core-js/modules/esnext.iterator.map.js
var esnext_iterator_map = __webpack_require__(1454);
;// ./src/core/primitives.js









const CIRCULAR_REF = Symbol("CIRCULAR_REF");
const EOF = Symbol("EOF");
let CmdCache = Object.create(null);
let NameCache = Object.create(null);
let RefCache = Object.create(null);
function clearPrimitiveCaches() {
  CmdCache = Object.create(null);
  NameCache = Object.create(null);
  RefCache = Object.create(null);
}
class Name {
  constructor(name) {
    this.name = name;
  }
  static get(name) {
    return NameCache[name] ||= new Name(name);
  }
}
class Cmd {
  constructor(cmd) {
    this.cmd = cmd;
  }
  static get(cmd) {
    return CmdCache[cmd] ||= new Cmd(cmd);
  }
}
const nonSerializable = function nonSerializableClosure() {
  return nonSerializable;
};
class primitives_Dict {
  constructor(xref = null) {
    this._map = new Map();
    this.xref = xref;
    this.objId = null;
    this.suppressEncryption = false;
    this.__nonSerializable__ = nonSerializable;
  }
  assignXref(newXref) {
    this.xref = newXref;
  }
  get size() {
    return this._map.size;
  }
  get(key1, key2, key3) {
    let value = this._map.get(key1);
    if (value === undefined && key2 !== undefined) {
      value = this._map.get(key2);
      if (value === undefined && key3 !== undefined) {
        value = this._map.get(key3);
      }
    }
    if (value instanceof primitives_Ref && this.xref) {
      return this.xref.fetch(value, this.suppressEncryption);
    }
    return value;
  }
  async getAsync(key1, key2, key3) {
    let value = this._map.get(key1);
    if (value === undefined && key2 !== undefined) {
      value = this._map.get(key2);
      if (value === undefined && key3 !== undefined) {
        value = this._map.get(key3);
      }
    }
    if (value instanceof primitives_Ref && this.xref) {
      return this.xref.fetchAsync(value, this.suppressEncryption);
    }
    return value;
  }
  getArray(key1, key2, key3) {
    let value = this._map.get(key1);
    if (value === undefined && key2 !== undefined) {
      value = this._map.get(key2);
      if (value === undefined && key3 !== undefined) {
        value = this._map.get(key3);
      }
    }
    if (value instanceof primitives_Ref && this.xref) {
      value = this.xref.fetch(value, this.suppressEncryption);
    }
    if (Array.isArray(value)) {
      value = value.slice();
      for (let i = 0, ii = value.length; i < ii; i++) {
        if (value[i] instanceof primitives_Ref && this.xref) {
          value[i] = this.xref.fetch(value[i], this.suppressEncryption);
        }
      }
    }
    return value;
  }
  getRaw(key) {
    return this._map.get(key);
  }
  getKeys() {
    return [...this._map.keys()];
  }
  getRawValues() {
    return [...this._map.values()];
  }
  set(key, value) {
    this._map.set(key, value);
  }
  has(key) {
    return this._map.has(key);
  }
  *[Symbol.iterator]() {
    for (const [key, value] of this._map) {
      yield [key, value instanceof primitives_Ref && this.xref ? this.xref.fetch(value, this.suppressEncryption) : value];
    }
  }
  static get empty() {
    const emptyDict = new primitives_Dict(null);
    emptyDict.set = (key, value) => {
      unreachable("Should not call `set` on the empty dictionary.");
    };
    return shadow(this, "empty", emptyDict);
  }
  static merge({
    xref,
    dictArray,
    mergeSubDicts = false
  }) {
    const mergedDict = new primitives_Dict(xref),
      properties = new Map();
    for (const dict of dictArray) {
      if (!(dict instanceof primitives_Dict)) {
        continue;
      }
      for (const [key, value] of dict._map) {
        let property = properties.get(key);
        if (property === undefined) {
          property = [];
          properties.set(key, property);
        } else if (!mergeSubDicts || !(value instanceof primitives_Dict)) {
          continue;
        }
        property.push(value);
      }
    }
    for (const [name, values] of properties) {
      if (values.length === 1 || !(values[0] instanceof primitives_Dict)) {
        mergedDict._map.set(name, values[0]);
        continue;
      }
      const subDict = new primitives_Dict(xref);
      for (const dict of values) {
        for (const [key, value] of dict._map) {
          if (!subDict._map.has(key)) {
            subDict._map.set(key, value);
          }
        }
      }
      if (subDict.size > 0) {
        mergedDict._map.set(name, subDict);
      }
    }
    properties.clear();
    return mergedDict.size > 0 ? mergedDict : primitives_Dict.empty;
  }
  clone() {
    const dict = new primitives_Dict(this.xref);
    for (const key of this.getKeys()) {
      dict.set(key, this.getRaw(key));
    }
    return dict;
  }
  delete(key) {
    delete this._map[key];
  }
}
class primitives_Ref {
  constructor(num, gen) {
    this.num = num;
    this.gen = gen;
  }
  toString() {
    if (this.gen === 0) {
      return `${this.num}R`;
    }
    return `${this.num}R${this.gen}`;
  }
  static fromString(str) {
    const ref = RefCache[str];
    if (ref) {
      return ref;
    }
    const m = /^(\d+)R(\d*)$/.exec(str);
    if (!m || m[1] === "0") {
      return null;
    }
    return RefCache[str] = new primitives_Ref(parseInt(m[1]), !m[2] ? 0 : parseInt(m[2]));
  }
  static get(num, gen) {
    const key = gen === 0 ? `${num}R` : `${num}R${gen}`;
    return RefCache[key] ||= new primitives_Ref(num, gen);
  }
}
class primitives_RefSet {
  constructor(parent = null) {
    this._set = new Set(parent?._set);
  }
  has(ref) {
    return this._set.has(ref.toString());
  }
  put(ref) {
    this._set.add(ref.toString());
  }
  remove(ref) {
    this._set.delete(ref.toString());
  }
  [Symbol.iterator]() {
    return this._set.values();
  }
  clear() {
    this._set.clear();
  }
}
class RefSetCache {
  constructor() {
    this._map = new Map();
  }
  get size() {
    return this._map.size;
  }
  get(ref) {
    return this._map.get(ref.toString());
  }
  has(ref) {
    return this._map.has(ref.toString());
  }
  put(ref, obj) {
    this._map.set(ref.toString(), obj);
  }
  putAlias(ref, aliasRef) {
    this._map.set(ref.toString(), this.get(aliasRef));
  }
  [Symbol.iterator]() {
    return this._map.values();
  }
  clear() {
    this._map.clear();
  }
  *values() {
    yield* this._map.values();
  }
  *items() {
    for (const [ref, value] of this._map) {
      yield [primitives_Ref.fromString(ref), value];
    }
  }
}
function primitives_isName(v, name) {
  return v instanceof Name && (name === undefined || v.name === name);
}
function isCmd(v, cmd) {
  return v instanceof Cmd && (cmd === undefined || v.cmd === cmd);
}
function isDict(v, type) {
  return v instanceof primitives_Dict && (type === undefined || primitives_isName(v.get("Type"), type));
}
function isRefsEqual(v1, v2) {
  return v1.num === v2.num && v1.gen === v2.gen;
}

;// ./src/core/base_stream.js

class base_stream_BaseStream {
  get length() {
    unreachable("Abstract getter `length` accessed");
  }
  get isEmpty() {
    unreachable("Abstract getter `isEmpty` accessed");
  }
  get isDataLoaded() {
    return shadow(this, "isDataLoaded", true);
  }
  getByte() {
    unreachable("Abstract method `getByte` called");
  }
  getBytes(length) {
    unreachable("Abstract method `getBytes` called");
  }
  async getImageData(length, decoderOptions) {
    return this.getBytes(length, decoderOptions);
  }
  async asyncGetBytes() {
    unreachable("Abstract method `asyncGetBytes` called");
  }
  get isAsync() {
    return false;
  }
  get canAsyncDecodeImageFromBuffer() {
    return false;
  }
  async getTransferableImage() {
    return null;
  }
  peekByte() {
    const peekedByte = this.getByte();
    if (peekedByte !== -1) {
      this.pos--;
    }
    return peekedByte;
  }
  peekBytes(length) {
    const bytes = this.getBytes(length);
    this.pos -= bytes.length;
    return bytes;
  }
  getUint16() {
    const b0 = this.getByte();
    const b1 = this.getByte();
    if (b0 === -1 || b1 === -1) {
      return -1;
    }
    return (b0 << 8) + b1;
  }
  getInt32() {
    const b0 = this.getByte();
    const b1 = this.getByte();
    const b2 = this.getByte();
    const b3 = this.getByte();
    return (b0 << 24) + (b1 << 16) + (b2 << 8) + b3;
  }
  getByteRange(begin, end) {
    unreachable("Abstract method `getByteRange` called");
  }
  getString(length) {
    return bytesToString(this.getBytes(length));
  }
  skip(n) {
    this.pos += n || 1;
  }
  reset() {
    unreachable("Abstract method `reset` called");
  }
  moveStart() {
    unreachable("Abstract method `moveStart` called");
  }
  makeSubStream(start, length, dict = null) {
    unreachable("Abstract method `makeSubStream` called");
  }
  getBaseStreams() {
    return null;
  }
}

;// ./src/core/core_utils.js




















const PDF_VERSION_REGEXP = /^[1-9]\.\d$/;
const MAX_INT_32 = 2 ** 31 - 1;
const MIN_INT_32 = -(2 ** 31);
function getLookupTableFactory(initializer) {
  let lookup;
  return function () {
    if (initializer) {
      lookup = Object.create(null);
      initializer(lookup);
      initializer = null;
    }
    return lookup;
  };
}
class MissingDataException extends BaseException {
  constructor(begin, end) {
    super(`Missing data [${begin}, ${end})`, "MissingDataException");
    this.begin = begin;
    this.end = end;
  }
}
class ParserEOFException extends BaseException {
  constructor(msg) {
    super(msg, "ParserEOFException");
  }
}
class XRefEntryException extends BaseException {
  constructor(msg) {
    super(msg, "XRefEntryException");
  }
}
class XRefParseException extends BaseException {
  constructor(msg) {
    super(msg, "XRefParseException");
  }
}
function arrayBuffersToBytes(arr) {
  const length = arr.length;
  if (length === 0) {
    return new Uint8Array(0);
  }
  if (length === 1) {
    return new Uint8Array(arr[0]);
  }
  let dataLength = 0;
  for (let i = 0; i < length; i++) {
    dataLength += arr[i].byteLength;
  }
  const data = new Uint8Array(dataLength);
  let pos = 0;
  for (let i = 0; i < length; i++) {
    const item = new Uint8Array(arr[i]);
    data.set(item, pos);
    pos += item.byteLength;
  }
  return data;
}
function getInheritableProperty({
  dict,
  key,
  getArray = false,
  stopWhenFound = true
}) {
  let values;
  const visited = new RefSet();
  while (dict instanceof Dict && !(dict.objId && visited.has(dict.objId))) {
    if (dict.objId) {
      visited.put(dict.objId);
    }
    const value = getArray ? dict.getArray(key) : dict.get(key);
    if (value !== undefined) {
      if (stopWhenFound) {
        return value;
      }
      (values ||= []).push(value);
    }
    dict = dict.get("Parent");
  }
  return values;
}
function getParentToUpdate(dict, ref, xref) {
  const visited = new RefSet();
  const firstDict = dict;
  const result = {
    dict: null,
    ref: null
  };
  while (dict instanceof Dict && !visited.has(ref)) {
    visited.put(ref);
    if (dict.has("T")) {
      break;
    }
    ref = dict.getRaw("Parent");
    if (!(ref instanceof Ref)) {
      return result;
    }
    dict = xref.fetch(ref);
  }
  if (dict instanceof Dict && dict !== firstDict) {
    result.dict = dict;
    result.ref = ref;
  }
  return result;
}
const ROMAN_NUMBER_MAP = (/* unused pure expression or super */ null && (["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM", "", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC", "", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]));
function toRomanNumerals(number, lowerCase = false) {
  assert(Number.isInteger(number) && number > 0, "The number should be a positive integer.");
  const roman = "M".repeat(number / 1000 | 0) + ROMAN_NUMBER_MAP[number % 1000 / 100 | 0] + ROMAN_NUMBER_MAP[10 + (number % 100 / 10 | 0)] + ROMAN_NUMBER_MAP[20 + number % 10];
  return lowerCase ? roman.toLowerCase() : roman;
}
function log2(x) {
  return x > 0 ? Math.ceil(Math.log2(x)) : 0;
}
function readInt8(data, offset) {
  return data[offset] << 24 >> 24;
}
function readUint16(data, offset) {
  return data[offset] << 8 | data[offset + 1];
}
function readUint32(data, offset) {
  return (data[offset] << 24 | data[offset + 1] << 16 | data[offset + 2] << 8 | data[offset + 3]) >>> 0;
}
function isWhiteSpace(ch) {
  return ch === 0x20 || ch === 0x09 || ch === 0x0d || ch === 0x0a;
}
function isBooleanArray(arr, len) {
  return Array.isArray(arr) && (len === null || arr.length === len) && arr.every(x => typeof x === "boolean");
}
function isNumberArray(arr, len) {
  if (Array.isArray(arr)) {
    return (len === null || arr.length === len) && arr.every(x => typeof x === "number");
  }
  return ArrayBuffer.isView(arr) && (arr.length === 0 || typeof arr[0] === "number") && (len === null || arr.length === len);
}
function lookupMatrix(arr, fallback) {
  return isNumberArray(arr, 6) ? arr : fallback;
}
function lookupRect(arr, fallback) {
  return isNumberArray(arr, 4) ? arr : fallback;
}
function lookupNormalRect(arr, fallback) {
  return isNumberArray(arr, 4) ? Util.normalizeRect(arr) : fallback;
}
function parseXFAPath(path) {
  const positionPattern = /(.+)\[(\d+)\]$/;
  return path.split(".").map(component => {
    const m = component.match(positionPattern);
    if (m) {
      return {
        name: m[1],
        pos: parseInt(m[2], 10)
      };
    }
    return {
      name: component,
      pos: 0
    };
  });
}
function escapePDFName(str) {
  const buffer = [];
  let start = 0;
  for (let i = 0, ii = str.length; i < ii; i++) {
    const char = str.charCodeAt(i);
    if (char < 0x21 || char > 0x7e || char === 0x23 || char === 0x28 || char === 0x29 || char === 0x3c || char === 0x3e || char === 0x5b || char === 0x5d || char === 0x7b || char === 0x7d || char === 0x2f || char === 0x25) {
      if (start < i) {
        buffer.push(str.substring(start, i));
      }
      buffer.push(`#${char.toString(16)}`);
      start = i + 1;
    }
  }
  if (buffer.length === 0) {
    return str;
  }
  if (start < str.length) {
    buffer.push(str.substring(start, str.length));
  }
  return buffer.join("");
}
function escapeString(str) {
  return str.replaceAll(/([()\\\n\r])/g, match => {
    if (match === "\n") {
      return "\\n";
    } else if (match === "\r") {
      return "\\r";
    }
    return `\\${match}`;
  });
}
function _collectJS(entry, xref, list, parents) {
  if (!entry) {
    return;
  }
  let parent = null;
  if (entry instanceof Ref) {
    if (parents.has(entry)) {
      return;
    }
    parent = entry;
    parents.put(parent);
    entry = xref.fetch(entry);
  }
  if (Array.isArray(entry)) {
    for (const element of entry) {
      _collectJS(element, xref, list, parents);
    }
  } else if (entry instanceof Dict) {
    if (isName(entry.get("S"), "JavaScript")) {
      const js = entry.get("JS");
      let code;
      if (js instanceof BaseStream) {
        code = js.getString();
      } else if (typeof js === "string") {
        code = js;
      }
      code &&= stringToPDFString(code).replaceAll("\x00", "");
      if (code) {
        list.push(code);
      }
    }
    _collectJS(entry.getRaw("Next"), xref, list, parents);
  }
  if (parent) {
    parents.remove(parent);
  }
}
function collectActions(xref, dict, eventType) {
  const actions = Object.create(null);
  const additionalActionsDicts = getInheritableProperty({
    dict,
    key: "AA",
    stopWhenFound: false
  });
  if (additionalActionsDicts) {
    for (let i = additionalActionsDicts.length - 1; i >= 0; i--) {
      const additionalActions = additionalActionsDicts[i];
      if (!(additionalActions instanceof Dict)) {
        continue;
      }
      for (const key of additionalActions.getKeys()) {
        const action = eventType[key];
        if (!action) {
          continue;
        }
        const actionDict = additionalActions.getRaw(key);
        const parents = new RefSet();
        const list = [];
        _collectJS(actionDict, xref, list, parents);
        if (list.length > 0) {
          actions[action] = list;
        }
      }
    }
  }
  if (dict.has("A")) {
    const actionDict = dict.get("A");
    const parents = new RefSet();
    const list = [];
    _collectJS(actionDict, xref, list, parents);
    if (list.length > 0) {
      actions.Action = list;
    }
  }
  return objectSize(actions) > 0 ? actions : null;
}
const XMLEntities = {
  0x3c: "&lt;",
  0x3e: "&gt;",
  0x26: "&amp;",
  0x22: "&quot;",
  0x27: "&apos;"
};
function* codePointIter(str) {
  for (let i = 0, ii = str.length; i < ii; i++) {
    const char = str.codePointAt(i);
    if (char > 0xd7ff && (char < 0xe000 || char > 0xfffd)) {
      i++;
    }
    yield char;
  }
}
function encodeToXmlString(str) {
  const buffer = [];
  let start = 0;
  for (let i = 0, ii = str.length; i < ii; i++) {
    const char = str.codePointAt(i);
    if (0x20 <= char && char <= 0x7e) {
      const entity = XMLEntities[char];
      if (entity) {
        if (start < i) {
          buffer.push(str.substring(start, i));
        }
        buffer.push(entity);
        start = i + 1;
      }
    } else {
      if (start < i) {
        buffer.push(str.substring(start, i));
      }
      buffer.push(`&#x${char.toString(16).toUpperCase()};`);
      if (char > 0xd7ff && (char < 0xe000 || char > 0xfffd)) {
        i++;
      }
      start = i + 1;
    }
  }
  if (buffer.length === 0) {
    return str;
  }
  if (start < str.length) {
    buffer.push(str.substring(start, str.length));
  }
  return buffer.join("");
}
function validateFontName(fontFamily, mustWarn = false) {
  const m = /^("|').*("|')$/.exec(fontFamily);
  if (m && m[1] === m[2]) {
    const re = new RegExp(`[^\\\\]${m[1]}`);
    if (re.test(fontFamily.slice(1, -1))) {
      if (mustWarn) {
        warn(`FontFamily contains unescaped ${m[1]}: ${fontFamily}.`);
      }
      return false;
    }
  } else {
    for (const ident of fontFamily.split(/[ \t]+/)) {
      if (/^(\d|(-(\d|-)))/.test(ident) || !/^[\w-\\]+$/.test(ident)) {
        if (mustWarn) {
          warn(`FontFamily contains invalid <custom-ident>: ${fontFamily}.`);
        }
        return false;
      }
    }
  }
  return true;
}
function validateCSSFont(cssFontInfo) {
  const DEFAULT_CSS_FONT_OBLIQUE = "14";
  const DEFAULT_CSS_FONT_WEIGHT = "400";
  const CSS_FONT_WEIGHT_VALUES = new Set(["100", "200", "300", "400", "500", "600", "700", "800", "900", "1000", "normal", "bold", "bolder", "lighter"]);
  const {
    fontFamily,
    fontWeight,
    italicAngle
  } = cssFontInfo;
  if (!validateFontName(fontFamily, true)) {
    return false;
  }
  const weight = fontWeight ? fontWeight.toString() : "";
  cssFontInfo.fontWeight = CSS_FONT_WEIGHT_VALUES.has(weight) ? weight : DEFAULT_CSS_FONT_WEIGHT;
  const angle = parseFloat(italicAngle);
  cssFontInfo.italicAngle = isNaN(angle) || angle < -90 || angle > 90 ? DEFAULT_CSS_FONT_OBLIQUE : italicAngle.toString();
  return true;
}
function recoverJsURL(str) {
  const URL_OPEN_METHODS = ["app.launchURL", "window.open", "xfa.host.gotoURL"];
  const regex = new RegExp("^\\s*(" + URL_OPEN_METHODS.join("|").replaceAll(".", "\\.") + ")\\((?:'|\")([^'\"]*)(?:'|\")(?:,\\s*(\\w+)\\)|\\))", "i");
  const jsUrl = regex.exec(str);
  if (jsUrl?.[2]) {
    return {
      url: jsUrl[2],
      newWindow: jsUrl[1] === "app.launchURL" && jsUrl[3] === "true"
    };
  }
  return null;
}
function numberToString(value) {
  if (Number.isInteger(value)) {
    return value.toString();
  }
  const roundedValue = Math.round(value * 100);
  if (roundedValue % 100 === 0) {
    return (roundedValue / 100).toString();
  }
  if (roundedValue % 10 === 0) {
    return value.toFixed(1);
  }
  return value.toFixed(2);
}
function getNewAnnotationsMap(annotationStorage) {
  if (!annotationStorage) {
    return null;
  }
  const newAnnotationsByPage = new Map();
  for (const [key, value] of annotationStorage) {
    if (!key.startsWith(AnnotationEditorPrefix)) {
      continue;
    }
    let annotations = newAnnotationsByPage.get(value.pageIndex);
    if (!annotations) {
      annotations = [];
      newAnnotationsByPage.set(value.pageIndex, annotations);
    }
    annotations.push(value);
  }
  return newAnnotationsByPage.size > 0 ? newAnnotationsByPage : null;
}
function stringToAsciiOrUTF16BE(str) {
  return isAscii(str) ? str : stringToUTF16String(str, true);
}
function isAscii(str) {
  return /^[\x00-\x7F]*$/.test(str);
}
function stringToUTF16HexString(str) {
  const buf = [];
  for (let i = 0, ii = str.length; i < ii; i++) {
    const char = str.charCodeAt(i);
    buf.push(hexNumbers[char >> 8 & 0xff], hexNumbers[char & 0xff]);
  }
  return buf.join("");
}
function stringToUTF16String(str, bigEndian = false) {
  const buf = [];
  if (bigEndian) {
    buf.push("\xFE\xFF");
  }
  for (let i = 0, ii = str.length; i < ii; i++) {
    const char = str.charCodeAt(i);
    buf.push(String.fromCharCode(char >> 8 & 0xff), String.fromCharCode(char & 0xff));
  }
  return buf.join("");
}
function getRotationMatrix(rotation, width, height) {
  switch (rotation) {
    case 90:
      return [0, 1, -1, 0, width, 0];
    case 180:
      return [-1, 0, 0, -1, width, height];
    case 270:
      return [0, -1, 1, 0, 0, height];
    default:
      throw new Error("Invalid rotation");
  }
}
function getSizeInBytes(x) {
  return Math.ceil(Math.ceil(Math.log2(1 + x)) / 8);
}

;// ./src/core/arithmetic_decoder.js
const QeTable = [{
  qe: 0x5601,
  nmps: 1,
  nlps: 1,
  switchFlag: 1
}, {
  qe: 0x3401,
  nmps: 2,
  nlps: 6,
  switchFlag: 0
}, {
  qe: 0x1801,
  nmps: 3,
  nlps: 9,
  switchFlag: 0
}, {
  qe: 0x0ac1,
  nmps: 4,
  nlps: 12,
  switchFlag: 0
}, {
  qe: 0x0521,
  nmps: 5,
  nlps: 29,
  switchFlag: 0
}, {
  qe: 0x0221,
  nmps: 38,
  nlps: 33,
  switchFlag: 0
}, {
  qe: 0x5601,
  nmps: 7,
  nlps: 6,
  switchFlag: 1
}, {
  qe: 0x5401,
  nmps: 8,
  nlps: 14,
  switchFlag: 0
}, {
  qe: 0x4801,
  nmps: 9,
  nlps: 14,
  switchFlag: 0
}, {
  qe: 0x3801,
  nmps: 10,
  nlps: 14,
  switchFlag: 0
}, {
  qe: 0x3001,
  nmps: 11,
  nlps: 17,
  switchFlag: 0
}, {
  qe: 0x2401,
  nmps: 12,
  nlps: 18,
  switchFlag: 0
}, {
  qe: 0x1c01,
  nmps: 13,
  nlps: 20,
  switchFlag: 0
}, {
  qe: 0x1601,
  nmps: 29,
  nlps: 21,
  switchFlag: 0
}, {
  qe: 0x5601,
  nmps: 15,
  nlps: 14,
  switchFlag: 1
}, {
  qe: 0x5401,
  nmps: 16,
  nlps: 14,
  switchFlag: 0
}, {
  qe: 0x5101,
  nmps: 17,
  nlps: 15,
  switchFlag: 0
}, {
  qe: 0x4801,
  nmps: 18,
  nlps: 16,
  switchFlag: 0
}, {
  qe: 0x3801,
  nmps: 19,
  nlps: 17,
  switchFlag: 0
}, {
  qe: 0x3401,
  nmps: 20,
  nlps: 18,
  switchFlag: 0
}, {
  qe: 0x3001,
  nmps: 21,
  nlps: 19,
  switchFlag: 0
}, {
  qe: 0x2801,
  nmps: 22,
  nlps: 19,
  switchFlag: 0
}, {
  qe: 0x2401,
  nmps: 23,
  nlps: 20,
  switchFlag: 0
}, {
  qe: 0x2201,
  nmps: 24,
  nlps: 21,
  switchFlag: 0
}, {
  qe: 0x1c01,
  nmps: 25,
  nlps: 22,
  switchFlag: 0
}, {
  qe: 0x1801,
  nmps: 26,
  nlps: 23,
  switchFlag: 0
}, {
  qe: 0x1601,
  nmps: 27,
  nlps: 24,
  switchFlag: 0
}, {
  qe: 0x1401,
  nmps: 28,
  nlps: 25,
  switchFlag: 0
}, {
  qe: 0x1201,
  nmps: 29,
  nlps: 26,
  switchFlag: 0
}, {
  qe: 0x1101,
  nmps: 30,
  nlps: 27,
  switchFlag: 0
}, {
  qe: 0x0ac1,
  nmps: 31,
  nlps: 28,
  switchFlag: 0
}, {
  qe: 0x09c1,
  nmps: 32,
  nlps: 29,
  switchFlag: 0
}, {
  qe: 0x08a1,
  nmps: 33,
  nlps: 30,
  switchFlag: 0
}, {
  qe: 0x0521,
  nmps: 34,
  nlps: 31,
  switchFlag: 0
}, {
  qe: 0x0441,
  nmps: 35,
  nlps: 32,
  switchFlag: 0
}, {
  qe: 0x02a1,
  nmps: 36,
  nlps: 33,
  switchFlag: 0
}, {
  qe: 0x0221,
  nmps: 37,
  nlps: 34,
  switchFlag: 0
}, {
  qe: 0x0141,
  nmps: 38,
  nlps: 35,
  switchFlag: 0
}, {
  qe: 0x0111,
  nmps: 39,
  nlps: 36,
  switchFlag: 0
}, {
  qe: 0x0085,
  nmps: 40,
  nlps: 37,
  switchFlag: 0
}, {
  qe: 0x0049,
  nmps: 41,
  nlps: 38,
  switchFlag: 0
}, {
  qe: 0x0025,
  nmps: 42,
  nlps: 39,
  switchFlag: 0
}, {
  qe: 0x0015,
  nmps: 43,
  nlps: 40,
  switchFlag: 0
}, {
  qe: 0x0009,
  nmps: 44,
  nlps: 41,
  switchFlag: 0
}, {
  qe: 0x0005,
  nmps: 45,
  nlps: 42,
  switchFlag: 0
}, {
  qe: 0x0001,
  nmps: 45,
  nlps: 43,
  switchFlag: 0
}, {
  qe: 0x5601,
  nmps: 46,
  nlps: 46,
  switchFlag: 0
}];
class ArithmeticDecoder {
  constructor(data, start, end) {
    this.data = data;
    this.bp = start;
    this.dataEnd = end;
    this.chigh = data[start];
    this.clow = 0;
    this.byteIn();
    this.chigh = this.chigh << 7 & 0xffff | this.clow >> 9 & 0x7f;
    this.clow = this.clow << 7 & 0xffff;
    this.ct -= 7;
    this.a = 0x8000;
  }
  byteIn() {
    const data = this.data;
    let bp = this.bp;
    if (data[bp] === 0xff) {
      if (data[bp + 1] > 0x8f) {
        this.clow += 0xff00;
        this.ct = 8;
      } else {
        bp++;
        this.clow += data[bp] << 9;
        this.ct = 7;
        this.bp = bp;
      }
    } else {
      bp++;
      this.clow += bp < this.dataEnd ? data[bp] << 8 : 0xff00;
      this.ct = 8;
      this.bp = bp;
    }
    if (this.clow > 0xffff) {
      this.chigh += this.clow >> 16;
      this.clow &= 0xffff;
    }
  }
  readBit(contexts, pos) {
    let cx_index = contexts[pos] >> 1,
      cx_mps = contexts[pos] & 1;
    const qeTableIcx = QeTable[cx_index];
    const qeIcx = qeTableIcx.qe;
    let d;
    let a = this.a - qeIcx;
    if (this.chigh < qeIcx) {
      if (a < qeIcx) {
        a = qeIcx;
        d = cx_mps;
        cx_index = qeTableIcx.nmps;
      } else {
        a = qeIcx;
        d = 1 ^ cx_mps;
        if (qeTableIcx.switchFlag === 1) {
          cx_mps = d;
        }
        cx_index = qeTableIcx.nlps;
      }
    } else {
      this.chigh -= qeIcx;
      if ((a & 0x8000) !== 0) {
        this.a = a;
        return cx_mps;
      }
      if (a < qeIcx) {
        d = 1 ^ cx_mps;
        if (qeTableIcx.switchFlag === 1) {
          cx_mps = d;
        }
        cx_index = qeTableIcx.nlps;
      } else {
        d = cx_mps;
        cx_index = qeTableIcx.nmps;
      }
    }
    do {
      if (this.ct === 0) {
        this.byteIn();
      }
      a <<= 1;
      this.chigh = this.chigh << 1 & 0xffff | this.clow >> 15 & 1;
      this.clow = this.clow << 1 & 0xffff;
      this.ct--;
    } while ((a & 0x8000) === 0);
    this.a = a;
    contexts[pos] = cx_index << 1 | cx_mps;
    return d;
  }
}

;// ./src/core/ccitt.js







const ccittEOL = -2;
const ccittEOF = -1;
const twoDimPass = 0;
const twoDimHoriz = 1;
const twoDimVert0 = 2;
const twoDimVertR1 = 3;
const twoDimVertL1 = 4;
const twoDimVertR2 = 5;
const twoDimVertL2 = 6;
const twoDimVertR3 = 7;
const twoDimVertL3 = 8;
const twoDimTable = [[-1, -1], [-1, -1], [7, twoDimVertL3], [7, twoDimVertR3], [6, twoDimVertL2], [6, twoDimVertL2], [6, twoDimVertR2], [6, twoDimVertR2], [4, twoDimPass], [4, twoDimPass], [4, twoDimPass], [4, twoDimPass], [4, twoDimPass], [4, twoDimPass], [4, twoDimPass], [4, twoDimPass], [3, twoDimHoriz], [3, twoDimHoriz], [3, twoDimHoriz], [3, twoDimHoriz], [3, twoDimHoriz], [3, twoDimHoriz], [3, twoDimHoriz], [3, twoDimHoriz], [3, twoDimHoriz], [3, twoDimHoriz], [3, twoDimHoriz], [3, twoDimHoriz], [3, twoDimHoriz], [3, twoDimHoriz], [3, twoDimHoriz], [3, twoDimHoriz], [3, twoDimVertL1], [3, twoDimVertL1], [3, twoDimVertL1], [3, twoDimVertL1], [3, twoDimVertL1], [3, twoDimVertL1], [3, twoDimVertL1], [3, twoDimVertL1], [3, twoDimVertL1], [3, twoDimVertL1], [3, twoDimVertL1], [3, twoDimVertL1], [3, twoDimVertL1], [3, twoDimVertL1], [3, twoDimVertL1], [3, twoDimVertL1], [3, twoDimVertR1], [3, twoDimVertR1], [3, twoDimVertR1], [3, twoDimVertR1], [3, twoDimVertR1], [3, twoDimVertR1], [3, twoDimVertR1], [3, twoDimVertR1], [3, twoDimVertR1], [3, twoDimVertR1], [3, twoDimVertR1], [3, twoDimVertR1], [3, twoDimVertR1], [3, twoDimVertR1], [3, twoDimVertR1], [3, twoDimVertR1], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0], [1, twoDimVert0]];
const whiteTable1 = [[-1, -1], [12, ccittEOL], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [11, 1792], [11, 1792], [12, 1984], [12, 2048], [12, 2112], [12, 2176], [12, 2240], [12, 2304], [11, 1856], [11, 1856], [11, 1920], [11, 1920], [12, 2368], [12, 2432], [12, 2496], [12, 2560]];
const whiteTable2 = [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [8, 29], [8, 29], [8, 30], [8, 30], [8, 45], [8, 45], [8, 46], [8, 46], [7, 22], [7, 22], [7, 22], [7, 22], [7, 23], [7, 23], [7, 23], [7, 23], [8, 47], [8, 47], [8, 48], [8, 48], [6, 13], [6, 13], [6, 13], [6, 13], [6, 13], [6, 13], [6, 13], [6, 13], [7, 20], [7, 20], [7, 20], [7, 20], [8, 33], [8, 33], [8, 34], [8, 34], [8, 35], [8, 35], [8, 36], [8, 36], [8, 37], [8, 37], [8, 38], [8, 38], [7, 19], [7, 19], [7, 19], [7, 19], [8, 31], [8, 31], [8, 32], [8, 32], [6, 1], [6, 1], [6, 1], [6, 1], [6, 1], [6, 1], [6, 1], [6, 1], [6, 12], [6, 12], [6, 12], [6, 12], [6, 12], [6, 12], [6, 12], [6, 12], [8, 53], [8, 53], [8, 54], [8, 54], [7, 26], [7, 26], [7, 26], [7, 26], [8, 39], [8, 39], [8, 40], [8, 40], [8, 41], [8, 41], [8, 42], [8, 42], [8, 43], [8, 43], [8, 44], [8, 44], [7, 21], [7, 21], [7, 21], [7, 21], [7, 28], [7, 28], [7, 28], [7, 28], [8, 61], [8, 61], [8, 62], [8, 62], [8, 63], [8, 63], [8, 0], [8, 0], [8, 320], [8, 320], [8, 384], [8, 384], [5, 10], [5, 10], [5, 10], [5, 10], [5, 10], [5, 10], [5, 10], [5, 10], [5, 10], [5, 10], [5, 10], [5, 10], [5, 10], [5, 10], [5, 10], [5, 10], [5, 11], [5, 11], [5, 11], [5, 11], [5, 11], [5, 11], [5, 11], [5, 11], [5, 11], [5, 11], [5, 11], [5, 11], [5, 11], [5, 11], [5, 11], [5, 11], [7, 27], [7, 27], [7, 27], [7, 27], [8, 59], [8, 59], [8, 60], [8, 60], [9, 1472], [9, 1536], [9, 1600], [9, 1728], [7, 18], [7, 18], [7, 18], [7, 18], [7, 24], [7, 24], [7, 24], [7, 24], [8, 49], [8, 49], [8, 50], [8, 50], [8, 51], [8, 51], [8, 52], [8, 52], [7, 25], [7, 25], [7, 25], [7, 25], [8, 55], [8, 55], [8, 56], [8, 56], [8, 57], [8, 57], [8, 58], [8, 58], [6, 192], [6, 192], [6, 192], [6, 192], [6, 192], [6, 192], [6, 192], [6, 192], [6, 1664], [6, 1664], [6, 1664], [6, 1664], [6, 1664], [6, 1664], [6, 1664], [6, 1664], [8, 448], [8, 448], [8, 512], [8, 512], [9, 704], [9, 768], [8, 640], [8, 640], [8, 576], [8, 576], [9, 832], [9, 896], [9, 960], [9, 1024], [9, 1088], [9, 1152], [9, 1216], [9, 1280], [9, 1344], [9, 1408], [7, 256], [7, 256], [7, 256], [7, 256], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 2], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [5, 128], [5, 128], [5, 128], [5, 128], [5, 128], [5, 128], [5, 128], [5, 128], [5, 128], [5, 128], [5, 128], [5, 128], [5, 128], [5, 128], [5, 128], [5, 128], [5, 8], [5, 8], [5, 8], [5, 8], [5, 8], [5, 8], [5, 8], [5, 8], [5, 8], [5, 8], [5, 8], [5, 8], [5, 8], [5, 8], [5, 8], [5, 8], [5, 9], [5, 9], [5, 9], [5, 9], [5, 9], [5, 9], [5, 9], [5, 9], [5, 9], [5, 9], [5, 9], [5, 9], [5, 9], [5, 9], [5, 9], [5, 9], [6, 16], [6, 16], [6, 16], [6, 16], [6, 16], [6, 16], [6, 16], [6, 16], [6, 17], [6, 17], [6, 17], [6, 17], [6, 17], [6, 17], [6, 17], [6, 17], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [6, 14], [6, 14], [6, 14], [6, 14], [6, 14], [6, 14], [6, 14], [6, 14], [6, 15], [6, 15], [6, 15], [6, 15], [6, 15], [6, 15], [6, 15], [6, 15], [5, 64], [5, 64], [5, 64], [5, 64], [5, 64], [5, 64], [5, 64], [5, 64], [5, 64], [5, 64], [5, 64], [5, 64], [5, 64], [5, 64], [5, 64], [5, 64], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 6], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7], [4, 7]];
const blackTable1 = [[-1, -1], [-1, -1], [12, ccittEOL], [12, ccittEOL], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [11, 1792], [11, 1792], [11, 1792], [11, 1792], [12, 1984], [12, 1984], [12, 2048], [12, 2048], [12, 2112], [12, 2112], [12, 2176], [12, 2176], [12, 2240], [12, 2240], [12, 2304], [12, 2304], [11, 1856], [11, 1856], [11, 1856], [11, 1856], [11, 1920], [11, 1920], [11, 1920], [11, 1920], [12, 2368], [12, 2368], [12, 2432], [12, 2432], [12, 2496], [12, 2496], [12, 2560], [12, 2560], [10, 18], [10, 18], [10, 18], [10, 18], [10, 18], [10, 18], [10, 18], [10, 18], [12, 52], [12, 52], [13, 640], [13, 704], [13, 768], [13, 832], [12, 55], [12, 55], [12, 56], [12, 56], [13, 1280], [13, 1344], [13, 1408], [13, 1472], [12, 59], [12, 59], [12, 60], [12, 60], [13, 1536], [13, 1600], [11, 24], [11, 24], [11, 24], [11, 24], [11, 25], [11, 25], [11, 25], [11, 25], [13, 1664], [13, 1728], [12, 320], [12, 320], [12, 384], [12, 384], [12, 448], [12, 448], [13, 512], [13, 576], [12, 53], [12, 53], [12, 54], [12, 54], [13, 896], [13, 960], [13, 1024], [13, 1088], [13, 1152], [13, 1216], [10, 64], [10, 64], [10, 64], [10, 64], [10, 64], [10, 64], [10, 64], [10, 64]];
const blackTable2 = [[8, 13], [8, 13], [8, 13], [8, 13], [8, 13], [8, 13], [8, 13], [8, 13], [8, 13], [8, 13], [8, 13], [8, 13], [8, 13], [8, 13], [8, 13], [8, 13], [11, 23], [11, 23], [12, 50], [12, 51], [12, 44], [12, 45], [12, 46], [12, 47], [12, 57], [12, 58], [12, 61], [12, 256], [10, 16], [10, 16], [10, 16], [10, 16], [10, 17], [10, 17], [10, 17], [10, 17], [12, 48], [12, 49], [12, 62], [12, 63], [12, 30], [12, 31], [12, 32], [12, 33], [12, 40], [12, 41], [11, 22], [11, 22], [8, 14], [8, 14], [8, 14], [8, 14], [8, 14], [8, 14], [8, 14], [8, 14], [8, 14], [8, 14], [8, 14], [8, 14], [8, 14], [8, 14], [8, 14], [8, 14], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 10], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [7, 11], [9, 15], [9, 15], [9, 15], [9, 15], [9, 15], [9, 15], [9, 15], [9, 15], [12, 128], [12, 192], [12, 26], [12, 27], [12, 28], [12, 29], [11, 19], [11, 19], [11, 20], [11, 20], [12, 34], [12, 35], [12, 36], [12, 37], [12, 38], [12, 39], [11, 21], [11, 21], [12, 42], [12, 43], [10, 0], [10, 0], [10, 0], [10, 0], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12], [7, 12]];
const blackTable3 = [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [6, 9], [6, 8], [5, 7], [5, 7], [4, 6], [4, 6], [4, 6], [4, 6], [4, 5], [4, 5], [4, 5], [4, 5], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1], [3, 4], [3, 4], [3, 4], [3, 4], [3, 4], [3, 4], [3, 4], [3, 4], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]];
class CCITTFaxDecoder {
  constructor(source, options = {}) {
    if (typeof source?.next !== "function") {
      throw new Error('CCITTFaxDecoder - invalid "source" parameter.');
    }
    this.source = source;
    this.eof = false;
    this.encoding = options.K || 0;
    this.eoline = options.EndOfLine || false;
    this.byteAlign = options.EncodedByteAlign || false;
    this.columns = options.Columns || 1728;
    this.rows = options.Rows || 0;
    this.eoblock = options.EndOfBlock ?? true;
    this.black = options.BlackIs1 || false;
    this.codingLine = new Uint32Array(this.columns + 1);
    this.refLine = new Uint32Array(this.columns + 2);
    this.codingLine[0] = this.columns;
    this.codingPos = 0;
    this.row = 0;
    this.nextLine2D = this.encoding < 0;
    this.inputBits = 0;
    this.inputBuf = 0;
    this.outputBits = 0;
    this.rowsDone = false;
    let code1;
    while ((code1 = this._lookBits(12)) === 0) {
      this._eatBits(1);
    }
    if (code1 === 1) {
      this._eatBits(12);
    }
    if (this.encoding > 0) {
      this.nextLine2D = !this._lookBits(1);
      this._eatBits(1);
    }
  }
  readNextChar() {
    if (this.eof) {
      return -1;
    }
    const refLine = this.refLine;
    const codingLine = this.codingLine;
    const columns = this.columns;
    let refPos, blackPixels, bits, i;
    if (this.outputBits === 0) {
      if (this.rowsDone) {
        this.eof = true;
      }
      if (this.eof) {
        return -1;
      }
      this.err = false;
      let code1, code2, code3;
      if (this.nextLine2D) {
        for (i = 0; codingLine[i] < columns; ++i) {
          refLine[i] = codingLine[i];
        }
        refLine[i++] = columns;
        refLine[i] = columns;
        codingLine[0] = 0;
        this.codingPos = 0;
        refPos = 0;
        blackPixels = 0;
        while (codingLine[this.codingPos] < columns) {
          code1 = this._getTwoDimCode();
          switch (code1) {
            case twoDimPass:
              this._addPixels(refLine[refPos + 1], blackPixels);
              if (refLine[refPos + 1] < columns) {
                refPos += 2;
              }
              break;
            case twoDimHoriz:
              code1 = code2 = 0;
              if (blackPixels) {
                do {
                  code1 += code3 = this._getBlackCode();
                } while (code3 >= 64);
                do {
                  code2 += code3 = this._getWhiteCode();
                } while (code3 >= 64);
              } else {
                do {
                  code1 += code3 = this._getWhiteCode();
                } while (code3 >= 64);
                do {
                  code2 += code3 = this._getBlackCode();
                } while (code3 >= 64);
              }
              this._addPixels(codingLine[this.codingPos] + code1, blackPixels);
              if (codingLine[this.codingPos] < columns) {
                this._addPixels(codingLine[this.codingPos] + code2, blackPixels ^ 1);
              }
              while (refLine[refPos] <= codingLine[this.codingPos] && refLine[refPos] < columns) {
                refPos += 2;
              }
              break;
            case twoDimVertR3:
              this._addPixels(refLine[refPos] + 3, blackPixels);
              blackPixels ^= 1;
              if (codingLine[this.codingPos] < columns) {
                ++refPos;
                while (refLine[refPos] <= codingLine[this.codingPos] && refLine[refPos] < columns) {
                  refPos += 2;
                }
              }
              break;
            case twoDimVertR2:
              this._addPixels(refLine[refPos] + 2, blackPixels);
              blackPixels ^= 1;
              if (codingLine[this.codingPos] < columns) {
                ++refPos;
                while (refLine[refPos] <= codingLine[this.codingPos] && refLine[refPos] < columns) {
                  refPos += 2;
                }
              }
              break;
            case twoDimVertR1:
              this._addPixels(refLine[refPos] + 1, blackPixels);
              blackPixels ^= 1;
              if (codingLine[this.codingPos] < columns) {
                ++refPos;
                while (refLine[refPos] <= codingLine[this.codingPos] && refLine[refPos] < columns) {
                  refPos += 2;
                }
              }
              break;
            case twoDimVert0:
              this._addPixels(refLine[refPos], blackPixels);
              blackPixels ^= 1;
              if (codingLine[this.codingPos] < columns) {
                ++refPos;
                while (refLine[refPos] <= codingLine[this.codingPos] && refLine[refPos] < columns) {
                  refPos += 2;
                }
              }
              break;
            case twoDimVertL3:
              this._addPixelsNeg(refLine[refPos] - 3, blackPixels);
              blackPixels ^= 1;
              if (codingLine[this.codingPos] < columns) {
                if (refPos > 0) {
                  --refPos;
                } else {
                  ++refPos;
                }
                while (refLine[refPos] <= codingLine[this.codingPos] && refLine[refPos] < columns) {
                  refPos += 2;
                }
              }
              break;
            case twoDimVertL2:
              this._addPixelsNeg(refLine[refPos] - 2, blackPixels);
              blackPixels ^= 1;
              if (codingLine[this.codingPos] < columns) {
                if (refPos > 0) {
                  --refPos;
                } else {
                  ++refPos;
                }
                while (refLine[refPos] <= codingLine[this.codingPos] && refLine[refPos] < columns) {
                  refPos += 2;
                }
              }
              break;
            case twoDimVertL1:
              this._addPixelsNeg(refLine[refPos] - 1, blackPixels);
              blackPixels ^= 1;
              if (codingLine[this.codingPos] < columns) {
                if (refPos > 0) {
                  --refPos;
                } else {
                  ++refPos;
                }
                while (refLine[refPos] <= codingLine[this.codingPos] && refLine[refPos] < columns) {
                  refPos += 2;
                }
              }
              break;
            case ccittEOF:
              this._addPixels(columns, 0);
              this.eof = true;
              break;
            default:
              info("bad 2d code");
              this._addPixels(columns, 0);
              this.err = true;
          }
        }
      } else {
        codingLine[0] = 0;
        this.codingPos = 0;
        blackPixels = 0;
        while (codingLine[this.codingPos] < columns) {
          code1 = 0;
          if (blackPixels) {
            do {
              code1 += code3 = this._getBlackCode();
            } while (code3 >= 64);
          } else {
            do {
              code1 += code3 = this._getWhiteCode();
            } while (code3 >= 64);
          }
          this._addPixels(codingLine[this.codingPos] + code1, blackPixels);
          blackPixels ^= 1;
        }
      }
      let gotEOL = false;
      if (this.byteAlign) {
        this.inputBits &= ~7;
      }
      if (!this.eoblock && this.row === this.rows - 1) {
        this.rowsDone = true;
      } else {
        code1 = this._lookBits(12);
        if (this.eoline) {
          while (code1 !== ccittEOF && code1 !== 1) {
            this._eatBits(1);
            code1 = this._lookBits(12);
          }
        } else {
          while (code1 === 0) {
            this._eatBits(1);
            code1 = this._lookBits(12);
          }
        }
        if (code1 === 1) {
          this._eatBits(12);
          gotEOL = true;
        } else if (code1 === ccittEOF) {
          this.eof = true;
        }
      }
      if (!this.eof && this.encoding > 0 && !this.rowsDone) {
        this.nextLine2D = !this._lookBits(1);
        this._eatBits(1);
      }
      if (this.eoblock && gotEOL && this.byteAlign) {
        code1 = this._lookBits(12);
        if (code1 === 1) {
          this._eatBits(12);
          if (this.encoding > 0) {
            this._lookBits(1);
            this._eatBits(1);
          }
          if (this.encoding >= 0) {
            for (i = 0; i < 4; ++i) {
              code1 = this._lookBits(12);
              if (code1 !== 1) {
                info("bad rtc code: " + code1);
              }
              this._eatBits(12);
              if (this.encoding > 0) {
                this._lookBits(1);
                this._eatBits(1);
              }
            }
          }
          this.eof = true;
        }
      } else if (this.err && this.eoline) {
        while (true) {
          code1 = this._lookBits(13);
          if (code1 === ccittEOF) {
            this.eof = true;
            return -1;
          }
          if (code1 >> 1 === 1) {
            break;
          }
          this._eatBits(1);
        }
        this._eatBits(12);
        if (this.encoding > 0) {
          this._eatBits(1);
          this.nextLine2D = !(code1 & 1);
        }
      }
      this.outputBits = codingLine[0] > 0 ? codingLine[this.codingPos = 0] : codingLine[this.codingPos = 1];
      this.row++;
    }
    let c;
    if (this.outputBits >= 8) {
      c = this.codingPos & 1 ? 0 : 0xff;
      this.outputBits -= 8;
      if (this.outputBits === 0 && codingLine[this.codingPos] < columns) {
        this.codingPos++;
        this.outputBits = codingLine[this.codingPos] - codingLine[this.codingPos - 1];
      }
    } else {
      bits = 8;
      c = 0;
      do {
        if (typeof this.outputBits !== "number") {
          throw new FormatError('Invalid /CCITTFaxDecode data, "outputBits" must be a number.');
        }
        if (this.outputBits > bits) {
          c <<= bits;
          if (!(this.codingPos & 1)) {
            c |= 0xff >> 8 - bits;
          }
          this.outputBits -= bits;
          bits = 0;
        } else {
          c <<= this.outputBits;
          if (!(this.codingPos & 1)) {
            c |= 0xff >> 8 - this.outputBits;
          }
          bits -= this.outputBits;
          this.outputBits = 0;
          if (codingLine[this.codingPos] < columns) {
            this.codingPos++;
            this.outputBits = codingLine[this.codingPos] - codingLine[this.codingPos - 1];
          } else if (bits > 0) {
            c <<= bits;
            bits = 0;
          }
        }
      } while (bits);
    }
    if (this.black) {
      c ^= 0xff;
    }
    return c;
  }
  _addPixels(a1, blackPixels) {
    const codingLine = this.codingLine;
    let codingPos = this.codingPos;
    if (a1 > codingLine[codingPos]) {
      if (a1 > this.columns) {
        info("row is wrong length");
        this.err = true;
        a1 = this.columns;
      }
      if (codingPos & 1 ^ blackPixels) {
        ++codingPos;
      }
      codingLine[codingPos] = a1;
    }
    this.codingPos = codingPos;
  }
  _addPixelsNeg(a1, blackPixels) {
    const codingLine = this.codingLine;
    let codingPos = this.codingPos;
    if (a1 > codingLine[codingPos]) {
      if (a1 > this.columns) {
        info("row is wrong length");
        this.err = true;
        a1 = this.columns;
      }
      if (codingPos & 1 ^ blackPixels) {
        ++codingPos;
      }
      codingLine[codingPos] = a1;
    } else if (a1 < codingLine[codingPos]) {
      if (a1 < 0) {
        info("invalid code");
        this.err = true;
        a1 = 0;
      }
      while (codingPos > 0 && a1 < codingLine[codingPos - 1]) {
        --codingPos;
      }
      codingLine[codingPos] = a1;
    }
    this.codingPos = codingPos;
  }
  _findTableCode(start, end, table, limit) {
    const limitValue = limit || 0;
    for (let i = start; i <= end; ++i) {
      let code = this._lookBits(i);
      if (code === ccittEOF) {
        return [true, 1, false];
      }
      if (i < end) {
        code <<= end - i;
      }
      if (!limitValue || code >= limitValue) {
        const p = table[code - limitValue];
        if (p[0] === i) {
          this._eatBits(i);
          return [true, p[1], true];
        }
      }
    }
    return [false, 0, false];
  }
  _getTwoDimCode() {
    let code = 0;
    let p;
    if (this.eoblock) {
      code = this._lookBits(7);
      p = twoDimTable[code];
      if (p?.[0] > 0) {
        this._eatBits(p[0]);
        return p[1];
      }
    } else {
      const result = this._findTableCode(1, 7, twoDimTable);
      if (result[0] && result[2]) {
        return result[1];
      }
    }
    info("Bad two dim code");
    return ccittEOF;
  }
  _getWhiteCode() {
    let code = 0;
    let p;
    if (this.eoblock) {
      code = this._lookBits(12);
      if (code === ccittEOF) {
        return 1;
      }
      p = code >> 5 === 0 ? whiteTable1[code] : whiteTable2[code >> 3];
      if (p[0] > 0) {
        this._eatBits(p[0]);
        return p[1];
      }
    } else {
      let result = this._findTableCode(1, 9, whiteTable2);
      if (result[0]) {
        return result[1];
      }
      result = this._findTableCode(11, 12, whiteTable1);
      if (result[0]) {
        return result[1];
      }
    }
    info("bad white code");
    this._eatBits(1);
    return 1;
  }
  _getBlackCode() {
    let code, p;
    if (this.eoblock) {
      code = this._lookBits(13);
      if (code === ccittEOF) {
        return 1;
      }
      if (code >> 7 === 0) {
        p = blackTable1[code];
      } else if (code >> 9 === 0 && code >> 7 !== 0) {
        p = blackTable2[(code >> 1) - 64];
      } else {
        p = blackTable3[code >> 7];
      }
      if (p[0] > 0) {
        this._eatBits(p[0]);
        return p[1];
      }
    } else {
      let result = this._findTableCode(2, 6, blackTable3);
      if (result[0]) {
        return result[1];
      }
      result = this._findTableCode(7, 12, blackTable2, 64);
      if (result[0]) {
        return result[1];
      }
      result = this._findTableCode(10, 13, blackTable1);
      if (result[0]) {
        return result[1];
      }
    }
    info("bad black code");
    this._eatBits(1);
    return 1;
  }
  _lookBits(n) {
    let c;
    while (this.inputBits < n) {
      if ((c = this.source.next()) === -1) {
        if (this.inputBits === 0) {
          return ccittEOF;
        }
        return this.inputBuf << n - this.inputBits & 0xffff >> 16 - n;
      }
      this.inputBuf = this.inputBuf << 8 | c;
      this.inputBits += 8;
    }
    return this.inputBuf >> this.inputBits - n & 0xffff >> 16 - n;
  }
  _eatBits(n) {
    if ((this.inputBits -= n) < 0) {
      this.inputBits = 0;
    }
  }
}

;// ./src/core/jbig2.js











class Jbig2Error extends BaseException {
  constructor(msg) {
    super(msg, "Jbig2Error");
  }
}
class ContextCache {
  getContexts(id) {
    if (id in this) {
      return this[id];
    }
    return this[id] = new Int8Array(1 << 16);
  }
}
class DecodingContext {
  constructor(data, start, end) {
    this.data = data;
    this.start = start;
    this.end = end;
  }
  get decoder() {
    const decoder = new ArithmeticDecoder(this.data, this.start, this.end);
    return shadow(this, "decoder", decoder);
  }
  get contextCache() {
    const cache = new ContextCache();
    return shadow(this, "contextCache", cache);
  }
}
function decodeInteger(contextCache, procedure, decoder) {
  const contexts = contextCache.getContexts(procedure);
  let prev = 1;
  function readBits(length) {
    let v = 0;
    for (let i = 0; i < length; i++) {
      const bit = decoder.readBit(contexts, prev);
      prev = prev < 256 ? prev << 1 | bit : (prev << 1 | bit) & 511 | 256;
      v = v << 1 | bit;
    }
    return v >>> 0;
  }
  const sign = readBits(1);
  const value = readBits(1) ? readBits(1) ? readBits(1) ? readBits(1) ? readBits(1) ? readBits(32) + 4436 : readBits(12) + 340 : readBits(8) + 84 : readBits(6) + 20 : readBits(4) + 4 : readBits(2);
  let signedValue;
  if (sign === 0) {
    signedValue = value;
  } else if (value > 0) {
    signedValue = -value;
  }
  if (signedValue >= MIN_INT_32 && signedValue <= MAX_INT_32) {
    return signedValue;
  }
  return null;
}
function decodeIAID(contextCache, decoder, codeLength) {
  const contexts = contextCache.getContexts("IAID");
  let prev = 1;
  for (let i = 0; i < codeLength; i++) {
    const bit = decoder.readBit(contexts, prev);
    prev = prev << 1 | bit;
  }
  if (codeLength < 31) {
    return prev & (1 << codeLength) - 1;
  }
  return prev & 0x7fffffff;
}
const SegmentTypes = ["SymbolDictionary", null, null, null, "IntermediateTextRegion", null, "ImmediateTextRegion", "ImmediateLosslessTextRegion", null, null, null, null, null, null, null, null, "PatternDictionary", null, null, null, "IntermediateHalftoneRegion", null, "ImmediateHalftoneRegion", "ImmediateLosslessHalftoneRegion", null, null, null, null, null, null, null, null, null, null, null, null, "IntermediateGenericRegion", null, "ImmediateGenericRegion", "ImmediateLosslessGenericRegion", "IntermediateGenericRefinementRegion", null, "ImmediateGenericRefinementRegion", "ImmediateLosslessGenericRefinementRegion", null, null, null, null, "PageInformation", "EndOfPage", "EndOfStripe", "EndOfFile", "Profiles", "Tables", null, null, null, null, null, null, null, null, "Extension"];
const CodingTemplates = [[{
  x: -1,
  y: -2
}, {
  x: 0,
  y: -2
}, {
  x: 1,
  y: -2
}, {
  x: -2,
  y: -1
}, {
  x: -1,
  y: -1
}, {
  x: 0,
  y: -1
}, {
  x: 1,
  y: -1
}, {
  x: 2,
  y: -1
}, {
  x: -4,
  y: 0
}, {
  x: -3,
  y: 0
}, {
  x: -2,
  y: 0
}, {
  x: -1,
  y: 0
}], [{
  x: -1,
  y: -2
}, {
  x: 0,
  y: -2
}, {
  x: 1,
  y: -2
}, {
  x: 2,
  y: -2
}, {
  x: -2,
  y: -1
}, {
  x: -1,
  y: -1
}, {
  x: 0,
  y: -1
}, {
  x: 1,
  y: -1
}, {
  x: 2,
  y: -1
}, {
  x: -3,
  y: 0
}, {
  x: -2,
  y: 0
}, {
  x: -1,
  y: 0
}], [{
  x: -1,
  y: -2
}, {
  x: 0,
  y: -2
}, {
  x: 1,
  y: -2
}, {
  x: -2,
  y: -1
}, {
  x: -1,
  y: -1
}, {
  x: 0,
  y: -1
}, {
  x: 1,
  y: -1
}, {
  x: -2,
  y: 0
}, {
  x: -1,
  y: 0
}], [{
  x: -3,
  y: -1
}, {
  x: -2,
  y: -1
}, {
  x: -1,
  y: -1
}, {
  x: 0,
  y: -1
}, {
  x: 1,
  y: -1
}, {
  x: -4,
  y: 0
}, {
  x: -3,
  y: 0
}, {
  x: -2,
  y: 0
}, {
  x: -1,
  y: 0
}]];
const RefinementTemplates = [{
  coding: [{
    x: 0,
    y: -1
  }, {
    x: 1,
    y: -1
  }, {
    x: -1,
    y: 0
  }],
  reference: [{
    x: 0,
    y: -1
  }, {
    x: 1,
    y: -1
  }, {
    x: -1,
    y: 0
  }, {
    x: 0,
    y: 0
  }, {
    x: 1,
    y: 0
  }, {
    x: -1,
    y: 1
  }, {
    x: 0,
    y: 1
  }, {
    x: 1,
    y: 1
  }]
}, {
  coding: [{
    x: -1,
    y: -1
  }, {
    x: 0,
    y: -1
  }, {
    x: 1,
    y: -1
  }, {
    x: -1,
    y: 0
  }],
  reference: [{
    x: 0,
    y: -1
  }, {
    x: -1,
    y: 0
  }, {
    x: 0,
    y: 0
  }, {
    x: 1,
    y: 0
  }, {
    x: 0,
    y: 1
  }, {
    x: 1,
    y: 1
  }]
}];
const ReusedContexts = [0x9b25, 0x0795, 0x00e5, 0x0195];
const RefinementReusedContexts = [0x0020, 0x0008];
function decodeBitmapTemplate0(width, height, decodingContext) {
  const decoder = decodingContext.decoder;
  const contexts = decodingContext.contextCache.getContexts("GB");
  const bitmap = [];
  let contextLabel, i, j, pixel, row, row1, row2;
  const OLD_PIXEL_MASK = 0x7bf7;
  for (i = 0; i < height; i++) {
    row = bitmap[i] = new Uint8Array(width);
    row1 = i < 1 ? row : bitmap[i - 1];
    row2 = i < 2 ? row : bitmap[i - 2];
    contextLabel = row2[0] << 13 | row2[1] << 12 | row2[2] << 11 | row1[0] << 7 | row1[1] << 6 | row1[2] << 5 | row1[3] << 4;
    for (j = 0; j < width; j++) {
      row[j] = pixel = decoder.readBit(contexts, contextLabel);
      contextLabel = (contextLabel & OLD_PIXEL_MASK) << 1 | (j + 3 < width ? row2[j + 3] << 11 : 0) | (j + 4 < width ? row1[j + 4] << 4 : 0) | pixel;
    }
  }
  return bitmap;
}
function decodeBitmap(mmr, width, height, templateIndex, prediction, skip, at, decodingContext) {
  if (mmr) {
    const input = new Reader(decodingContext.data, decodingContext.start, decodingContext.end);
    return decodeMMRBitmap(input, width, height, false);
  }
  if (templateIndex === 0 && !skip && !prediction && at.length === 4 && at[0].x === 3 && at[0].y === -1 && at[1].x === -3 && at[1].y === -1 && at[2].x === 2 && at[2].y === -2 && at[3].x === -2 && at[3].y === -2) {
    return decodeBitmapTemplate0(width, height, decodingContext);
  }
  const useskip = !!skip;
  const template = CodingTemplates[templateIndex].concat(at);
  template.sort(function (a, b) {
    return a.y - b.y || a.x - b.x;
  });
  const templateLength = template.length;
  const templateX = new Int8Array(templateLength);
  const templateY = new Int8Array(templateLength);
  const changingTemplateEntries = [];
  let reuseMask = 0,
    minX = 0,
    maxX = 0,
    minY = 0;
  let c, k;
  for (k = 0; k < templateLength; k++) {
    templateX[k] = template[k].x;
    templateY[k] = template[k].y;
    minX = Math.min(minX, template[k].x);
    maxX = Math.max(maxX, template[k].x);
    minY = Math.min(minY, template[k].y);
    if (k < templateLength - 1 && template[k].y === template[k + 1].y && template[k].x === template[k + 1].x - 1) {
      reuseMask |= 1 << templateLength - 1 - k;
    } else {
      changingTemplateEntries.push(k);
    }
  }
  const changingEntriesLength = changingTemplateEntries.length;
  const changingTemplateX = new Int8Array(changingEntriesLength);
  const changingTemplateY = new Int8Array(changingEntriesLength);
  const changingTemplateBit = new Uint16Array(changingEntriesLength);
  for (c = 0; c < changingEntriesLength; c++) {
    k = changingTemplateEntries[c];
    changingTemplateX[c] = template[k].x;
    changingTemplateY[c] = template[k].y;
    changingTemplateBit[c] = 1 << templateLength - 1 - k;
  }
  const sbb_left = -minX;
  const sbb_top = -minY;
  const sbb_right = width - maxX;
  const pseudoPixelContext = ReusedContexts[templateIndex];
  let row = new Uint8Array(width);
  const bitmap = [];
  const decoder = decodingContext.decoder;
  const contexts = decodingContext.contextCache.getContexts("GB");
  let ltp = 0,
    j,
    i0,
    j0,
    contextLabel = 0,
    bit,
    shift;
  for (let i = 0; i < height; i++) {
    if (prediction) {
      const sltp = decoder.readBit(contexts, pseudoPixelContext);
      ltp ^= sltp;
      if (ltp) {
        bitmap.push(row);
        continue;
      }
    }
    row = new Uint8Array(row);
    bitmap.push(row);
    for (j = 0; j < width; j++) {
      if (useskip && skip[i][j]) {
        row[j] = 0;
        continue;
      }
      if (j >= sbb_left && j < sbb_right && i >= sbb_top) {
        contextLabel = contextLabel << 1 & reuseMask;
        for (k = 0; k < changingEntriesLength; k++) {
          i0 = i + changingTemplateY[k];
          j0 = j + changingTemplateX[k];
          bit = bitmap[i0][j0];
          if (bit) {
            bit = changingTemplateBit[k];
            contextLabel |= bit;
          }
        }
      } else {
        contextLabel = 0;
        shift = templateLength - 1;
        for (k = 0; k < templateLength; k++, shift--) {
          j0 = j + templateX[k];
          if (j0 >= 0 && j0 < width) {
            i0 = i + templateY[k];
            if (i0 >= 0) {
              bit = bitmap[i0][j0];
              if (bit) {
                contextLabel |= bit << shift;
              }
            }
          }
        }
      }
      const pixel = decoder.readBit(contexts, contextLabel);
      row[j] = pixel;
    }
  }
  return bitmap;
}
function decodeRefinement(width, height, templateIndex, referenceBitmap, offsetX, offsetY, prediction, at, decodingContext) {
  let codingTemplate = RefinementTemplates[templateIndex].coding;
  if (templateIndex === 0) {
    codingTemplate = codingTemplate.concat([at[0]]);
  }
  const codingTemplateLength = codingTemplate.length;
  const codingTemplateX = new Int32Array(codingTemplateLength);
  const codingTemplateY = new Int32Array(codingTemplateLength);
  let k;
  for (k = 0; k < codingTemplateLength; k++) {
    codingTemplateX[k] = codingTemplate[k].x;
    codingTemplateY[k] = codingTemplate[k].y;
  }
  let referenceTemplate = RefinementTemplates[templateIndex].reference;
  if (templateIndex === 0) {
    referenceTemplate = referenceTemplate.concat([at[1]]);
  }
  const referenceTemplateLength = referenceTemplate.length;
  const referenceTemplateX = new Int32Array(referenceTemplateLength);
  const referenceTemplateY = new Int32Array(referenceTemplateLength);
  for (k = 0; k < referenceTemplateLength; k++) {
    referenceTemplateX[k] = referenceTemplate[k].x;
    referenceTemplateY[k] = referenceTemplate[k].y;
  }
  const referenceWidth = referenceBitmap[0].length;
  const referenceHeight = referenceBitmap.length;
  const pseudoPixelContext = RefinementReusedContexts[templateIndex];
  const bitmap = [];
  const decoder = decodingContext.decoder;
  const contexts = decodingContext.contextCache.getContexts("GR");
  let ltp = 0;
  for (let i = 0; i < height; i++) {
    if (prediction) {
      const sltp = decoder.readBit(contexts, pseudoPixelContext);
      ltp ^= sltp;
      if (ltp) {
        throw new Jbig2Error("prediction is not supported");
      }
    }
    const row = new Uint8Array(width);
    bitmap.push(row);
    for (let j = 0; j < width; j++) {
      let i0, j0;
      let contextLabel = 0;
      for (k = 0; k < codingTemplateLength; k++) {
        i0 = i + codingTemplateY[k];
        j0 = j + codingTemplateX[k];
        if (i0 < 0 || j0 < 0 || j0 >= width) {
          contextLabel <<= 1;
        } else {
          contextLabel = contextLabel << 1 | bitmap[i0][j0];
        }
      }
      for (k = 0; k < referenceTemplateLength; k++) {
        i0 = i + referenceTemplateY[k] - offsetY;
        j0 = j + referenceTemplateX[k] - offsetX;
        if (i0 < 0 || i0 >= referenceHeight || j0 < 0 || j0 >= referenceWidth) {
          contextLabel <<= 1;
        } else {
          contextLabel = contextLabel << 1 | referenceBitmap[i0][j0];
        }
      }
      const pixel = decoder.readBit(contexts, contextLabel);
      row[j] = pixel;
    }
  }
  return bitmap;
}
function decodeSymbolDictionary(huffman, refinement, symbols, numberOfNewSymbols, numberOfExportedSymbols, huffmanTables, templateIndex, at, refinementTemplateIndex, refinementAt, decodingContext, huffmanInput) {
  if (huffman && refinement) {
    throw new Jbig2Error("symbol refinement with Huffman is not supported");
  }
  const newSymbols = [];
  let currentHeight = 0;
  let symbolCodeLength = log2(symbols.length + numberOfNewSymbols);
  const decoder = decodingContext.decoder;
  const contextCache = decodingContext.contextCache;
  let tableB1, symbolWidths;
  if (huffman) {
    tableB1 = getStandardTable(1);
    symbolWidths = [];
    symbolCodeLength = Math.max(symbolCodeLength, 1);
  }
  while (newSymbols.length < numberOfNewSymbols) {
    const deltaHeight = huffman ? huffmanTables.tableDeltaHeight.decode(huffmanInput) : decodeInteger(contextCache, "IADH", decoder);
    currentHeight += deltaHeight;
    let currentWidth = 0,
      totalWidth = 0;
    const firstSymbol = huffman ? symbolWidths.length : 0;
    while (true) {
      const deltaWidth = huffman ? huffmanTables.tableDeltaWidth.decode(huffmanInput) : decodeInteger(contextCache, "IADW", decoder);
      if (deltaWidth === null) {
        break;
      }
      currentWidth += deltaWidth;
      totalWidth += currentWidth;
      let bitmap;
      if (refinement) {
        const numberOfInstances = decodeInteger(contextCache, "IAAI", decoder);
        if (numberOfInstances > 1) {
          bitmap = decodeTextRegion(huffman, refinement, currentWidth, currentHeight, 0, numberOfInstances, 1, symbols.concat(newSymbols), symbolCodeLength, 0, 0, 1, 0, huffmanTables, refinementTemplateIndex, refinementAt, decodingContext, 0, huffmanInput);
        } else {
          const symbolId = decodeIAID(contextCache, decoder, symbolCodeLength);
          const rdx = decodeInteger(contextCache, "IARDX", decoder);
          const rdy = decodeInteger(contextCache, "IARDY", decoder);
          const symbol = symbolId < symbols.length ? symbols[symbolId] : newSymbols[symbolId - symbols.length];
          bitmap = decodeRefinement(currentWidth, currentHeight, refinementTemplateIndex, symbol, rdx, rdy, false, refinementAt, decodingContext);
        }
        newSymbols.push(bitmap);
      } else if (huffman) {
        symbolWidths.push(currentWidth);
      } else {
        bitmap = decodeBitmap(false, currentWidth, currentHeight, templateIndex, false, null, at, decodingContext);
        newSymbols.push(bitmap);
      }
    }
    if (huffman && !refinement) {
      const bitmapSize = huffmanTables.tableBitmapSize.decode(huffmanInput);
      huffmanInput.byteAlign();
      let collectiveBitmap;
      if (bitmapSize === 0) {
        collectiveBitmap = readUncompressedBitmap(huffmanInput, totalWidth, currentHeight);
      } else {
        const originalEnd = huffmanInput.end;
        const bitmapEnd = huffmanInput.position + bitmapSize;
        huffmanInput.end = bitmapEnd;
        collectiveBitmap = decodeMMRBitmap(huffmanInput, totalWidth, currentHeight, false);
        huffmanInput.end = originalEnd;
        huffmanInput.position = bitmapEnd;
      }
      const numberOfSymbolsDecoded = symbolWidths.length;
      if (firstSymbol === numberOfSymbolsDecoded - 1) {
        newSymbols.push(collectiveBitmap);
      } else {
        let i,
          y,
          xMin = 0,
          xMax,
          bitmapWidth,
          symbolBitmap;
        for (i = firstSymbol; i < numberOfSymbolsDecoded; i++) {
          bitmapWidth = symbolWidths[i];
          xMax = xMin + bitmapWidth;
          symbolBitmap = [];
          for (y = 0; y < currentHeight; y++) {
            symbolBitmap.push(collectiveBitmap[y].subarray(xMin, xMax));
          }
          newSymbols.push(symbolBitmap);
          xMin = xMax;
        }
      }
    }
  }
  const exportedSymbols = [],
    flags = [];
  let currentFlag = false,
    i,
    ii;
  const totalSymbolsLength = symbols.length + numberOfNewSymbols;
  while (flags.length < totalSymbolsLength) {
    let runLength = huffman ? tableB1.decode(huffmanInput) : decodeInteger(contextCache, "IAEX", decoder);
    while (runLength--) {
      flags.push(currentFlag);
    }
    currentFlag = !currentFlag;
  }
  for (i = 0, ii = symbols.length; i < ii; i++) {
    if (flags[i]) {
      exportedSymbols.push(symbols[i]);
    }
  }
  for (let j = 0; j < numberOfNewSymbols; i++, j++) {
    if (flags[i]) {
      exportedSymbols.push(newSymbols[j]);
    }
  }
  return exportedSymbols;
}
function decodeTextRegion(huffman, refinement, width, height, defaultPixelValue, numberOfSymbolInstances, stripSize, inputSymbols, symbolCodeLength, transposed, dsOffset, referenceCorner, combinationOperator, huffmanTables, refinementTemplateIndex, refinementAt, decodingContext, logStripSize, huffmanInput) {
  if (huffman && refinement) {
    throw new Jbig2Error("refinement with Huffman is not supported");
  }
  const bitmap = [];
  let i, row;
  for (i = 0; i < height; i++) {
    row = new Uint8Array(width);
    if (defaultPixelValue) {
      for (let j = 0; j < width; j++) {
        row[j] = defaultPixelValue;
      }
    }
    bitmap.push(row);
  }
  const decoder = decodingContext.decoder;
  const contextCache = decodingContext.contextCache;
  let stripT = huffman ? -huffmanTables.tableDeltaT.decode(huffmanInput) : -decodeInteger(contextCache, "IADT", decoder);
  let firstS = 0;
  i = 0;
  while (i < numberOfSymbolInstances) {
    const deltaT = huffman ? huffmanTables.tableDeltaT.decode(huffmanInput) : decodeInteger(contextCache, "IADT", decoder);
    stripT += deltaT;
    const deltaFirstS = huffman ? huffmanTables.tableFirstS.decode(huffmanInput) : decodeInteger(contextCache, "IAFS", decoder);
    firstS += deltaFirstS;
    let currentS = firstS;
    do {
      let currentT = 0;
      if (stripSize > 1) {
        currentT = huffman ? huffmanInput.readBits(logStripSize) : decodeInteger(contextCache, "IAIT", decoder);
      }
      const t = stripSize * stripT + currentT;
      const symbolId = huffman ? huffmanTables.symbolIDTable.decode(huffmanInput) : decodeIAID(contextCache, decoder, symbolCodeLength);
      const applyRefinement = refinement && (huffman ? huffmanInput.readBit() : decodeInteger(contextCache, "IARI", decoder));
      let symbolBitmap = inputSymbols[symbolId];
      let symbolWidth = symbolBitmap[0].length;
      let symbolHeight = symbolBitmap.length;
      if (applyRefinement) {
        const rdw = decodeInteger(contextCache, "IARDW", decoder);
        const rdh = decodeInteger(contextCache, "IARDH", decoder);
        const rdx = decodeInteger(contextCache, "IARDX", decoder);
        const rdy = decodeInteger(contextCache, "IARDY", decoder);
        symbolWidth += rdw;
        symbolHeight += rdh;
        symbolBitmap = decodeRefinement(symbolWidth, symbolHeight, refinementTemplateIndex, symbolBitmap, (rdw >> 1) + rdx, (rdh >> 1) + rdy, false, refinementAt, decodingContext);
      }
      let increment = 0;
      if (!transposed) {
        if (referenceCorner > 1) {
          currentS += symbolWidth - 1;
        } else {
          increment = symbolWidth - 1;
        }
      } else if (!(referenceCorner & 1)) {
        currentS += symbolHeight - 1;
      } else {
        increment = symbolHeight - 1;
      }
      const offsetT = t - (referenceCorner & 1 ? 0 : symbolHeight - 1);
      const offsetS = currentS - (referenceCorner & 2 ? symbolWidth - 1 : 0);
      let s2, t2, symbolRow;
      if (transposed) {
        for (s2 = 0; s2 < symbolHeight; s2++) {
          row = bitmap[offsetS + s2];
          if (!row) {
            continue;
          }
          symbolRow = symbolBitmap[s2];
          const maxWidth = Math.min(width - offsetT, symbolWidth);
          switch (combinationOperator) {
            case 0:
              for (t2 = 0; t2 < maxWidth; t2++) {
                row[offsetT + t2] |= symbolRow[t2];
              }
              break;
            case 2:
              for (t2 = 0; t2 < maxWidth; t2++) {
                row[offsetT + t2] ^= symbolRow[t2];
              }
              break;
            default:
              throw new Jbig2Error(`operator ${combinationOperator} is not supported`);
          }
        }
      } else {
        for (t2 = 0; t2 < symbolHeight; t2++) {
          row = bitmap[offsetT + t2];
          if (!row) {
            continue;
          }
          symbolRow = symbolBitmap[t2];
          switch (combinationOperator) {
            case 0:
              for (s2 = 0; s2 < symbolWidth; s2++) {
                row[offsetS + s2] |= symbolRow[s2];
              }
              break;
            case 2:
              for (s2 = 0; s2 < symbolWidth; s2++) {
                row[offsetS + s2] ^= symbolRow[s2];
              }
              break;
            default:
              throw new Jbig2Error(`operator ${combinationOperator} is not supported`);
          }
        }
      }
      i++;
      const deltaS = huffman ? huffmanTables.tableDeltaS.decode(huffmanInput) : decodeInteger(contextCache, "IADS", decoder);
      if (deltaS === null) {
        break;
      }
      currentS += increment + deltaS + dsOffset;
    } while (true);
  }
  return bitmap;
}
function decodePatternDictionary(mmr, patternWidth, patternHeight, maxPatternIndex, template, decodingContext) {
  const at = [];
  if (!mmr) {
    at.push({
      x: -patternWidth,
      y: 0
    });
    if (template === 0) {
      at.push({
        x: -3,
        y: -1
      }, {
        x: 2,
        y: -2
      }, {
        x: -2,
        y: -2
      });
    }
  }
  const collectiveWidth = (maxPatternIndex + 1) * patternWidth;
  const collectiveBitmap = decodeBitmap(mmr, collectiveWidth, patternHeight, template, false, null, at, decodingContext);
  const patterns = [];
  for (let i = 0; i <= maxPatternIndex; i++) {
    const patternBitmap = [];
    const xMin = patternWidth * i;
    const xMax = xMin + patternWidth;
    for (let y = 0; y < patternHeight; y++) {
      patternBitmap.push(collectiveBitmap[y].subarray(xMin, xMax));
    }
    patterns.push(patternBitmap);
  }
  return patterns;
}
function decodeHalftoneRegion(mmr, patterns, template, regionWidth, regionHeight, defaultPixelValue, enableSkip, combinationOperator, gridWidth, gridHeight, gridOffsetX, gridOffsetY, gridVectorX, gridVectorY, decodingContext) {
  const skip = null;
  if (enableSkip) {
    throw new Jbig2Error("skip is not supported");
  }
  if (combinationOperator !== 0) {
    throw new Jbig2Error(`operator "${combinationOperator}" is not supported in halftone region`);
  }
  const regionBitmap = [];
  let i, j, row;
  for (i = 0; i < regionHeight; i++) {
    row = new Uint8Array(regionWidth);
    if (defaultPixelValue) {
      for (j = 0; j < regionWidth; j++) {
        row[j] = defaultPixelValue;
      }
    }
    regionBitmap.push(row);
  }
  const numberOfPatterns = patterns.length;
  const pattern0 = patterns[0];
  const patternWidth = pattern0[0].length,
    patternHeight = pattern0.length;
  const bitsPerValue = log2(numberOfPatterns);
  const at = [];
  if (!mmr) {
    at.push({
      x: template <= 1 ? 3 : 2,
      y: -1
    });
    if (template === 0) {
      at.push({
        x: -3,
        y: -1
      }, {
        x: 2,
        y: -2
      }, {
        x: -2,
        y: -2
      });
    }
  }
  const grayScaleBitPlanes = [];
  let mmrInput, bitmap;
  if (mmr) {
    mmrInput = new Reader(decodingContext.data, decodingContext.start, decodingContext.end);
  }
  for (i = bitsPerValue - 1; i >= 0; i--) {
    if (mmr) {
      bitmap = decodeMMRBitmap(mmrInput, gridWidth, gridHeight, true);
    } else {
      bitmap = decodeBitmap(false, gridWidth, gridHeight, template, false, skip, at, decodingContext);
    }
    grayScaleBitPlanes[i] = bitmap;
  }
  let mg, ng, bit, patternIndex, patternBitmap, x, y, patternRow, regionRow;
  for (mg = 0; mg < gridHeight; mg++) {
    for (ng = 0; ng < gridWidth; ng++) {
      bit = 0;
      patternIndex = 0;
      for (j = bitsPerValue - 1; j >= 0; j--) {
        bit ^= grayScaleBitPlanes[j][mg][ng];
        patternIndex |= bit << j;
      }
      patternBitmap = patterns[patternIndex];
      x = gridOffsetX + mg * gridVectorY + ng * gridVectorX >> 8;
      y = gridOffsetY + mg * gridVectorX - ng * gridVectorY >> 8;
      if (x >= 0 && x + patternWidth <= regionWidth && y >= 0 && y + patternHeight <= regionHeight) {
        for (i = 0; i < patternHeight; i++) {
          regionRow = regionBitmap[y + i];
          patternRow = patternBitmap[i];
          for (j = 0; j < patternWidth; j++) {
            regionRow[x + j] |= patternRow[j];
          }
        }
      } else {
        let regionX, regionY;
        for (i = 0; i < patternHeight; i++) {
          regionY = y + i;
          if (regionY < 0 || regionY >= regionHeight) {
            continue;
          }
          regionRow = regionBitmap[regionY];
          patternRow = patternBitmap[i];
          for (j = 0; j < patternWidth; j++) {
            regionX = x + j;
            if (regionX >= 0 && regionX < regionWidth) {
              regionRow[regionX] |= patternRow[j];
            }
          }
        }
      }
    }
  }
  return regionBitmap;
}
function readSegmentHeader(data, start) {
  const segmentHeader = {};
  segmentHeader.number = readUint32(data, start);
  const flags = data[start + 4];
  const segmentType = flags & 0x3f;
  if (!SegmentTypes[segmentType]) {
    throw new Jbig2Error("invalid segment type: " + segmentType);
  }
  segmentHeader.type = segmentType;
  segmentHeader.typeName = SegmentTypes[segmentType];
  segmentHeader.deferredNonRetain = !!(flags & 0x80);
  const pageAssociationFieldSize = !!(flags & 0x40);
  const referredFlags = data[start + 5];
  let referredToCount = referredFlags >> 5 & 7;
  const retainBits = [referredFlags & 31];
  let position = start + 6;
  if (referredFlags === 7) {
    referredToCount = readUint32(data, position - 1) & 0x1fffffff;
    position += 3;
    let bytes = referredToCount + 7 >> 3;
    retainBits[0] = data[position++];
    while (--bytes > 0) {
      retainBits.push(data[position++]);
    }
  } else if (referredFlags === 5 || referredFlags === 6) {
    throw new Jbig2Error("invalid referred-to flags");
  }
  segmentHeader.retainBits = retainBits;
  let referredToSegmentNumberSize = 4;
  if (segmentHeader.number <= 256) {
    referredToSegmentNumberSize = 1;
  } else if (segmentHeader.number <= 65536) {
    referredToSegmentNumberSize = 2;
  }
  const referredTo = [];
  let i, ii;
  for (i = 0; i < referredToCount; i++) {
    let number;
    if (referredToSegmentNumberSize === 1) {
      number = data[position];
    } else if (referredToSegmentNumberSize === 2) {
      number = readUint16(data, position);
    } else {
      number = readUint32(data, position);
    }
    referredTo.push(number);
    position += referredToSegmentNumberSize;
  }
  segmentHeader.referredTo = referredTo;
  if (!pageAssociationFieldSize) {
    segmentHeader.pageAssociation = data[position++];
  } else {
    segmentHeader.pageAssociation = readUint32(data, position);
    position += 4;
  }
  segmentHeader.length = readUint32(data, position);
  position += 4;
  if (segmentHeader.length === 0xffffffff) {
    if (segmentType === 38) {
      const genericRegionInfo = readRegionSegmentInformation(data, position);
      const genericRegionSegmentFlags = data[position + RegionSegmentInformationFieldLength];
      const genericRegionMmr = !!(genericRegionSegmentFlags & 1);
      const searchPatternLength = 6;
      const searchPattern = new Uint8Array(searchPatternLength);
      if (!genericRegionMmr) {
        searchPattern[0] = 0xff;
        searchPattern[1] = 0xac;
      }
      searchPattern[2] = genericRegionInfo.height >>> 24 & 0xff;
      searchPattern[3] = genericRegionInfo.height >> 16 & 0xff;
      searchPattern[4] = genericRegionInfo.height >> 8 & 0xff;
      searchPattern[5] = genericRegionInfo.height & 0xff;
      for (i = position, ii = data.length; i < ii; i++) {
        let j = 0;
        while (j < searchPatternLength && searchPattern[j] === data[i + j]) {
          j++;
        }
        if (j === searchPatternLength) {
          segmentHeader.length = i + searchPatternLength;
          break;
        }
      }
      if (segmentHeader.length === 0xffffffff) {
        throw new Jbig2Error("segment end was not found");
      }
    } else {
      throw new Jbig2Error("invalid unknown segment length");
    }
  }
  segmentHeader.headerEnd = position;
  return segmentHeader;
}
function readSegments(header, data, start, end) {
  const segments = [];
  let position = start;
  while (position < end) {
    const segmentHeader = readSegmentHeader(data, position);
    position = segmentHeader.headerEnd;
    const segment = {
      header: segmentHeader,
      data
    };
    if (!header.randomAccess) {
      segment.start = position;
      position += segmentHeader.length;
      segment.end = position;
    }
    segments.push(segment);
    if (segmentHeader.type === 51) {
      break;
    }
  }
  if (header.randomAccess) {
    for (let i = 0, ii = segments.length; i < ii; i++) {
      segments[i].start = position;
      position += segments[i].header.length;
      segments[i].end = position;
    }
  }
  return segments;
}
function readRegionSegmentInformation(data, start) {
  return {
    width: readUint32(data, start),
    height: readUint32(data, start + 4),
    x: readUint32(data, start + 8),
    y: readUint32(data, start + 12),
    combinationOperator: data[start + 16] & 7
  };
}
const RegionSegmentInformationFieldLength = 17;
function processSegment(segment, visitor) {
  const header = segment.header;
  const data = segment.data,
    end = segment.end;
  let position = segment.start;
  let args, at, i, atLength;
  switch (header.type) {
    case 0:
      const dictionary = {};
      const dictionaryFlags = readUint16(data, position);
      dictionary.huffman = !!(dictionaryFlags & 1);
      dictionary.refinement = !!(dictionaryFlags & 2);
      dictionary.huffmanDHSelector = dictionaryFlags >> 2 & 3;
      dictionary.huffmanDWSelector = dictionaryFlags >> 4 & 3;
      dictionary.bitmapSizeSelector = dictionaryFlags >> 6 & 1;
      dictionary.aggregationInstancesSelector = dictionaryFlags >> 7 & 1;
      dictionary.bitmapCodingContextUsed = !!(dictionaryFlags & 256);
      dictionary.bitmapCodingContextRetained = !!(dictionaryFlags & 512);
      dictionary.template = dictionaryFlags >> 10 & 3;
      dictionary.refinementTemplate = dictionaryFlags >> 12 & 1;
      position += 2;
      if (!dictionary.huffman) {
        atLength = dictionary.template === 0 ? 4 : 1;
        at = [];
        for (i = 0; i < atLength; i++) {
          at.push({
            x: readInt8(data, position),
            y: readInt8(data, position + 1)
          });
          position += 2;
        }
        dictionary.at = at;
      }
      if (dictionary.refinement && !dictionary.refinementTemplate) {
        at = [];
        for (i = 0; i < 2; i++) {
          at.push({
            x: readInt8(data, position),
            y: readInt8(data, position + 1)
          });
          position += 2;
        }
        dictionary.refinementAt = at;
      }
      dictionary.numberOfExportedSymbols = readUint32(data, position);
      position += 4;
      dictionary.numberOfNewSymbols = readUint32(data, position);
      position += 4;
      args = [dictionary, header.number, header.referredTo, data, position, end];
      break;
    case 6:
    case 7:
      const textRegion = {};
      textRegion.info = readRegionSegmentInformation(data, position);
      position += RegionSegmentInformationFieldLength;
      const textRegionSegmentFlags = readUint16(data, position);
      position += 2;
      textRegion.huffman = !!(textRegionSegmentFlags & 1);
      textRegion.refinement = !!(textRegionSegmentFlags & 2);
      textRegion.logStripSize = textRegionSegmentFlags >> 2 & 3;
      textRegion.stripSize = 1 << textRegion.logStripSize;
      textRegion.referenceCorner = textRegionSegmentFlags >> 4 & 3;
      textRegion.transposed = !!(textRegionSegmentFlags & 64);
      textRegion.combinationOperator = textRegionSegmentFlags >> 7 & 3;
      textRegion.defaultPixelValue = textRegionSegmentFlags >> 9 & 1;
      textRegion.dsOffset = textRegionSegmentFlags << 17 >> 27;
      textRegion.refinementTemplate = textRegionSegmentFlags >> 15 & 1;
      if (textRegion.huffman) {
        const textRegionHuffmanFlags = readUint16(data, position);
        position += 2;
        textRegion.huffmanFS = textRegionHuffmanFlags & 3;
        textRegion.huffmanDS = textRegionHuffmanFlags >> 2 & 3;
        textRegion.huffmanDT = textRegionHuffmanFlags >> 4 & 3;
        textRegion.huffmanRefinementDW = textRegionHuffmanFlags >> 6 & 3;
        textRegion.huffmanRefinementDH = textRegionHuffmanFlags >> 8 & 3;
        textRegion.huffmanRefinementDX = textRegionHuffmanFlags >> 10 & 3;
        textRegion.huffmanRefinementDY = textRegionHuffmanFlags >> 12 & 3;
        textRegion.huffmanRefinementSizeSelector = !!(textRegionHuffmanFlags & 0x4000);
      }
      if (textRegion.refinement && !textRegion.refinementTemplate) {
        at = [];
        for (i = 0; i < 2; i++) {
          at.push({
            x: readInt8(data, position),
            y: readInt8(data, position + 1)
          });
          position += 2;
        }
        textRegion.refinementAt = at;
      }
      textRegion.numberOfSymbolInstances = readUint32(data, position);
      position += 4;
      args = [textRegion, header.referredTo, data, position, end];
      break;
    case 16:
      const patternDictionary = {};
      const patternDictionaryFlags = data[position++];
      patternDictionary.mmr = !!(patternDictionaryFlags & 1);
      patternDictionary.template = patternDictionaryFlags >> 1 & 3;
      patternDictionary.patternWidth = data[position++];
      patternDictionary.patternHeight = data[position++];
      patternDictionary.maxPatternIndex = readUint32(data, position);
      position += 4;
      args = [patternDictionary, header.number, data, position, end];
      break;
    case 22:
    case 23:
      const halftoneRegion = {};
      halftoneRegion.info = readRegionSegmentInformation(data, position);
      position += RegionSegmentInformationFieldLength;
      const halftoneRegionFlags = data[position++];
      halftoneRegion.mmr = !!(halftoneRegionFlags & 1);
      halftoneRegion.template = halftoneRegionFlags >> 1 & 3;
      halftoneRegion.enableSkip = !!(halftoneRegionFlags & 8);
      halftoneRegion.combinationOperator = halftoneRegionFlags >> 4 & 7;
      halftoneRegion.defaultPixelValue = halftoneRegionFlags >> 7 & 1;
      halftoneRegion.gridWidth = readUint32(data, position);
      position += 4;
      halftoneRegion.gridHeight = readUint32(data, position);
      position += 4;
      halftoneRegion.gridOffsetX = readUint32(data, position) & 0xffffffff;
      position += 4;
      halftoneRegion.gridOffsetY = readUint32(data, position) & 0xffffffff;
      position += 4;
      halftoneRegion.gridVectorX = readUint16(data, position);
      position += 2;
      halftoneRegion.gridVectorY = readUint16(data, position);
      position += 2;
      args = [halftoneRegion, header.referredTo, data, position, end];
      break;
    case 38:
    case 39:
      const genericRegion = {};
      genericRegion.info = readRegionSegmentInformation(data, position);
      position += RegionSegmentInformationFieldLength;
      const genericRegionSegmentFlags = data[position++];
      genericRegion.mmr = !!(genericRegionSegmentFlags & 1);
      genericRegion.template = genericRegionSegmentFlags >> 1 & 3;
      genericRegion.prediction = !!(genericRegionSegmentFlags & 8);
      if (!genericRegion.mmr) {
        atLength = genericRegion.template === 0 ? 4 : 1;
        at = [];
        for (i = 0; i < atLength; i++) {
          at.push({
            x: readInt8(data, position),
            y: readInt8(data, position + 1)
          });
          position += 2;
        }
        genericRegion.at = at;
      }
      args = [genericRegion, data, position, end];
      break;
    case 48:
      const pageInfo = {
        width: readUint32(data, position),
        height: readUint32(data, position + 4),
        resolutionX: readUint32(data, position + 8),
        resolutionY: readUint32(data, position + 12)
      };
      if (pageInfo.height === 0xffffffff) {
        delete pageInfo.height;
      }
      const pageSegmentFlags = data[position + 16];
      readUint16(data, position + 17);
      pageInfo.lossless = !!(pageSegmentFlags & 1);
      pageInfo.refinement = !!(pageSegmentFlags & 2);
      pageInfo.defaultPixelValue = pageSegmentFlags >> 2 & 1;
      pageInfo.combinationOperator = pageSegmentFlags >> 3 & 3;
      pageInfo.requiresBuffer = !!(pageSegmentFlags & 32);
      pageInfo.combinationOperatorOverride = !!(pageSegmentFlags & 64);
      args = [pageInfo];
      break;
    case 49:
      break;
    case 50:
      break;
    case 51:
      break;
    case 53:
      args = [header.number, data, position, end];
      break;
    case 62:
      break;
    default:
      throw new Jbig2Error(`segment type ${header.typeName}(${header.type}) is not implemented`);
  }
  const callbackName = "on" + header.typeName;
  if (callbackName in visitor) {
    visitor[callbackName].apply(visitor, args);
  }
}
function processSegments(segments, visitor) {
  for (let i = 0, ii = segments.length; i < ii; i++) {
    processSegment(segments[i], visitor);
  }
}
function parseJbig2Chunks(chunks) {
  const visitor = new SimpleSegmentVisitor();
  for (let i = 0, ii = chunks.length; i < ii; i++) {
    const chunk = chunks[i];
    const segments = readSegments({}, chunk.data, chunk.start, chunk.end);
    processSegments(segments, visitor);
  }
  return visitor.buffer;
}
function parseJbig2(data) {
  const end = data.length;
  let position = 0;
  if (data[position] !== 0x97 || data[position + 1] !== 0x4a || data[position + 2] !== 0x42 || data[position + 3] !== 0x32 || data[position + 4] !== 0x0d || data[position + 5] !== 0x0a || data[position + 6] !== 0x1a || data[position + 7] !== 0x0a) {
    throw new Jbig2Error("parseJbig2 - invalid header.");
  }
  const header = Object.create(null);
  position += 8;
  const flags = data[position++];
  header.randomAccess = !(flags & 1);
  if (!(flags & 2)) {
    header.numberOfPages = readUint32(data, position);
    position += 4;
  }
  const segments = readSegments(header, data, position, end);
  const visitor = new SimpleSegmentVisitor();
  processSegments(segments, visitor);
  const {
    width,
    height
  } = visitor.currentPageInfo;
  const bitPacked = visitor.buffer;
  const imgData = new Uint8ClampedArray(width * height);
  let q = 0,
    k = 0;
  for (let i = 0; i < height; i++) {
    let mask = 0,
      buffer;
    for (let j = 0; j < width; j++) {
      if (!mask) {
        mask = 128;
        buffer = bitPacked[k++];
      }
      imgData[q++] = buffer & mask ? 0 : 255;
      mask >>= 1;
    }
  }
  return {
    imgData,
    width,
    height
  };
}
class SimpleSegmentVisitor {
  onPageInformation(info) {
    this.currentPageInfo = info;
    const rowSize = info.width + 7 >> 3;
    const buffer = new Uint8ClampedArray(rowSize * info.height);
    if (info.defaultPixelValue) {
      buffer.fill(0xff);
    }
    this.buffer = buffer;
  }
  drawBitmap(regionInfo, bitmap) {
    const pageInfo = this.currentPageInfo;
    const width = regionInfo.width,
      height = regionInfo.height;
    const rowSize = pageInfo.width + 7 >> 3;
    const combinationOperator = pageInfo.combinationOperatorOverride ? regionInfo.combinationOperator : pageInfo.combinationOperator;
    const buffer = this.buffer;
    const mask0 = 128 >> (regionInfo.x & 7);
    let offset0 = regionInfo.y * rowSize + (regionInfo.x >> 3);
    let i, j, mask, offset;
    switch (combinationOperator) {
      case 0:
        for (i = 0; i < height; i++) {
          mask = mask0;
          offset = offset0;
          for (j = 0; j < width; j++) {
            if (bitmap[i][j]) {
              buffer[offset] |= mask;
            }
            mask >>= 1;
            if (!mask) {
              mask = 128;
              offset++;
            }
          }
          offset0 += rowSize;
        }
        break;
      case 2:
        for (i = 0; i < height; i++) {
          mask = mask0;
          offset = offset0;
          for (j = 0; j < width; j++) {
            if (bitmap[i][j]) {
              buffer[offset] ^= mask;
            }
            mask >>= 1;
            if (!mask) {
              mask = 128;
              offset++;
            }
          }
          offset0 += rowSize;
        }
        break;
      default:
        throw new Jbig2Error(`operator ${combinationOperator} is not supported`);
    }
  }
  onImmediateGenericRegion(region, data, start, end) {
    const regionInfo = region.info;
    const decodingContext = new DecodingContext(data, start, end);
    const bitmap = decodeBitmap(region.mmr, regionInfo.width, regionInfo.height, region.template, region.prediction, null, region.at, decodingContext);
    this.drawBitmap(regionInfo, bitmap);
  }
  onImmediateLosslessGenericRegion() {
    this.onImmediateGenericRegion(...arguments);
  }
  onSymbolDictionary(dictionary, currentSegment, referredSegments, data, start, end) {
    let huffmanTables, huffmanInput;
    if (dictionary.huffman) {
      huffmanTables = getSymbolDictionaryHuffmanTables(dictionary, referredSegments, this.customTables);
      huffmanInput = new Reader(data, start, end);
    }
    let symbols = this.symbols;
    if (!symbols) {
      this.symbols = symbols = {};
    }
    const inputSymbols = [];
    for (const referredSegment of referredSegments) {
      const referredSymbols = symbols[referredSegment];
      if (referredSymbols) {
        inputSymbols.push(...referredSymbols);
      }
    }
    const decodingContext = new DecodingContext(data, start, end);
    symbols[currentSegment] = decodeSymbolDictionary(dictionary.huffman, dictionary.refinement, inputSymbols, dictionary.numberOfNewSymbols, dictionary.numberOfExportedSymbols, huffmanTables, dictionary.template, dictionary.at, dictionary.refinementTemplate, dictionary.refinementAt, decodingContext, huffmanInput);
  }
  onImmediateTextRegion(region, referredSegments, data, start, end) {
    const regionInfo = region.info;
    let huffmanTables, huffmanInput;
    const symbols = this.symbols;
    const inputSymbols = [];
    for (const referredSegment of referredSegments) {
      const referredSymbols = symbols[referredSegment];
      if (referredSymbols) {
        inputSymbols.push(...referredSymbols);
      }
    }
    const symbolCodeLength = log2(inputSymbols.length);
    if (region.huffman) {
      huffmanInput = new Reader(data, start, end);
      huffmanTables = getTextRegionHuffmanTables(region, referredSegments, this.customTables, inputSymbols.length, huffmanInput);
    }
    const decodingContext = new DecodingContext(data, start, end);
    const bitmap = decodeTextRegion(region.huffman, region.refinement, regionInfo.width, regionInfo.height, region.defaultPixelValue, region.numberOfSymbolInstances, region.stripSize, inputSymbols, symbolCodeLength, region.transposed, region.dsOffset, region.referenceCorner, region.combinationOperator, huffmanTables, region.refinementTemplate, region.refinementAt, decodingContext, region.logStripSize, huffmanInput);
    this.drawBitmap(regionInfo, bitmap);
  }
  onImmediateLosslessTextRegion() {
    this.onImmediateTextRegion(...arguments);
  }
  onPatternDictionary(dictionary, currentSegment, data, start, end) {
    let patterns = this.patterns;
    if (!patterns) {
      this.patterns = patterns = {};
    }
    const decodingContext = new DecodingContext(data, start, end);
    patterns[currentSegment] = decodePatternDictionary(dictionary.mmr, dictionary.patternWidth, dictionary.patternHeight, dictionary.maxPatternIndex, dictionary.template, decodingContext);
  }
  onImmediateHalftoneRegion(region, referredSegments, data, start, end) {
    const patterns = this.patterns[referredSegments[0]];
    const regionInfo = region.info;
    const decodingContext = new DecodingContext(data, start, end);
    const bitmap = decodeHalftoneRegion(region.mmr, patterns, region.template, regionInfo.width, regionInfo.height, region.defaultPixelValue, region.enableSkip, region.combinationOperator, region.gridWidth, region.gridHeight, region.gridOffsetX, region.gridOffsetY, region.gridVectorX, region.gridVectorY, decodingContext);
    this.drawBitmap(regionInfo, bitmap);
  }
  onImmediateLosslessHalftoneRegion() {
    this.onImmediateHalftoneRegion(...arguments);
  }
  onTables(currentSegment, data, start, end) {
    let customTables = this.customTables;
    if (!customTables) {
      this.customTables = customTables = {};
    }
    customTables[currentSegment] = decodeTablesSegment(data, start, end);
  }
}
class HuffmanLine {
  constructor(lineData) {
    if (lineData.length === 2) {
      this.isOOB = true;
      this.rangeLow = 0;
      this.prefixLength = lineData[0];
      this.rangeLength = 0;
      this.prefixCode = lineData[1];
      this.isLowerRange = false;
    } else {
      this.isOOB = false;
      this.rangeLow = lineData[0];
      this.prefixLength = lineData[1];
      this.rangeLength = lineData[2];
      this.prefixCode = lineData[3];
      this.isLowerRange = lineData[4] === "lower";
    }
  }
}
class HuffmanTreeNode {
  constructor(line) {
    this.children = [];
    if (line) {
      this.isLeaf = true;
      this.rangeLength = line.rangeLength;
      this.rangeLow = line.rangeLow;
      this.isLowerRange = line.isLowerRange;
      this.isOOB = line.isOOB;
    } else {
      this.isLeaf = false;
    }
  }
  buildTree(line, shift) {
    const bit = line.prefixCode >> shift & 1;
    if (shift <= 0) {
      this.children[bit] = new HuffmanTreeNode(line);
    } else {
      let node = this.children[bit];
      if (!node) {
        this.children[bit] = node = new HuffmanTreeNode(null);
      }
      node.buildTree(line, shift - 1);
    }
  }
  decodeNode(reader) {
    if (this.isLeaf) {
      if (this.isOOB) {
        return null;
      }
      const htOffset = reader.readBits(this.rangeLength);
      return this.rangeLow + (this.isLowerRange ? -htOffset : htOffset);
    }
    const node = this.children[reader.readBit()];
    if (!node) {
      throw new Jbig2Error("invalid Huffman data");
    }
    return node.decodeNode(reader);
  }
}
class HuffmanTable {
  constructor(lines, prefixCodesDone) {
    if (!prefixCodesDone) {
      this.assignPrefixCodes(lines);
    }
    this.rootNode = new HuffmanTreeNode(null);
    for (let i = 0, ii = lines.length; i < ii; i++) {
      const line = lines[i];
      if (line.prefixLength > 0) {
        this.rootNode.buildTree(line, line.prefixLength - 1);
      }
    }
  }
  decode(reader) {
    return this.rootNode.decodeNode(reader);
  }
  assignPrefixCodes(lines) {
    const linesLength = lines.length;
    let prefixLengthMax = 0;
    for (let i = 0; i < linesLength; i++) {
      prefixLengthMax = Math.max(prefixLengthMax, lines[i].prefixLength);
    }
    const histogram = new Uint32Array(prefixLengthMax + 1);
    for (let i = 0; i < linesLength; i++) {
      histogram[lines[i].prefixLength]++;
    }
    let currentLength = 1,
      firstCode = 0,
      currentCode,
      currentTemp,
      line;
    histogram[0] = 0;
    while (currentLength <= prefixLengthMax) {
      firstCode = firstCode + histogram[currentLength - 1] << 1;
      currentCode = firstCode;
      currentTemp = 0;
      while (currentTemp < linesLength) {
        line = lines[currentTemp];
        if (line.prefixLength === currentLength) {
          line.prefixCode = currentCode;
          currentCode++;
        }
        currentTemp++;
      }
      currentLength++;
    }
  }
}
function decodeTablesSegment(data, start, end) {
  const flags = data[start];
  const lowestValue = readUint32(data, start + 1) & 0xffffffff;
  const highestValue = readUint32(data, start + 5) & 0xffffffff;
  const reader = new Reader(data, start + 9, end);
  const prefixSizeBits = (flags >> 1 & 7) + 1;
  const rangeSizeBits = (flags >> 4 & 7) + 1;
  const lines = [];
  let prefixLength,
    rangeLength,
    currentRangeLow = lowestValue;
  do {
    prefixLength = reader.readBits(prefixSizeBits);
    rangeLength = reader.readBits(rangeSizeBits);
    lines.push(new HuffmanLine([currentRangeLow, prefixLength, rangeLength, 0]));
    currentRangeLow += 1 << rangeLength;
  } while (currentRangeLow < highestValue);
  prefixLength = reader.readBits(prefixSizeBits);
  lines.push(new HuffmanLine([lowestValue - 1, prefixLength, 32, 0, "lower"]));
  prefixLength = reader.readBits(prefixSizeBits);
  lines.push(new HuffmanLine([highestValue, prefixLength, 32, 0]));
  if (flags & 1) {
    prefixLength = reader.readBits(prefixSizeBits);
    lines.push(new HuffmanLine([prefixLength, 0]));
  }
  return new HuffmanTable(lines, false);
}
const standardTablesCache = {};
function getStandardTable(number) {
  let table = standardTablesCache[number];
  if (table) {
    return table;
  }
  let lines;
  switch (number) {
    case 1:
      lines = [[0, 1, 4, 0x0], [16, 2, 8, 0x2], [272, 3, 16, 0x6], [65808, 3, 32, 0x7]];
      break;
    case 2:
      lines = [[0, 1, 0, 0x0], [1, 2, 0, 0x2], [2, 3, 0, 0x6], [3, 4, 3, 0xe], [11, 5, 6, 0x1e], [75, 6, 32, 0x3e], [6, 0x3f]];
      break;
    case 3:
      lines = [[-256, 8, 8, 0xfe], [0, 1, 0, 0x0], [1, 2, 0, 0x2], [2, 3, 0, 0x6], [3, 4, 3, 0xe], [11, 5, 6, 0x1e], [-257, 8, 32, 0xff, "lower"], [75, 7, 32, 0x7e], [6, 0x3e]];
      break;
    case 4:
      lines = [[1, 1, 0, 0x0], [2, 2, 0, 0x2], [3, 3, 0, 0x6], [4, 4, 3, 0xe], [12, 5, 6, 0x1e], [76, 5, 32, 0x1f]];
      break;
    case 5:
      lines = [[-255, 7, 8, 0x7e], [1, 1, 0, 0x0], [2, 2, 0, 0x2], [3, 3, 0, 0x6], [4, 4, 3, 0xe], [12, 5, 6, 0x1e], [-256, 7, 32, 0x7f, "lower"], [76, 6, 32, 0x3e]];
      break;
    case 6:
      lines = [[-2048, 5, 10, 0x1c], [-1024, 4, 9, 0x8], [-512, 4, 8, 0x9], [-256, 4, 7, 0xa], [-128, 5, 6, 0x1d], [-64, 5, 5, 0x1e], [-32, 4, 5, 0xb], [0, 2, 7, 0x0], [128, 3, 7, 0x2], [256, 3, 8, 0x3], [512, 4, 9, 0xc], [1024, 4, 10, 0xd], [-2049, 6, 32, 0x3e, "lower"], [2048, 6, 32, 0x3f]];
      break;
    case 7:
      lines = [[-1024, 4, 9, 0x8], [-512, 3, 8, 0x0], [-256, 4, 7, 0x9], [-128, 5, 6, 0x1a], [-64, 5, 5, 0x1b], [-32, 4, 5, 0xa], [0, 4, 5, 0xb], [32, 5, 5, 0x1c], [64, 5, 6, 0x1d], [128, 4, 7, 0xc], [256, 3, 8, 0x1], [512, 3, 9, 0x2], [1024, 3, 10, 0x3], [-1025, 5, 32, 0x1e, "lower"], [2048, 5, 32, 0x1f]];
      break;
    case 8:
      lines = [[-15, 8, 3, 0xfc], [-7, 9, 1, 0x1fc], [-5, 8, 1, 0xfd], [-3, 9, 0, 0x1fd], [-2, 7, 0, 0x7c], [-1, 4, 0, 0xa], [0, 2, 1, 0x0], [2, 5, 0, 0x1a], [3, 6, 0, 0x3a], [4, 3, 4, 0x4], [20, 6, 1, 0x3b], [22, 4, 4, 0xb], [38, 4, 5, 0xc], [70, 5, 6, 0x1b], [134, 5, 7, 0x1c], [262, 6, 7, 0x3c], [390, 7, 8, 0x7d], [646, 6, 10, 0x3d], [-16, 9, 32, 0x1fe, "lower"], [1670, 9, 32, 0x1ff], [2, 0x1]];
      break;
    case 9:
      lines = [[-31, 8, 4, 0xfc], [-15, 9, 2, 0x1fc], [-11, 8, 2, 0xfd], [-7, 9, 1, 0x1fd], [-5, 7, 1, 0x7c], [-3, 4, 1, 0xa], [-1, 3, 1, 0x2], [1, 3, 1, 0x3], [3, 5, 1, 0x1a], [5, 6, 1, 0x3a], [7, 3, 5, 0x4], [39, 6, 2, 0x3b], [43, 4, 5, 0xb], [75, 4, 6, 0xc], [139, 5, 7, 0x1b], [267, 5, 8, 0x1c], [523, 6, 8, 0x3c], [779, 7, 9, 0x7d], [1291, 6, 11, 0x3d], [-32, 9, 32, 0x1fe, "lower"], [3339, 9, 32, 0x1ff], [2, 0x0]];
      break;
    case 10:
      lines = [[-21, 7, 4, 0x7a], [-5, 8, 0, 0xfc], [-4, 7, 0, 0x7b], [-3, 5, 0, 0x18], [-2, 2, 2, 0x0], [2, 5, 0, 0x19], [3, 6, 0, 0x36], [4, 7, 0, 0x7c], [5, 8, 0, 0xfd], [6, 2, 6, 0x1], [70, 5, 5, 0x1a], [102, 6, 5, 0x37], [134, 6, 6, 0x38], [198, 6, 7, 0x39], [326, 6, 8, 0x3a], [582, 6, 9, 0x3b], [1094, 6, 10, 0x3c], [2118, 7, 11, 0x7d], [-22, 8, 32, 0xfe, "lower"], [4166, 8, 32, 0xff], [2, 0x2]];
      break;
    case 11:
      lines = [[1, 1, 0, 0x0], [2, 2, 1, 0x2], [4, 4, 0, 0xc], [5, 4, 1, 0xd], [7, 5, 1, 0x1c], [9, 5, 2, 0x1d], [13, 6, 2, 0x3c], [17, 7, 2, 0x7a], [21, 7, 3, 0x7b], [29, 7, 4, 0x7c], [45, 7, 5, 0x7d], [77, 7, 6, 0x7e], [141, 7, 32, 0x7f]];
      break;
    case 12:
      lines = [[1, 1, 0, 0x0], [2, 2, 0, 0x2], [3, 3, 1, 0x6], [5, 5, 0, 0x1c], [6, 5, 1, 0x1d], [8, 6, 1, 0x3c], [10, 7, 0, 0x7a], [11, 7, 1, 0x7b], [13, 7, 2, 0x7c], [17, 7, 3, 0x7d], [25, 7, 4, 0x7e], [41, 8, 5, 0xfe], [73, 8, 32, 0xff]];
      break;
    case 13:
      lines = [[1, 1, 0, 0x0], [2, 3, 0, 0x4], [3, 4, 0, 0xc], [4, 5, 0, 0x1c], [5, 4, 1, 0xd], [7, 3, 3, 0x5], [15, 6, 1, 0x3a], [17, 6, 2, 0x3b], [21, 6, 3, 0x3c], [29, 6, 4, 0x3d], [45, 6, 5, 0x3e], [77, 7, 6, 0x7e], [141, 7, 32, 0x7f]];
      break;
    case 14:
      lines = [[-2, 3, 0, 0x4], [-1, 3, 0, 0x5], [0, 1, 0, 0x0], [1, 3, 0, 0x6], [2, 3, 0, 0x7]];
      break;
    case 15:
      lines = [[-24, 7, 4, 0x7c], [-8, 6, 2, 0x3c], [-4, 5, 1, 0x1c], [-2, 4, 0, 0xc], [-1, 3, 0, 0x4], [0, 1, 0, 0x0], [1, 3, 0, 0x5], [2, 4, 0, 0xd], [3, 5, 1, 0x1d], [5, 6, 2, 0x3d], [9, 7, 4, 0x7d], [-25, 7, 32, 0x7e, "lower"], [25, 7, 32, 0x7f]];
      break;
    default:
      throw new Jbig2Error(`standard table B.${number} does not exist`);
  }
  for (let i = 0, ii = lines.length; i < ii; i++) {
    lines[i] = new HuffmanLine(lines[i]);
  }
  table = new HuffmanTable(lines, true);
  standardTablesCache[number] = table;
  return table;
}
class Reader {
  constructor(data, start, end) {
    this.data = data;
    this.start = start;
    this.end = end;
    this.position = start;
    this.shift = -1;
    this.currentByte = 0;
  }
  readBit() {
    if (this.shift < 0) {
      if (this.position >= this.end) {
        throw new Jbig2Error("end of data while reading bit");
      }
      this.currentByte = this.data[this.position++];
      this.shift = 7;
    }
    const bit = this.currentByte >> this.shift & 1;
    this.shift--;
    return bit;
  }
  readBits(numBits) {
    let result = 0,
      i;
    for (i = numBits - 1; i >= 0; i--) {
      result |= this.readBit() << i;
    }
    return result;
  }
  byteAlign() {
    this.shift = -1;
  }
  next() {
    if (this.position >= this.end) {
      return -1;
    }
    return this.data[this.position++];
  }
}
function getCustomHuffmanTable(index, referredTo, customTables) {
  let currentIndex = 0;
  for (let i = 0, ii = referredTo.length; i < ii; i++) {
    const table = customTables[referredTo[i]];
    if (table) {
      if (index === currentIndex) {
        return table;
      }
      currentIndex++;
    }
  }
  throw new Jbig2Error("can't find custom Huffman table");
}
function getTextRegionHuffmanTables(textRegion, referredTo, customTables, numberOfSymbols, reader) {
  const codes = [];
  for (let i = 0; i <= 34; i++) {
    const codeLength = reader.readBits(4);
    codes.push(new HuffmanLine([i, codeLength, 0, 0]));
  }
  const runCodesTable = new HuffmanTable(codes, false);
  codes.length = 0;
  for (let i = 0; i < numberOfSymbols;) {
    const codeLength = runCodesTable.decode(reader);
    if (codeLength >= 32) {
      let repeatedLength, numberOfRepeats, j;
      switch (codeLength) {
        case 32:
          if (i === 0) {
            throw new Jbig2Error("no previous value in symbol ID table");
          }
          numberOfRepeats = reader.readBits(2) + 3;
          repeatedLength = codes[i - 1].prefixLength;
          break;
        case 33:
          numberOfRepeats = reader.readBits(3) + 3;
          repeatedLength = 0;
          break;
        case 34:
          numberOfRepeats = reader.readBits(7) + 11;
          repeatedLength = 0;
          break;
        default:
          throw new Jbig2Error("invalid code length in symbol ID table");
      }
      for (j = 0; j < numberOfRepeats; j++) {
        codes.push(new HuffmanLine([i, repeatedLength, 0, 0]));
        i++;
      }
    } else {
      codes.push(new HuffmanLine([i, codeLength, 0, 0]));
      i++;
    }
  }
  reader.byteAlign();
  const symbolIDTable = new HuffmanTable(codes, false);
  let customIndex = 0,
    tableFirstS,
    tableDeltaS,
    tableDeltaT;
  switch (textRegion.huffmanFS) {
    case 0:
    case 1:
      tableFirstS = getStandardTable(textRegion.huffmanFS + 6);
      break;
    case 3:
      tableFirstS = getCustomHuffmanTable(customIndex, referredTo, customTables);
      customIndex++;
      break;
    default:
      throw new Jbig2Error("invalid Huffman FS selector");
  }
  switch (textRegion.huffmanDS) {
    case 0:
    case 1:
    case 2:
      tableDeltaS = getStandardTable(textRegion.huffmanDS + 8);
      break;
    case 3:
      tableDeltaS = getCustomHuffmanTable(customIndex, referredTo, customTables);
      customIndex++;
      break;
    default:
      throw new Jbig2Error("invalid Huffman DS selector");
  }
  switch (textRegion.huffmanDT) {
    case 0:
    case 1:
    case 2:
      tableDeltaT = getStandardTable(textRegion.huffmanDT + 11);
      break;
    case 3:
      tableDeltaT = getCustomHuffmanTable(customIndex, referredTo, customTables);
      customIndex++;
      break;
    default:
      throw new Jbig2Error("invalid Huffman DT selector");
  }
  if (textRegion.refinement) {
    throw new Jbig2Error("refinement with Huffman is not supported");
  }
  return {
    symbolIDTable,
    tableFirstS,
    tableDeltaS,
    tableDeltaT
  };
}
function getSymbolDictionaryHuffmanTables(dictionary, referredTo, customTables) {
  let customIndex = 0,
    tableDeltaHeight,
    tableDeltaWidth;
  switch (dictionary.huffmanDHSelector) {
    case 0:
    case 1:
      tableDeltaHeight = getStandardTable(dictionary.huffmanDHSelector + 4);
      break;
    case 3:
      tableDeltaHeight = getCustomHuffmanTable(customIndex, referredTo, customTables);
      customIndex++;
      break;
    default:
      throw new Jbig2Error("invalid Huffman DH selector");
  }
  switch (dictionary.huffmanDWSelector) {
    case 0:
    case 1:
      tableDeltaWidth = getStandardTable(dictionary.huffmanDWSelector + 2);
      break;
    case 3:
      tableDeltaWidth = getCustomHuffmanTable(customIndex, referredTo, customTables);
      customIndex++;
      break;
    default:
      throw new Jbig2Error("invalid Huffman DW selector");
  }
  let tableBitmapSize, tableAggregateInstances;
  if (dictionary.bitmapSizeSelector) {
    tableBitmapSize = getCustomHuffmanTable(customIndex, referredTo, customTables);
    customIndex++;
  } else {
    tableBitmapSize = getStandardTable(1);
  }
  if (dictionary.aggregationInstancesSelector) {
    tableAggregateInstances = getCustomHuffmanTable(customIndex, referredTo, customTables);
  } else {
    tableAggregateInstances = getStandardTable(1);
  }
  return {
    tableDeltaHeight,
    tableDeltaWidth,
    tableBitmapSize,
    tableAggregateInstances
  };
}
function readUncompressedBitmap(reader, width, height) {
  const bitmap = [];
  for (let y = 0; y < height; y++) {
    const row = new Uint8Array(width);
    bitmap.push(row);
    for (let x = 0; x < width; x++) {
      row[x] = reader.readBit();
    }
    reader.byteAlign();
  }
  return bitmap;
}
function decodeMMRBitmap(input, width, height, endOfBlock) {
  const params = {
    K: -1,
    Columns: width,
    Rows: height,
    BlackIs1: true,
    EndOfBlock: endOfBlock
  };
  const decoder = new CCITTFaxDecoder(input, params);
  const bitmap = [];
  let currentByte,
    eof = false;
  for (let y = 0; y < height; y++) {
    const row = new Uint8Array(width);
    bitmap.push(row);
    let shift = -1;
    for (let x = 0; x < width; x++) {
      if (shift < 0) {
        currentByte = decoder.readNextChar();
        if (currentByte === -1) {
          currentByte = 0;
          eof = true;
        }
        shift = 7;
      }
      row[x] = currentByte >> shift & 1;
      shift--;
    }
  }
  if (endOfBlock && !eof) {
    const lookForEOFLimit = 5;
    for (let i = 0; i < lookForEOFLimit; i++) {
      if (decoder.readNextChar() === -1) {
        break;
      }
    }
  }
  return bitmap;
}
class Jbig2Image {
  parseChunks(chunks) {
    return parseJbig2Chunks(chunks);
  }
  parse(data) {
    const {
      imgData,
      width,
      height
    } = parseJbig2(data);
    this.width = width;
    this.height = height;
    return imgData;
  }
}

;// ./src/shared/image_utils.js







function convertToRGBA(params) {
  switch (params.kind) {
    case ImageKind.GRAYSCALE_1BPP:
      return convertBlackAndWhiteToRGBA(params);
    case ImageKind.RGB_24BPP:
      return convertRGBToRGBA(params);
  }
  return null;
}
function convertBlackAndWhiteToRGBA({
  src,
  srcPos = 0,
  dest,
  width,
  height,
  nonBlackColor = 0xffffffff,
  inverseDecode = false
}) {
  const black = FeatureTest.isLittleEndian ? 0xff000000 : 0x000000ff;
  const [zeroMapping, oneMapping] = inverseDecode ? [nonBlackColor, black] : [black, nonBlackColor];
  const widthInSource = width >> 3;
  const widthRemainder = width & 7;
  const srcLength = src.length;
  dest = new Uint32Array(dest.buffer);
  let destPos = 0;
  for (let i = 0; i < height; i++) {
    for (const max = srcPos + widthInSource; srcPos < max; srcPos++) {
      const elem = srcPos < srcLength ? src[srcPos] : 255;
      dest[destPos++] = elem & 0b10000000 ? oneMapping : zeroMapping;
      dest[destPos++] = elem & 0b1000000 ? oneMapping : zeroMapping;
      dest[destPos++] = elem & 0b100000 ? oneMapping : zeroMapping;
      dest[destPos++] = elem & 0b10000 ? oneMapping : zeroMapping;
      dest[destPos++] = elem & 0b1000 ? oneMapping : zeroMapping;
      dest[destPos++] = elem & 0b100 ? oneMapping : zeroMapping;
      dest[destPos++] = elem & 0b10 ? oneMapping : zeroMapping;
      dest[destPos++] = elem & 0b1 ? oneMapping : zeroMapping;
    }
    if (widthRemainder === 0) {
      continue;
    }
    const elem = srcPos < srcLength ? src[srcPos++] : 255;
    for (let j = 0; j < widthRemainder; j++) {
      dest[destPos++] = elem & 1 << 7 - j ? oneMapping : zeroMapping;
    }
  }
  return {
    srcPos,
    destPos
  };
}
function convertRGBToRGBA({
  src,
  srcPos = 0,
  dest,
  destPos = 0,
  width,
  height
}) {
  let i = 0;
  const len = width * height * 3;
  const len32 = len >> 2;
  const src32 = new Uint32Array(src.buffer, srcPos, len32);
  if (FeatureTest.isLittleEndian) {
    for (; i < len32 - 2; i += 3, destPos += 4) {
      const s1 = src32[i];
      const s2 = src32[i + 1];
      const s3 = src32[i + 2];
      dest[destPos] = s1 | 0xff000000;
      dest[destPos + 1] = s1 >>> 24 | s2 << 8 | 0xff000000;
      dest[destPos + 2] = s2 >>> 16 | s3 << 16 | 0xff000000;
      dest[destPos + 3] = s3 >>> 8 | 0xff000000;
    }
    for (let j = i * 4, jj = srcPos + len; j < jj; j += 3) {
      dest[destPos++] = src[j] | src[j + 1] << 8 | src[j + 2] << 16 | 0xff000000;
    }
  } else {
    for (; i < len32 - 2; i += 3, destPos += 4) {
      const s1 = src32[i];
      const s2 = src32[i + 1];
      const s3 = src32[i + 2];
      dest[destPos] = s1 | 0xff;
      dest[destPos + 1] = s1 << 24 | s2 >>> 8 | 0xff;
      dest[destPos + 2] = s2 << 16 | s3 >>> 16 | 0xff;
      dest[destPos + 3] = s3 << 8 | 0xff;
    }
    for (let j = i * 4, jj = srcPos + len; j < jj; j += 3) {
      dest[destPos++] = src[j] << 24 | src[j + 1] << 16 | src[j + 2] << 8 | 0xff;
    }
  }
  return {
    srcPos: srcPos + len,
    destPos
  };
}
function grayToRGBA(src, dest) {
  if (util_FeatureTest.isLittleEndian) {
    for (let i = 0, ii = src.length; i < ii; i++) {
      dest[i] = src[i] * 0x10101 | 0xff000000;
    }
  } else {
    for (let i = 0, ii = src.length; i < ii; i++) {
      dest[i] = src[i] * 0x1010100 | 0x000000ff;
    }
  }
}

;// ./src/core/jpg.js










class JpegError extends BaseException {
  constructor(msg) {
    super(msg, "JpegError");
  }
}
class DNLMarkerError extends BaseException {
  constructor(message, scanLines) {
    super(message, "DNLMarkerError");
    this.scanLines = scanLines;
  }
}
class EOIMarkerError extends BaseException {
  constructor(msg) {
    super(msg, "EOIMarkerError");
  }
}
const dctZigZag = new Uint8Array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]);
const dctCos1 = 4017;
const dctSin1 = 799;
const dctCos3 = 3406;
const dctSin3 = 2276;
const dctCos6 = 1567;
const dctSin6 = 3784;
const dctSqrt2 = 5793;
const dctSqrt1d2 = 2896;
function buildHuffmanTable(codeLengths, values) {
  let k = 0,
    i,
    j,
    length = 16;
  while (length > 0 && !codeLengths[length - 1]) {
    length--;
  }
  const code = [{
    children: [],
    index: 0
  }];
  let p = code[0],
    q;
  for (i = 0; i < length; i++) {
    for (j = 0; j < codeLengths[i]; j++) {
      p = code.pop();
      p.children[p.index] = values[k];
      while (p.index > 0) {
        p = code.pop();
      }
      p.index++;
      code.push(p);
      while (code.length <= i) {
        code.push(q = {
          children: [],
          index: 0
        });
        p.children[p.index] = q.children;
        p = q;
      }
      k++;
    }
    if (i + 1 < length) {
      code.push(q = {
        children: [],
        index: 0
      });
      p.children[p.index] = q.children;
      p = q;
    }
  }
  return code[0].children;
}
function getBlockBufferOffset(component, row, col) {
  return 64 * ((component.blocksPerLine + 1) * row + col);
}
function decodeScan(data, offset, frame, components, resetInterval, spectralStart, spectralEnd, successivePrev, successive, parseDNLMarker = false) {
  const mcusPerLine = frame.mcusPerLine;
  const progressive = frame.progressive;
  const startOffset = offset;
  let bitsData = 0,
    bitsCount = 0;
  function readBit() {
    if (bitsCount > 0) {
      bitsCount--;
      return bitsData >> bitsCount & 1;
    }
    bitsData = data[offset++];
    if (bitsData === 0xff) {
      const nextByte = data[offset++];
      if (nextByte) {
        if (nextByte === 0xdc && parseDNLMarker) {
          offset += 2;
          const scanLines = readUint16(data, offset);
          offset += 2;
          if (scanLines > 0 && scanLines !== frame.scanLines) {
            throw new DNLMarkerError("Found DNL marker (0xFFDC) while parsing scan data", scanLines);
          }
        } else if (nextByte === 0xd9) {
          if (parseDNLMarker) {
            const maybeScanLines = blockRow * (frame.precision === 8 ? 8 : 0);
            if (maybeScanLines > 0 && Math.round(frame.scanLines / maybeScanLines) >= 5) {
              throw new DNLMarkerError("Found EOI marker (0xFFD9) while parsing scan data, " + "possibly caused by incorrect `scanLines` parameter", maybeScanLines);
            }
          }
          throw new EOIMarkerError("Found EOI marker (0xFFD9) while parsing scan data");
        }
        throw new JpegError(`unexpected marker ${(bitsData << 8 | nextByte).toString(16)}`);
      }
    }
    bitsCount = 7;
    return bitsData >>> 7;
  }
  function decodeHuffman(tree) {
    let node = tree;
    while (true) {
      node = node[readBit()];
      switch (typeof node) {
        case "number":
          return node;
        case "object":
          continue;
      }
      throw new JpegError("invalid huffman sequence");
    }
  }
  function receive(length) {
    let n = 0;
    while (length > 0) {
      n = n << 1 | readBit();
      length--;
    }
    return n;
  }
  function receiveAndExtend(length) {
    if (length === 1) {
      return readBit() === 1 ? 1 : -1;
    }
    const n = receive(length);
    if (n >= 1 << length - 1) {
      return n;
    }
    return n + (-1 << length) + 1;
  }
  function decodeBaseline(component, blockOffset) {
    const t = decodeHuffman(component.huffmanTableDC);
    const diff = t === 0 ? 0 : receiveAndExtend(t);
    component.blockData[blockOffset] = component.pred += diff;
    let k = 1;
    while (k < 64) {
      const rs = decodeHuffman(component.huffmanTableAC);
      const s = rs & 15,
        r = rs >> 4;
      if (s === 0) {
        if (r < 15) {
          break;
        }
        k += 16;
        continue;
      }
      k += r;
      const z = dctZigZag[k];
      component.blockData[blockOffset + z] = receiveAndExtend(s);
      k++;
    }
  }
  function decodeDCFirst(component, blockOffset) {
    const t = decodeHuffman(component.huffmanTableDC);
    const diff = t === 0 ? 0 : receiveAndExtend(t) << successive;
    component.blockData[blockOffset] = component.pred += diff;
  }
  function decodeDCSuccessive(component, blockOffset) {
    component.blockData[blockOffset] |= readBit() << successive;
  }
  let eobrun = 0;
  function decodeACFirst(component, blockOffset) {
    if (eobrun > 0) {
      eobrun--;
      return;
    }
    let k = spectralStart;
    const e = spectralEnd;
    while (k <= e) {
      const rs = decodeHuffman(component.huffmanTableAC);
      const s = rs & 15,
        r = rs >> 4;
      if (s === 0) {
        if (r < 15) {
          eobrun = receive(r) + (1 << r) - 1;
          break;
        }
        k += 16;
        continue;
      }
      k += r;
      const z = dctZigZag[k];
      component.blockData[blockOffset + z] = receiveAndExtend(s) * (1 << successive);
      k++;
    }
  }
  let successiveACState = 0,
    successiveACNextValue;
  function decodeACSuccessive(component, blockOffset) {
    let k = spectralStart;
    const e = spectralEnd;
    let r = 0;
    let s;
    let rs;
    while (k <= e) {
      const offsetZ = blockOffset + dctZigZag[k];
      const sign = component.blockData[offsetZ] < 0 ? -1 : 1;
      switch (successiveACState) {
        case 0:
          rs = decodeHuffman(component.huffmanTableAC);
          s = rs & 15;
          r = rs >> 4;
          if (s === 0) {
            if (r < 15) {
              eobrun = receive(r) + (1 << r);
              successiveACState = 4;
            } else {
              r = 16;
              successiveACState = 1;
            }
          } else {
            if (s !== 1) {
              throw new JpegError("invalid ACn encoding");
            }
            successiveACNextValue = receiveAndExtend(s);
            successiveACState = r ? 2 : 3;
          }
          continue;
        case 1:
        case 2:
          if (component.blockData[offsetZ]) {
            component.blockData[offsetZ] += sign * (readBit() << successive);
          } else {
            r--;
            if (r === 0) {
              successiveACState = successiveACState === 2 ? 3 : 0;
            }
          }
          break;
        case 3:
          if (component.blockData[offsetZ]) {
            component.blockData[offsetZ] += sign * (readBit() << successive);
          } else {
            component.blockData[offsetZ] = successiveACNextValue << successive;
            successiveACState = 0;
          }
          break;
        case 4:
          if (component.blockData[offsetZ]) {
            component.blockData[offsetZ] += sign * (readBit() << successive);
          }
          break;
      }
      k++;
    }
    if (successiveACState === 4) {
      eobrun--;
      if (eobrun === 0) {
        successiveACState = 0;
      }
    }
  }
  let blockRow = 0;
  function decodeMcu(component, decode, mcu, row, col) {
    const mcuRow = mcu / mcusPerLine | 0;
    const mcuCol = mcu % mcusPerLine;
    blockRow = mcuRow * component.v + row;
    const blockCol = mcuCol * component.h + col;
    const blockOffset = getBlockBufferOffset(component, blockRow, blockCol);
    decode(component, blockOffset);
  }
  function decodeBlock(component, decode, mcu) {
    blockRow = mcu / component.blocksPerLine | 0;
    const blockCol = mcu % component.blocksPerLine;
    const blockOffset = getBlockBufferOffset(component, blockRow, blockCol);
    decode(component, blockOffset);
  }
  const componentsLength = components.length;
  let component, i, j, k, n;
  let decodeFn;
  if (progressive) {
    if (spectralStart === 0) {
      decodeFn = successivePrev === 0 ? decodeDCFirst : decodeDCSuccessive;
    } else {
      decodeFn = successivePrev === 0 ? decodeACFirst : decodeACSuccessive;
    }
  } else {
    decodeFn = decodeBaseline;
  }
  let mcu = 0,
    fileMarker;
  const mcuExpected = componentsLength === 1 ? components[0].blocksPerLine * components[0].blocksPerColumn : mcusPerLine * frame.mcusPerColumn;
  let h, v;
  while (mcu <= mcuExpected) {
    const mcuToRead = resetInterval ? Math.min(mcuExpected - mcu, resetInterval) : mcuExpected;
    if (mcuToRead > 0) {
      for (i = 0; i < componentsLength; i++) {
        components[i].pred = 0;
      }
      eobrun = 0;
      if (componentsLength === 1) {
        component = components[0];
        for (n = 0; n < mcuToRead; n++) {
          decodeBlock(component, decodeFn, mcu);
          mcu++;
        }
      } else {
        for (n = 0; n < mcuToRead; n++) {
          for (i = 0; i < componentsLength; i++) {
            component = components[i];
            h = component.h;
            v = component.v;
            for (j = 0; j < v; j++) {
              for (k = 0; k < h; k++) {
                decodeMcu(component, decodeFn, mcu, j, k);
              }
            }
          }
          mcu++;
        }
      }
    }
    bitsCount = 0;
    fileMarker = findNextFileMarker(data, offset);
    if (!fileMarker) {
      break;
    }
    if (fileMarker.invalid) {
      const partialMsg = mcuToRead > 0 ? "unexpected" : "excessive";
      util_warn(`decodeScan - ${partialMsg} MCU data, current marker is: ${fileMarker.invalid}`);
      offset = fileMarker.offset;
    }
    if (fileMarker.marker >= 0xffd0 && fileMarker.marker <= 0xffd7) {
      offset += 2;
    } else {
      break;
    }
  }
  return offset - startOffset;
}
function quantizeAndInverse(component, blockBufferOffset, p) {
  const qt = component.quantizationTable,
    blockData = component.blockData;
  let v0, v1, v2, v3, v4, v5, v6, v7;
  let p0, p1, p2, p3, p4, p5, p6, p7;
  let t;
  if (!qt) {
    throw new JpegError("missing required Quantization Table.");
  }
  for (let row = 0; row < 64; row += 8) {
    p0 = blockData[blockBufferOffset + row];
    p1 = blockData[blockBufferOffset + row + 1];
    p2 = blockData[blockBufferOffset + row + 2];
    p3 = blockData[blockBufferOffset + row + 3];
    p4 = blockData[blockBufferOffset + row + 4];
    p5 = blockData[blockBufferOffset + row + 5];
    p6 = blockData[blockBufferOffset + row + 6];
    p7 = blockData[blockBufferOffset + row + 7];
    p0 *= qt[row];
    if ((p1 | p2 | p3 | p4 | p5 | p6 | p7) === 0) {
      t = dctSqrt2 * p0 + 512 >> 10;
      p[row] = t;
      p[row + 1] = t;
      p[row + 2] = t;
      p[row + 3] = t;
      p[row + 4] = t;
      p[row + 5] = t;
      p[row + 6] = t;
      p[row + 7] = t;
      continue;
    }
    p1 *= qt[row + 1];
    p2 *= qt[row + 2];
    p3 *= qt[row + 3];
    p4 *= qt[row + 4];
    p5 *= qt[row + 5];
    p6 *= qt[row + 6];
    p7 *= qt[row + 7];
    v0 = dctSqrt2 * p0 + 128 >> 8;
    v1 = dctSqrt2 * p4 + 128 >> 8;
    v2 = p2;
    v3 = p6;
    v4 = dctSqrt1d2 * (p1 - p7) + 128 >> 8;
    v7 = dctSqrt1d2 * (p1 + p7) + 128 >> 8;
    v5 = p3 << 4;
    v6 = p5 << 4;
    v0 = v0 + v1 + 1 >> 1;
    v1 = v0 - v1;
    t = v2 * dctSin6 + v3 * dctCos6 + 128 >> 8;
    v2 = v2 * dctCos6 - v3 * dctSin6 + 128 >> 8;
    v3 = t;
    v4 = v4 + v6 + 1 >> 1;
    v6 = v4 - v6;
    v7 = v7 + v5 + 1 >> 1;
    v5 = v7 - v5;
    v0 = v0 + v3 + 1 >> 1;
    v3 = v0 - v3;
    v1 = v1 + v2 + 1 >> 1;
    v2 = v1 - v2;
    t = v4 * dctSin3 + v7 * dctCos3 + 2048 >> 12;
    v4 = v4 * dctCos3 - v7 * dctSin3 + 2048 >> 12;
    v7 = t;
    t = v5 * dctSin1 + v6 * dctCos1 + 2048 >> 12;
    v5 = v5 * dctCos1 - v6 * dctSin1 + 2048 >> 12;
    v6 = t;
    p[row] = v0 + v7;
    p[row + 7] = v0 - v7;
    p[row + 1] = v1 + v6;
    p[row + 6] = v1 - v6;
    p[row + 2] = v2 + v5;
    p[row + 5] = v2 - v5;
    p[row + 3] = v3 + v4;
    p[row + 4] = v3 - v4;
  }
  for (let col = 0; col < 8; ++col) {
    p0 = p[col];
    p1 = p[col + 8];
    p2 = p[col + 16];
    p3 = p[col + 24];
    p4 = p[col + 32];
    p5 = p[col + 40];
    p6 = p[col + 48];
    p7 = p[col + 56];
    if ((p1 | p2 | p3 | p4 | p5 | p6 | p7) === 0) {
      t = dctSqrt2 * p0 + 8192 >> 14;
      if (t < -2040) {
        t = 0;
      } else if (t >= 2024) {
        t = 255;
      } else {
        t = t + 2056 >> 4;
      }
      blockData[blockBufferOffset + col] = t;
      blockData[blockBufferOffset + col + 8] = t;
      blockData[blockBufferOffset + col + 16] = t;
      blockData[blockBufferOffset + col + 24] = t;
      blockData[blockBufferOffset + col + 32] = t;
      blockData[blockBufferOffset + col + 40] = t;
      blockData[blockBufferOffset + col + 48] = t;
      blockData[blockBufferOffset + col + 56] = t;
      continue;
    }
    v0 = dctSqrt2 * p0 + 2048 >> 12;
    v1 = dctSqrt2 * p4 + 2048 >> 12;
    v2 = p2;
    v3 = p6;
    v4 = dctSqrt1d2 * (p1 - p7) + 2048 >> 12;
    v7 = dctSqrt1d2 * (p1 + p7) + 2048 >> 12;
    v5 = p3;
    v6 = p5;
    v0 = (v0 + v1 + 1 >> 1) + 4112;
    v1 = v0 - v1;
    t = v2 * dctSin6 + v3 * dctCos6 + 2048 >> 12;
    v2 = v2 * dctCos6 - v3 * dctSin6 + 2048 >> 12;
    v3 = t;
    v4 = v4 + v6 + 1 >> 1;
    v6 = v4 - v6;
    v7 = v7 + v5 + 1 >> 1;
    v5 = v7 - v5;
    v0 = v0 + v3 + 1 >> 1;
    v3 = v0 - v3;
    v1 = v1 + v2 + 1 >> 1;
    v2 = v1 - v2;
    t = v4 * dctSin3 + v7 * dctCos3 + 2048 >> 12;
    v4 = v4 * dctCos3 - v7 * dctSin3 + 2048 >> 12;
    v7 = t;
    t = v5 * dctSin1 + v6 * dctCos1 + 2048 >> 12;
    v5 = v5 * dctCos1 - v6 * dctSin1 + 2048 >> 12;
    v6 = t;
    p0 = v0 + v7;
    p7 = v0 - v7;
    p1 = v1 + v6;
    p6 = v1 - v6;
    p2 = v2 + v5;
    p5 = v2 - v5;
    p3 = v3 + v4;
    p4 = v3 - v4;
    if (p0 < 16) {
      p0 = 0;
    } else if (p0 >= 4080) {
      p0 = 255;
    } else {
      p0 >>= 4;
    }
    if (p1 < 16) {
      p1 = 0;
    } else if (p1 >= 4080) {
      p1 = 255;
    } else {
      p1 >>= 4;
    }
    if (p2 < 16) {
      p2 = 0;
    } else if (p2 >= 4080) {
      p2 = 255;
    } else {
      p2 >>= 4;
    }
    if (p3 < 16) {
      p3 = 0;
    } else if (p3 >= 4080) {
      p3 = 255;
    } else {
      p3 >>= 4;
    }
    if (p4 < 16) {
      p4 = 0;
    } else if (p4 >= 4080) {
      p4 = 255;
    } else {
      p4 >>= 4;
    }
    if (p5 < 16) {
      p5 = 0;
    } else if (p5 >= 4080) {
      p5 = 255;
    } else {
      p5 >>= 4;
    }
    if (p6 < 16) {
      p6 = 0;
    } else if (p6 >= 4080) {
      p6 = 255;
    } else {
      p6 >>= 4;
    }
    if (p7 < 16) {
      p7 = 0;
    } else if (p7 >= 4080) {
      p7 = 255;
    } else {
      p7 >>= 4;
    }
    blockData[blockBufferOffset + col] = p0;
    blockData[blockBufferOffset + col + 8] = p1;
    blockData[blockBufferOffset + col + 16] = p2;
    blockData[blockBufferOffset + col + 24] = p3;
    blockData[blockBufferOffset + col + 32] = p4;
    blockData[blockBufferOffset + col + 40] = p5;
    blockData[blockBufferOffset + col + 48] = p6;
    blockData[blockBufferOffset + col + 56] = p7;
  }
}
function buildComponentData(frame, component) {
  const blocksPerLine = component.blocksPerLine;
  const blocksPerColumn = component.blocksPerColumn;
  const computationBuffer = new Int16Array(64);
  for (let blockRow = 0; blockRow < blocksPerColumn; blockRow++) {
    for (let blockCol = 0; blockCol < blocksPerLine; blockCol++) {
      const offset = getBlockBufferOffset(component, blockRow, blockCol);
      quantizeAndInverse(component, offset, computationBuffer);
    }
  }
  return component.blockData;
}
function findNextFileMarker(data, currentPos, startPos = currentPos) {
  const maxPos = data.length - 1;
  let newPos = startPos < currentPos ? startPos : currentPos;
  if (currentPos >= maxPos) {
    return null;
  }
  const currentMarker = readUint16(data, currentPos);
  if (currentMarker >= 0xffc0 && currentMarker <= 0xfffe) {
    return {
      invalid: null,
      marker: currentMarker,
      offset: currentPos
    };
  }
  let newMarker = readUint16(data, newPos);
  while (!(newMarker >= 0xffc0 && newMarker <= 0xfffe)) {
    if (++newPos >= maxPos) {
      return null;
    }
    newMarker = readUint16(data, newPos);
  }
  return {
    invalid: currentMarker.toString(16),
    marker: newMarker,
    offset: newPos
  };
}
function prepareComponents(frame) {
  const mcusPerLine = Math.ceil(frame.samplesPerLine / 8 / frame.maxH);
  const mcusPerColumn = Math.ceil(frame.scanLines / 8 / frame.maxV);
  for (const component of frame.components) {
    const blocksPerLine = Math.ceil(Math.ceil(frame.samplesPerLine / 8) * component.h / frame.maxH);
    const blocksPerColumn = Math.ceil(Math.ceil(frame.scanLines / 8) * component.v / frame.maxV);
    const blocksPerLineForMcu = mcusPerLine * component.h;
    const blocksPerColumnForMcu = mcusPerColumn * component.v;
    const blocksBufferSize = 64 * blocksPerColumnForMcu * (blocksPerLineForMcu + 1);
    component.blockData = new Int16Array(blocksBufferSize);
    component.blocksPerLine = blocksPerLine;
    component.blocksPerColumn = blocksPerColumn;
  }
  frame.mcusPerLine = mcusPerLine;
  frame.mcusPerColumn = mcusPerColumn;
}
function readDataBlock(data, offset) {
  const length = readUint16(data, offset);
  offset += 2;
  let endOffset = offset + length - 2;
  const fileMarker = findNextFileMarker(data, endOffset, offset);
  if (fileMarker?.invalid) {
    util_warn("readDataBlock - incorrect length, current marker is: " + fileMarker.invalid);
    endOffset = fileMarker.offset;
  }
  const array = data.subarray(offset, endOffset);
  offset += array.length;
  return {
    appData: array,
    newOffset: offset
  };
}
function skipData(data, offset) {
  const length = readUint16(data, offset);
  offset += 2;
  const endOffset = offset + length - 2;
  const fileMarker = findNextFileMarker(data, endOffset, offset);
  if (fileMarker?.invalid) {
    return fileMarker.offset;
  }
  return endOffset;
}
class JpegImage {
  constructor({
    decodeTransform = null,
    colorTransform = -1
  } = {}) {
    this._decodeTransform = decodeTransform;
    this._colorTransform = colorTransform;
  }
  static canUseImageDecoder(data, colorTransform = -1) {
    let offset = 0;
    let numComponents = null;
    let fileMarker = readUint16(data, offset);
    offset += 2;
    if (fileMarker !== 0xffd8) {
      throw new JpegError("SOI not found");
    }
    fileMarker = readUint16(data, offset);
    offset += 2;
    markerLoop: while (fileMarker !== 0xffd9) {
      switch (fileMarker) {
        case 0xffc0:
        case 0xffc1:
        case 0xffc2:
          numComponents = data[offset + (2 + 1 + 2 + 2)];
          break markerLoop;
        case 0xffff:
          if (data[offset] !== 0xff) {
            offset--;
          }
          break;
      }
      offset = skipData(data, offset);
      fileMarker = readUint16(data, offset);
      offset += 2;
    }
    if (numComponents === 4) {
      return false;
    }
    if (numComponents === 3 && colorTransform === 0) {
      return false;
    }
    return true;
  }
  parse(data, {
    dnlScanLines = null
  } = {}) {
    let offset = 0;
    let jfif = null;
    let adobe = null;
    let frame, resetInterval;
    let numSOSMarkers = 0;
    const quantizationTables = [];
    const huffmanTablesAC = [],
      huffmanTablesDC = [];
    let fileMarker = readUint16(data, offset);
    offset += 2;
    if (fileMarker !== 0xffd8) {
      throw new JpegError("SOI not found");
    }
    fileMarker = readUint16(data, offset);
    offset += 2;
    markerLoop: while (fileMarker !== 0xffd9) {
      let i, j, l;
      switch (fileMarker) {
        case 0xffe0:
        case 0xffe1:
        case 0xffe2:
        case 0xffe3:
        case 0xffe4:
        case 0xffe5:
        case 0xffe6:
        case 0xffe7:
        case 0xffe8:
        case 0xffe9:
        case 0xffea:
        case 0xffeb:
        case 0xffec:
        case 0xffed:
        case 0xffee:
        case 0xffef:
        case 0xfffe:
          const {
            appData,
            newOffset
          } = readDataBlock(data, offset);
          offset = newOffset;
          if (fileMarker === 0xffe0) {
            if (appData[0] === 0x4a && appData[1] === 0x46 && appData[2] === 0x49 && appData[3] === 0x46 && appData[4] === 0) {
              jfif = {
                version: {
                  major: appData[5],
                  minor: appData[6]
                },
                densityUnits: appData[7],
                xDensity: appData[8] << 8 | appData[9],
                yDensity: appData[10] << 8 | appData[11],
                thumbWidth: appData[12],
                thumbHeight: appData[13],
                thumbData: appData.subarray(14, 14 + 3 * appData[12] * appData[13])
              };
            }
          }
          if (fileMarker === 0xffee) {
            if (appData[0] === 0x41 && appData[1] === 0x64 && appData[2] === 0x6f && appData[3] === 0x62 && appData[4] === 0x65) {
              adobe = {
                version: appData[5] << 8 | appData[6],
                flags0: appData[7] << 8 | appData[8],
                flags1: appData[9] << 8 | appData[10],
                transformCode: appData[11]
              };
            }
          }
          break;
        case 0xffdb:
          const quantizationTablesLength = readUint16(data, offset);
          offset += 2;
          const quantizationTablesEnd = quantizationTablesLength + offset - 2;
          let z;
          while (offset < quantizationTablesEnd) {
            const quantizationTableSpec = data[offset++];
            const tableData = new Uint16Array(64);
            if (quantizationTableSpec >> 4 === 0) {
              for (j = 0; j < 64; j++) {
                z = dctZigZag[j];
                tableData[z] = data[offset++];
              }
            } else if (quantizationTableSpec >> 4 === 1) {
              for (j = 0; j < 64; j++) {
                z = dctZigZag[j];
                tableData[z] = readUint16(data, offset);
                offset += 2;
              }
            } else {
              throw new JpegError("DQT - invalid table spec");
            }
            quantizationTables[quantizationTableSpec & 15] = tableData;
          }
          break;
        case 0xffc0:
        case 0xffc1:
        case 0xffc2:
          if (frame) {
            throw new JpegError("Only single frame JPEGs supported");
          }
          offset += 2;
          frame = {};
          frame.extended = fileMarker === 0xffc1;
          frame.progressive = fileMarker === 0xffc2;
          frame.precision = data[offset++];
          const sofScanLines = readUint16(data, offset);
          offset += 2;
          frame.scanLines = dnlScanLines || sofScanLines;
          frame.samplesPerLine = readUint16(data, offset);
          offset += 2;
          frame.components = [];
          frame.componentIds = {};
          const componentsCount = data[offset++];
          let maxH = 0,
            maxV = 0;
          for (i = 0; i < componentsCount; i++) {
            const componentId = data[offset];
            const h = data[offset + 1] >> 4;
            const v = data[offset + 1] & 15;
            if (maxH < h) {
              maxH = h;
            }
            if (maxV < v) {
              maxV = v;
            }
            const qId = data[offset + 2];
            l = frame.components.push({
              h,
              v,
              quantizationId: qId,
              quantizationTable: null
            });
            frame.componentIds[componentId] = l - 1;
            offset += 3;
          }
          frame.maxH = maxH;
          frame.maxV = maxV;
          prepareComponents(frame);
          break;
        case 0xffc4:
          const huffmanLength = readUint16(data, offset);
          offset += 2;
          for (i = 2; i < huffmanLength;) {
            const huffmanTableSpec = data[offset++];
            const codeLengths = new Uint8Array(16);
            let codeLengthSum = 0;
            for (j = 0; j < 16; j++, offset++) {
              codeLengthSum += codeLengths[j] = data[offset];
            }
            const huffmanValues = new Uint8Array(codeLengthSum);
            for (j = 0; j < codeLengthSum; j++, offset++) {
              huffmanValues[j] = data[offset];
            }
            i += 17 + codeLengthSum;
            (huffmanTableSpec >> 4 === 0 ? huffmanTablesDC : huffmanTablesAC)[huffmanTableSpec & 15] = buildHuffmanTable(codeLengths, huffmanValues);
          }
          break;
        case 0xffdd:
          offset += 2;
          resetInterval = readUint16(data, offset);
          offset += 2;
          break;
        case 0xffda:
          const parseDNLMarker = ++numSOSMarkers === 1 && !dnlScanLines;
          offset += 2;
          const selectorsCount = data[offset++],
            components = [];
          for (i = 0; i < selectorsCount; i++) {
            const index = data[offset++];
            const componentIndex = frame.componentIds[index];
            const component = frame.components[componentIndex];
            component.index = index;
            const tableSpec = data[offset++];
            component.huffmanTableDC = huffmanTablesDC[tableSpec >> 4];
            component.huffmanTableAC = huffmanTablesAC[tableSpec & 15];
            components.push(component);
          }
          const spectralStart = data[offset++],
            spectralEnd = data[offset++],
            successiveApproximation = data[offset++];
          try {
            const processed = decodeScan(data, offset, frame, components, resetInterval, spectralStart, spectralEnd, successiveApproximation >> 4, successiveApproximation & 15, parseDNLMarker);
            offset += processed;
          } catch (ex) {
            if (ex instanceof DNLMarkerError) {
              util_warn(`${ex.message} -- attempting to re-parse the JPEG image.`);
              return this.parse(data, {
                dnlScanLines: ex.scanLines
              });
            } else if (ex instanceof EOIMarkerError) {
              util_warn(`${ex.message} -- ignoring the rest of the image data.`);
              break markerLoop;
            }
            throw ex;
          }
          break;
        case 0xffdc:
          offset += 4;
          break;
        case 0xffff:
          if (data[offset] !== 0xff) {
            offset--;
          }
          break;
        default:
          const nextFileMarker = findNextFileMarker(data, offset - 2, offset - 3);
          if (nextFileMarker?.invalid) {
            util_warn("JpegImage.parse - unexpected data, current marker is: " + nextFileMarker.invalid);
            offset = nextFileMarker.offset;
            break;
          }
          if (!nextFileMarker || offset >= data.length - 1) {
            util_warn("JpegImage.parse - reached the end of the image data " + "without finding an EOI marker (0xFFD9).");
            break markerLoop;
          }
          throw new JpegError("JpegImage.parse - unknown marker: " + fileMarker.toString(16));
      }
      fileMarker = readUint16(data, offset);
      offset += 2;
    }
    if (!frame) {
      throw new JpegError("JpegImage.parse - no frame data found.");
    }
    this.width = frame.samplesPerLine;
    this.height = frame.scanLines;
    this.jfif = jfif;
    this.adobe = adobe;
    this.components = [];
    for (const component of frame.components) {
      const quantizationTable = quantizationTables[component.quantizationId];
      if (quantizationTable) {
        component.quantizationTable = quantizationTable;
      }
      this.components.push({
        index: component.index,
        output: buildComponentData(frame, component),
        scaleX: component.h / frame.maxH,
        scaleY: component.v / frame.maxV,
        blocksPerLine: component.blocksPerLine,
        blocksPerColumn: component.blocksPerColumn
      });
    }
    this.numComponents = this.components.length;
    return undefined;
  }
  _getLinearizedBlockData(width, height, isSourcePDF = false) {
    const scaleX = this.width / width,
      scaleY = this.height / height;
    let component, componentScaleX, componentScaleY, blocksPerScanline;
    let x, y, i, j, k;
    let index;
    let offset = 0;
    let output;
    const numComponents = this.components.length;
    const dataLength = width * height * numComponents;
    const data = new Uint8ClampedArray(dataLength);
    const xScaleBlockOffset = new Uint32Array(width);
    const mask3LSB = 0xfffffff8;
    let lastComponentScaleX;
    for (i = 0; i < numComponents; i++) {
      component = this.components[i];
      componentScaleX = component.scaleX * scaleX;
      componentScaleY = component.scaleY * scaleY;
      offset = i;
      output = component.output;
      blocksPerScanline = component.blocksPerLine + 1 << 3;
      if (componentScaleX !== lastComponentScaleX) {
        for (x = 0; x < width; x++) {
          j = 0 | x * componentScaleX;
          xScaleBlockOffset[x] = (j & mask3LSB) << 3 | j & 7;
        }
        lastComponentScaleX = componentScaleX;
      }
      for (y = 0; y < height; y++) {
        j = 0 | y * componentScaleY;
        index = blocksPerScanline * (j & mask3LSB) | (j & 7) << 3;
        for (x = 0; x < width; x++) {
          data[offset] = output[index + xScaleBlockOffset[x]];
          offset += numComponents;
        }
      }
    }
    let transform = this._decodeTransform;
    if (!isSourcePDF && numComponents === 4 && !transform) {
      transform = new Int32Array([-256, 255, -256, 255, -256, 255, -256, 255]);
    }
    if (transform) {
      for (i = 0; i < dataLength;) {
        for (j = 0, k = 0; j < numComponents; j++, i++, k += 2) {
          data[i] = (data[i] * transform[k] >> 8) + transform[k + 1];
        }
      }
    }
    return data;
  }
  get _isColorConversionNeeded() {
    if (this.adobe) {
      return !!this.adobe.transformCode;
    }
    if (this.numComponents === 3) {
      if (this._colorTransform === 0) {
        return false;
      } else if (this.components[0].index === 0x52 && this.components[1].index === 0x47 && this.components[2].index === 0x42) {
        return false;
      }
      return true;
    }
    if (this._colorTransform === 1) {
      return true;
    }
    return false;
  }
  _convertYccToRgb(data) {
    let Y, Cb, Cr;
    for (let i = 0, length = data.length; i < length; i += 3) {
      Y = data[i];
      Cb = data[i + 1];
      Cr = data[i + 2];
      data[i] = Y - 179.456 + 1.402 * Cr;
      data[i + 1] = Y + 135.459 - 0.344 * Cb - 0.714 * Cr;
      data[i + 2] = Y - 226.816 + 1.772 * Cb;
    }
    return data;
  }
  _convertYccToRgba(data, out) {
    for (let i = 0, j = 0, length = data.length; i < length; i += 3, j += 4) {
      const Y = data[i];
      const Cb = data[i + 1];
      const Cr = data[i + 2];
      out[j] = Y - 179.456 + 1.402 * Cr;
      out[j + 1] = Y + 135.459 - 0.344 * Cb - 0.714 * Cr;
      out[j + 2] = Y - 226.816 + 1.772 * Cb;
      out[j + 3] = 255;
    }
    return out;
  }
  _convertYcckToRgb(data) {
    let Y, Cb, Cr, k;
    let offset = 0;
    for (let i = 0, length = data.length; i < length; i += 4) {
      Y = data[i];
      Cb = data[i + 1];
      Cr = data[i + 2];
      k = data[i + 3];
      data[offset++] = -122.67195406894 + Cb * (-6.60635669420364e-5 * Cb + 0.000437130475926232 * Cr - 5.4080610064599e-5 * Y + 0.00048449797120281 * k - 0.154362151871126) + Cr * (-0.000957964378445773 * Cr + 0.000817076911346625 * Y - 0.00477271405408747 * k + 1.53380253221734) + Y * (0.000961250184130688 * Y - 0.00266257332283933 * k + 0.48357088451265) + k * (-0.000336197177618394 * k + 0.484791561490776);
      data[offset++] = 107.268039397724 + Cb * (2.19927104525741e-5 * Cb - 0.000640992018297945 * Cr + 0.000659397001245577 * Y + 0.000426105652938837 * k - 0.176491792462875) + Cr * (-0.000778269941513683 * Cr + 0.00130872261408275 * Y + 0.000770482631801132 * k - 0.151051492775562) + Y * (0.00126935368114843 * Y - 0.00265090189010898 * k + 0.25802910206845) + k * (-0.000318913117588328 * k - 0.213742400323665);
      data[offset++] = -20.810012546947 + Cb * (-0.000570115196973677 * Cb - 2.63409051004589e-5 * Cr + 0.0020741088115012 * Y - 0.00288260236853442 * k + 0.814272968359295) + Cr * (-1.53496057440975e-5 * Cr - 0.000132689043961446 * Y + 0.000560833691242812 * k - 0.195152027534049) + Y * (0.00174418132927582 * Y - 0.00255243321439347 * k + 0.116935020465145) + k * (-0.000343531996510555 * k + 0.24165260232407);
    }
    return data.subarray(0, offset);
  }
  _convertYcckToRgba(data) {
    for (let i = 0, length = data.length; i < length; i += 4) {
      const Y = data[i];
      const Cb = data[i + 1];
      const Cr = data[i + 2];
      const k = data[i + 3];
      data[i] = -122.67195406894 + Cb * (-6.60635669420364e-5 * Cb + 0.000437130475926232 * Cr - 5.4080610064599e-5 * Y + 0.00048449797120281 * k - 0.154362151871126) + Cr * (-0.000957964378445773 * Cr + 0.000817076911346625 * Y - 0.00477271405408747 * k + 1.53380253221734) + Y * (0.000961250184130688 * Y - 0.00266257332283933 * k + 0.48357088451265) + k * (-0.000336197177618394 * k + 0.484791561490776);
      data[i + 1] = 107.268039397724 + Cb * (2.19927104525741e-5 * Cb - 0.000640992018297945 * Cr + 0.000659397001245577 * Y + 0.000426105652938837 * k - 0.176491792462875) + Cr * (-0.000778269941513683 * Cr + 0.00130872261408275 * Y + 0.000770482631801132 * k - 0.151051492775562) + Y * (0.00126935368114843 * Y - 0.00265090189010898 * k + 0.25802910206845) + k * (-0.000318913117588328 * k - 0.213742400323665);
      data[i + 2] = -20.810012546947 + Cb * (-0.000570115196973677 * Cb - 2.63409051004589e-5 * Cr + 0.0020741088115012 * Y - 0.00288260236853442 * k + 0.814272968359295) + Cr * (-1.53496057440975e-5 * Cr - 0.000132689043961446 * Y + 0.000560833691242812 * k - 0.195152027534049) + Y * (0.00174418132927582 * Y - 0.00255243321439347 * k + 0.116935020465145) + k * (-0.000343531996510555 * k + 0.24165260232407);
      data[i + 3] = 255;
    }
    return data;
  }
  _convertYcckToCmyk(data) {
    let Y, Cb, Cr;
    for (let i = 0, length = data.length; i < length; i += 4) {
      Y = data[i];
      Cb = data[i + 1];
      Cr = data[i + 2];
      data[i] = 434.456 - Y - 1.402 * Cr;
      data[i + 1] = 119.541 - Y + 0.344 * Cb + 0.714 * Cr;
      data[i + 2] = 481.816 - Y - 1.772 * Cb;
    }
    return data;
  }
  _convertCmykToRgb(data) {
    let c, m, y, k;
    let offset = 0;
    for (let i = 0, length = data.length; i < length; i += 4) {
      c = data[i];
      m = data[i + 1];
      y = data[i + 2];
      k = data[i + 3];
      data[offset++] = 255 + c * (-0.00006747147073602441 * c + 0.0008379262121013727 * m + 0.0002894718188643294 * y + 0.003264231057537806 * k - 1.1185611867203937) + m * (0.000026374107616089405 * m - 0.00008626949158638572 * y - 0.0002748769067499491 * k - 0.02155688794978967) + y * (-0.00003878099212869363 * y - 0.0003267808279485286 * k + 0.0686742238595345) - k * (0.0003361971776183937 * k + 0.7430659151342254);
      data[offset++] = 255 + c * (0.00013596372813588848 * c + 0.000924537132573585 * m + 0.00010567359618683593 * y + 0.0004791864687436512 * k - 0.3109689587515875) + m * (-0.00023545346108370344 * m + 0.0002702845253534714 * y + 0.0020200308977307156 * k - 0.7488052167015494) + y * (0.00006834815998235662 * y + 0.00015168452363460973 * k - 0.09751927774728933) - k * (0.0003189131175883281 * k + 0.7364883807733168);
      data[offset++] = 255 + c * (0.000013598650411385307 * c + 0.00012423956175490851 * m + 0.0004751985097583589 * y - 0.0000036729317476630422 * k - 0.05562186980264034) + m * (0.00016141380598724676 * m + 0.0009692239130725186 * y + 0.0007782692450036253 * k - 0.44015232367526463) + y * (5.068882914068769e-7 * y + 0.0017778369011375071 * k - 0.7591454649749609) - k * (0.0003435319965105553 * k + 0.7063770186160144);
    }
    return data.subarray(0, offset);
  }
  _convertCmykToRgba(data) {
    for (let i = 0, length = data.length; i < length; i += 4) {
      const c = data[i];
      const m = data[i + 1];
      const y = data[i + 2];
      const k = data[i + 3];
      data[i] = 255 + c * (-0.00006747147073602441 * c + 0.0008379262121013727 * m + 0.0002894718188643294 * y + 0.003264231057537806 * k - 1.1185611867203937) + m * (0.000026374107616089405 * m - 0.00008626949158638572 * y - 0.0002748769067499491 * k - 0.02155688794978967) + y * (-0.00003878099212869363 * y - 0.0003267808279485286 * k + 0.0686742238595345) - k * (0.0003361971776183937 * k + 0.7430659151342254);
      data[i + 1] = 255 + c * (0.00013596372813588848 * c + 0.000924537132573585 * m + 0.00010567359618683593 * y + 0.0004791864687436512 * k - 0.3109689587515875) + m * (-0.00023545346108370344 * m + 0.0002702845253534714 * y + 0.0020200308977307156 * k - 0.7488052167015494) + y * (0.00006834815998235662 * y + 0.00015168452363460973 * k - 0.09751927774728933) - k * (0.0003189131175883281 * k + 0.7364883807733168);
      data[i + 2] = 255 + c * (0.000013598650411385307 * c + 0.00012423956175490851 * m + 0.0004751985097583589 * y - 0.0000036729317476630422 * k - 0.05562186980264034) + m * (0.00016141380598724676 * m + 0.0009692239130725186 * y + 0.0007782692450036253 * k - 0.44015232367526463) + y * (5.068882914068769e-7 * y + 0.0017778369011375071 * k - 0.7591454649749609) - k * (0.0003435319965105553 * k + 0.7063770186160144);
      data[i + 3] = 255;
    }
    return data;
  }
  getData({
    width,
    height,
    forceRGBA = false,
    forceRGB = false,
    isSourcePDF = false
  }) {
    if (this.numComponents > 4) {
      throw new JpegError("Unsupported color mode");
    }
    const data = this._getLinearizedBlockData(width, height, isSourcePDF);
    if (this.numComponents === 1 && (forceRGBA || forceRGB)) {
      const len = data.length * (forceRGBA ? 4 : 3);
      const rgbaData = new Uint8ClampedArray(len);
      let offset = 0;
      if (forceRGBA) {
        grayToRGBA(data, new Uint32Array(rgbaData.buffer));
      } else {
        for (const grayColor of data) {
          rgbaData[offset++] = grayColor;
          rgbaData[offset++] = grayColor;
          rgbaData[offset++] = grayColor;
        }
      }
      return rgbaData;
    } else if (this.numComponents === 3 && this._isColorConversionNeeded) {
      if (forceRGBA) {
        const rgbaData = new Uint8ClampedArray(data.length / 3 * 4);
        return this._convertYccToRgba(data, rgbaData);
      }
      return this._convertYccToRgb(data);
    } else if (this.numComponents === 4) {
      if (this._isColorConversionNeeded) {
        if (forceRGBA) {
          return this._convertYcckToRgba(data);
        }
        if (forceRGB) {
          return this._convertYcckToRgb(data);
        }
        return this._convertYcckToCmyk(data);
      } else if (forceRGBA) {
        return this._convertCmykToRgba(data);
      } else if (forceRGB) {
        return this._convertCmykToRgb(data);
      }
    }
    return data;
  }
}

// EXTERNAL MODULE: ./node_modules/core-js/modules/esnext.iterator.for-each.js
var esnext_iterator_for_each = __webpack_require__(3949);
// EXTERNAL MODULE: ./node_modules/core-js/modules/web.self.js
var web_self = __webpack_require__(3611);
;// ./external/openjpeg/openjpeg.js











var OpenJPEG = (() => {
  var _scriptName = typeof document != 'undefined' ? document.currentScript?.src : undefined;
  return function (moduleArg = {}) {
    var moduleRtn;
    var Module = moduleArg;
    var readyPromiseResolve, readyPromiseReject;
    var readyPromise = new Promise((resolve, reject) => {
      readyPromiseResolve = resolve;
      readyPromiseReject = reject;
    });
    var ENVIRONMENT_IS_WEB = true;
    var ENVIRONMENT_IS_WORKER = false;
    Module.decode = function (bytes, {
      numComponents = 4,
      isIndexedColormap = false,
      smaskInData = false
    }) {
      const size = bytes.length;
      const ptr = Module._malloc(size);
      Module.HEAPU8.set(bytes, ptr);
      const ret = Module._jp2_decode(ptr, size, numComponents > 0 ? numComponents : 0, !!isIndexedColormap, !!smaskInData);
      Module._free(ptr);
      if (ret) {
        const {
          errorMessages
        } = Module;
        if (errorMessages) {
          delete Module.errorMessages;
          return errorMessages;
        }
        return "Unknown error";
      }
      const {
        imageData
      } = Module;
      Module.imageData = null;
      return imageData;
    };
    var moduleOverrides = Object.assign({}, Module);
    var arguments_ = [];
    var thisProgram = "./this.program";
    var quit_ = (status, toThrow) => {
      throw toThrow;
    };
    var scriptDirectory = "";
    var readAsync, readBinary;
    if (ENVIRONMENT_IS_WEB || ENVIRONMENT_IS_WORKER) {
      if (ENVIRONMENT_IS_WORKER) {
        scriptDirectory = self.location.href;
      } else if (typeof document != "undefined" && document.currentScript) {
        scriptDirectory = document.currentScript.src;
      }
      if (_scriptName) {
        scriptDirectory = _scriptName;
      }
      if (scriptDirectory.startsWith("blob:")) {
        scriptDirectory = "";
      } else {
        scriptDirectory = scriptDirectory.substr(0, scriptDirectory.replace(/[?#].*/, "").lastIndexOf("/") + 1);
      }
      readAsync = url => fetch(url, {
        credentials: "same-origin"
      }).then(response => {
        if (response.ok) {
          return response.arrayBuffer();
        }
        return Promise.reject(new Error(response.status + " : " + response.url));
      });
    } else {}
    var out = Module["print"] || console.log.bind(console);
    var err = Module["printErr"] || console.error.bind(console);
    Object.assign(Module, moduleOverrides);
    moduleOverrides = null;
    if (Module["arguments"]) arguments_ = Module["arguments"];
    if (Module["thisProgram"]) thisProgram = Module["thisProgram"];
    var wasmBinary = Module["wasmBinary"];
    function intArrayFromBase64(s) {
      var decoded = atob(s);
      var bytes = new Uint8Array(decoded.length);
      for (var i = 0; i < decoded.length; ++i) {
        bytes[i] = decoded.charCodeAt(i);
      }
      return bytes;
    }
    function tryParseAsDataURI(filename) {
      if (!isDataURI(filename)) {
        return;
      }
      return intArrayFromBase64(filename.slice(dataURIPrefix.length));
    }
    var wasmMemory;
    var ABORT = false;
    var EXITSTATUS;
    var HEAP8, HEAPU8, HEAP16, HEAPU16, HEAP32, HEAPU32, HEAPF32, HEAPF64;
    function updateMemoryViews() {
      var b = wasmMemory.buffer;
      Module["HEAP8"] = HEAP8 = new Int8Array(b);
      Module["HEAP16"] = HEAP16 = new Int16Array(b);
      Module["HEAPU8"] = HEAPU8 = new Uint8Array(b);
      Module["HEAPU16"] = HEAPU16 = new Uint16Array(b);
      Module["HEAP32"] = HEAP32 = new Int32Array(b);
      Module["HEAPU32"] = HEAPU32 = new Uint32Array(b);
      Module["HEAPF32"] = HEAPF32 = new Float32Array(b);
      Module["HEAPF64"] = HEAPF64 = new Float64Array(b);
    }
    var __ATPRERUN__ = [];
    var __ATINIT__ = [];
    var __ATPOSTRUN__ = [];
    var runtimeInitialized = false;
    function preRun() {
      if (Module["preRun"]) {
        if (typeof Module["preRun"] == "function") Module["preRun"] = [Module["preRun"]];
        while (Module["preRun"].length) {
          addOnPreRun(Module["preRun"].shift());
        }
      }
      callRuntimeCallbacks(__ATPRERUN__);
    }
    function initRuntime() {
      runtimeInitialized = true;
      callRuntimeCallbacks(__ATINIT__);
    }
    function postRun() {
      if (Module["postRun"]) {
        if (typeof Module["postRun"] == "function") Module["postRun"] = [Module["postRun"]];
        while (Module["postRun"].length) {
          addOnPostRun(Module["postRun"].shift());
        }
      }
      callRuntimeCallbacks(__ATPOSTRUN__);
    }
    function addOnPreRun(cb) {
      __ATPRERUN__.unshift(cb);
    }
    function addOnInit(cb) {
      __ATINIT__.unshift(cb);
    }
    function addOnPostRun(cb) {
      __ATPOSTRUN__.unshift(cb);
    }
    var runDependencies = 0;
    var runDependencyWatcher = null;
    var dependenciesFulfilled = null;
    function addRunDependency(id) {
      runDependencies++;
      Module["monitorRunDependencies"]?.(runDependencies);
    }
    function removeRunDependency(id) {
      runDependencies--;
      Module["monitorRunDependencies"]?.(runDependencies);
      if (runDependencies == 0) {
        if (runDependencyWatcher !== null) {
          clearInterval(runDependencyWatcher);
          runDependencyWatcher = null;
        }
        if (dependenciesFulfilled) {
          var callback = dependenciesFulfilled;
          dependenciesFulfilled = null;
          callback();
        }
      }
    }
    function abort(what) {
      Module["onAbort"]?.(what);
      what = "Aborted(" + what + ")";
      err(what);
      ABORT = true;
      what += ". Build with -sASSERTIONS for more info.";
      var e = new WebAssembly.RuntimeError(what);
      readyPromiseReject(e);
      throw e;
    }
    var dataURIPrefix = "data:application/octet-stream;base64,";
    var isDataURI = filename => filename.startsWith(dataURIPrefix);
    function findWasmBinary() {
      var f = "data:application/octet-stream;base64,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A3FFBEACQCAAQQNxRQRAIAAhAgwBCyACRQRAIAAhAgwBCyAAIQIDQCACIAEtAAA6AAAgAUEBaiEBIAJBAWoiAkEDcUUNASACIANJDQALCyADQXxxIQQCQCADQcAASQ0AIAIgBEFAaiIFSw0AA0AgAiABKAIANgIAIAIgASgCBDYCBCACIAEoAgg2AgggAiABKAIMNgIMIAIgASgCEDYCECACIAEoAhQ2AhQgAiABKAIYNgIYIAIgASgCHDYCHCACIAEoAiA2AiAgAiABKAIkNgIkIAIgASgCKDYCKCACIAEoAiw2AiwgAiABKAIwNgIwIAIgASgCNDYCNCACIAEoAjg2AjggAiABKAI8NgI8IAFBQGshASACQUBrIgIgBU0NAAsLIAIgBE8NAQNAIAIgASgCADYCACABQQRqIQEgAkEEaiICIARJDQALDAELIANBBEkEQCAAIQIMAQsgA0EEayIEIABJBEAgACECDAELIAAhAgNAIAIgAS0AADoAACACIAEtAAE6AAEgAiABLQACOgACIAIgAS0AAzoAAyABQQRqIQEgAkEEaiICIARNDQALCyACIANJBEADQCACIAEtAAA6AAAgAUEBaiEBIAJBAWoiAiADRw0ACwsgAAswAQF/AkAgAEUNACABRQ0AQQggACABbCIBECkiAARAIABBACABEBkaCyAAIQILIAILEQAgAEUEQEEADwtBCCAAECkL8gICAn8BfgJAIAJFDQAgACABOgAAIAAgAmoiA0EBayABOgAAIAJBA0kNACAAIAE6AAIgACABOgABIANBA2sgAToAACADQQJrIAE6AAAgAkEHSQ0AIAAgAToAAyADQQRrIAE6AAAgAkEJSQ0AIABBACAAa0EDcSIEaiIDIAFB/wFxQYGChAhsIgE2AgAgAyACIARrQXxxIgRqIgJBBGsgATYCACAEQQlJDQAgAyABNgIIIAMgATYCBCACQQhrIAE2AgAgAkEMayABNgIAIARBGUkNACADIAE2AhggAyABNgIUIAMgATYCECADIAE2AgwgAkEQayABNgIAIAJBFGsgATYCACACQRhrIAE2AgAgAkEcayABNgIAIAQgA0EEcUEYciIEayICQSBJDQAgAa1CgYCAgBB+IQUgAyAEaiEBA0AgASAFNwMYIAEgBTcDECABIAU3AwggASAFNwMAIAFBIGohASAC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CyABQf8PTQRAIAAgAUE/cUGAAXI6AAEgACABQQZ2QcABcjoAAEECDwsgAUGAQHFBgMADRyABQYCwA09xRQRAIAAgAUE/cUGAAXI6AAIgACABQQx2QeABcjoAACAAIAFBBnZBP3FBgAFyOgABQQMPCyABQYCABGtB//8/TQRAIAAgAUE/cUGAAXI6AAMgACABQRJ2QfABcjoAACAAIAFBBnZBP3FBgAFyOgACIAAgAUEMdkE/cUGAAXI6AAFBBA8LC0HUzQFBGTYCAEF/DwsgACABOgAAQQELfgIBfwF+IAC9IgNCNIinQf8PcSICQf8PRwR8IAJFBEAgASAARAAAAAAAAAAAYQR/QQAFIABEAAAAAAAA8EOiIAEQcCEAIAEoAgBBQGoLNgIAIAAPCyABIAJB/gdrNgIAIANC/////////4eAf4NCgICAgICAgPA/hL8FIAALC7wCAAJAAkACQAJAAkACQAJAAkACQAJAAkAgAUEJaw4SAAgJCggJAQIDBAoJCgoICQUGBwsgAiACKAIAIgFBBGo2AgAgACABKAIANgIADwsgAiACKAIAIgFBBGo2AgAgACABMgEANwMADwsgAiACKAIAIgFBBGo2AgAgACABMwEANwMADwsgAiACKAIAIgFBBGo2AgAgACABMAAANwMADwsgAiACKAIAIgFBBGo2AgAgACABMQAANwMADwsgAiACKAIAQQdqQXhxIgFBCGo2AgAgACABKwMAOQMADwsgACACIAMRAwALDwsgAiACKAIAIgFBBGo2AgAgACABNAIANwMADwsgAiACKAIAIgFBBGo2AgAgACABNQIANwMADwsgAiACKAIAQQdqQXhxIgFBCGo2AgAgACABKQMANwMAC28BBX8gACgCACIDLAAAQTBrIgFBCUsEQEEADwsDQEF/IQQgAkHMmbPmAE0EQEF/IAEgAkEKbCIFaiABIAVB/////wdzSxshBAsgACADQQFqIgU2AgAgAywAASAEIQIgBSEDQTBrIgFBCkkNAAsgAgtJAQF/AkBBAUEsEBciAQRAIAFBADYCEAJAIABBAEwEQCABQQFBCBAXIgA2AiQgAEUNAQwDCyABQQA2AgwLIAEQFAtBACEBCyABC64UAhJ/An4jAEFAaiIIJAAgCCABNgI8IAhBJ2ohFyAIQShqIRICQAJAAkACQANAQQAhBwNA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KAIcIgdrIgYgBCAGSRsiBgRAIAUgByAGEBYaIAMgAygCACAGaiIFNgIAIAMgAygCBCAGayIENgIECyAEIAIgAiAESxsiBARAIAUgASAEEBYaIAMgAygCACAEaiIFNgIAIAMgAygCBCAEazYCBAsgBUEAOgAAIAAgACgCLCIBNgIcIAAgATYCFCACC6YFAgZ+BH8gASABKAIAQQdqQXhxIgFBEGo2AgAgACABKQMAIQIgASkDCCEHIwBBIGsiCCQAIAdC////////P4MhBAJ+IAdCMIhC//8BgyIDpyIKQYH4AGtB/Q9NBEAgBEIEhiACQjyIhCEDIApBgPgAa60hBAJAIAJC//////////8PgyICQoGAgICAgICACFoEQCADQgF8IQMMAQsgAkKAgICAgICAgAhSDQAgA0IBgyADfCEDC0IAIAMgA0L/////////B1YiABshAiAArSAEfAwBCwJAIAIgBIRQDQAgA0L//wFSDQAgBEIEhiACQjyIhEKAgICAgICABIQhAkL/DwwBCyAKQf6HAUsEQEIAIQJC/w8MAQtBgPgAQYH4ACADUCIBGyIAIAprIglB8ABKBEBCACECQgAMAQsgAiEDIAQgBEKAgICAgIDAAIQgARsiBSEGAkBBgAEgCWsiAUHAAHEEQCACIAFBQGqthiEGQgAhAwwBCyABRQ0AIAYgAa0iBIYgA0HAACABa62IhCEGIAMgBIYhAwsgCCADNwMQIAggBjcDGAJAIAlBwABxBEAgBSAJQUBqrYghAkIAIQUMAQsgCUUNACAFQcAAIAlrrYYgAiAJrSIDiIQhAiAFIAOIIQULIAggAjcDACAIIAU3AwggCCkDCEIEhiAIKQMAIgNCPIiEIQICQCAAIApHIAgpAxAgCCkDGIRCAFJxrSADQv//////////D4OEIgNCgYCAgICAgIAIWgRAIAJCAXwhAgwBCyADQoCAgICAgICACFINACACQgGDIAJ8IQILIAJCgICAgICAgAiFIAIgAkL/////////B1YiABshAiAArQshAyAIQSBqJAAgB0KAgICAgICAgIB/gyADQjSGhCAChL85AwAL9BcDEn8BfAN+IwBBsARrIgwkACAMQQA2AiwCQCABvSIZQgBTBEBBASEQQboIIRQgAZoiAb0hGQwBCyAEQYAQcQRAQQEhEEG9CCEUDAELQcAI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ACgCRCICQQN0aiIFIAQoAhQ7AQAgBSAEKAIQNgIEQQEhBSAAIAJBAWo2AkQgASAJaiECIAZBAWoiBiAKRw0ACwwCC0EBIAYgBkEBTRshByAAKAJEIQFBACEGA0AgBCABNgIUIAEgACgChAEgACgCgAFsTw0BIAIgBEEQaiAJEBUgCCAAKAJEIgpBA3RqIgUgATsBACAFIAQoAhA2AgRBASEFIAAgCkEBaiIBNgJEIAIgCWohAiAGQQFqIgYgB0cNAAsMAQsgAEEBNgJMIAQgATYCACADQQJB0jwgBBATCyAEQSBqJAAgBQsEAEEACwvLwQEhAEGACAvgmQFjYW5ub3QgYWxsb2NhdGUgb3BqX3RjZF9zZWdfZGF0YV9jaHVua190KiBhcnJheQAtKyAgIDBYMHgALTBYKzBYIDBYLTB4KzB4IDB4AFVua25vd24gZm9ybWF0AEZhaWxlZCB0byBzZXR1cCB0aGUgZGVjb2RlcgBGYWlsZWQgdG8gcmVhZCB0aGUgaGVhZGVyAG5hbgAqbF90aWxlX2xlbiA+IFVJTlRfTUFYIC0gT1BKX0NPTU1PTl9DQkxLX0RBVEFfRVhUUkEgLSBwX2oyay0+bV9zcGVjaWZpY19wYXJhbS5tX2RlY29kZXIubV9zb3RfbGVuZ3RoAGluZgBGYWlsZWQgdG8gZGVjb2RlIHRoZSBpbWFnZQBJbnZhbGlkIGFjY2VzcyB0byBwaS0+aW5jbHVkZQAvdG1wL29wZW5qcGVnL3NyYy9iaW4vY29tbW9uL2NvbG9yLmMAQUxMX0NQVVMAT1BKX05VTV9USFJFQURTAE5BTgBPSlBfRE9fTk9UX0RJU1BMQVlfVElMRV9JTkRFWF9JRl9UTE0ASU5GAHBfajJrLT5tX3NwZWNpZmljX3BhcmFtLm1fZGVjb2Rlci5tX3NvdF9sZW5ndGggPiBVSU5UX01BWCAtIE9QSl9DT01NT05fQ0JMS19EQVRBX0VYVFJBAAkJCSBwcmVjY2ludHNpemUgKHcsaCk9AAkJCSBzdGVwc2l6ZXMgKG0sZSk9AFNPVCBtYXJrZXIgZm9yIHRpbGUgJXUgZGVjbGFyZXMgbW9yZSB0aWxlLXBhcnRzIHRoYW4gZm91bmQgaW4gVExNIG1hcmtlci4AKG51bGwpACglZCwlZCkgACVzfQoACQkgfQoAW0RFVl0gRHVtcCBhbiBpbWFnZV9jb21wX2hlYWRlciBzdHJ1Y3QgewoAW0RFVl0gRHVtcCBhbiBpbWFnZV9oZWFkZXIgc3RydWN0IHsKAEltYWdlIGluZm8gewoACSBkZWZhdWx0IHRpbGUgewoAJXMJIGNvbXBvbmVudCAlZCB7CgAJCSBjb21wICVkIHsKAAkgVGlsZSBpbmRleDogewoACSBNYXJrZXIgbGlzdDogewoAQ29kZXN0cmVhbSBpbmRleCBmcm9tIG1haW4gaGVhZGVyOiB7CgBDb2Rlc3RyZWFtIGluZm8gZnJvbSBtYWluIGhlYWRlcjogewoAU3RyZWFtIGVycm9yIHdoaWxlIHJlYWRpbmcgSlAyIEhlYWRlciBib3gKAEZvdW5kIGEgbWlzcGxhY2VkICclYyVjJWMlYycgYm94IG91dHNpZGUganAyaCBib3gKAE1hbGZvcm1lZCBKUDIgZmlsZSBmb3JtYXQ6IGZpcnN0IGJveCBtdXN0IGJlIEpQRUcgMjAwMCBzaWduYXR1cmUgYm94CgBNYWxmb3JtZWQgSlAyIGZpbGUgZm9ybWF0OiBzZWNvbmQgYm94IG11c3QgYmUgZmlsZSB0eXBlIGJveAoATm90IGVub3VnaCBtZW1vcnkgdG8gaGFuZGxlIGpwZWcyMDAwIGJveAoATm90IGVub3VnaCBtZW1vcnkgd2l0aCBGVFlQIEJveAoAQSBtYXJrZXIgSUQgd2FzIGV4cGVjdGVkICgweGZmLS0pIGluc3RlYWQgb2YgJS44eAoACQkgbWN0PSV4CgAJCQkgY2Jsa3N0eT0lI3gKAAkJCSBjc3R5PSUjeAoACQkgcHJnPSUjeAoASW50ZWdlciBvdmVyZmxvdwoACSB0ZHg9JXUsIHRkeT0ldQoACSB0dz0ldSwgdGg9JXUKAAkgdHgwPSV1LCB0eTA9JXUKAEludmFsaWQgY29tcG9uZW50IGluZGV4OiAldQoAU3RyZWFtIHRvbyBzaG9ydAoATWFya2VyIGhhbmRsZXIgZnVuY3Rpb24gZmFpbGVkIHRvIHJlYWQgdGhlIG1hcmtlciBzZWdtZW50CgBOb3QgZW5vdWdoIG1lbW9yeSBmb3IgY3VycmVudCBwcmVjaW5jdCBjb2RlYmxvY2sgZWxlbWVudAoARXJyb3IgcmVhZGluZyBTUENvZCBTUENvYyBlbGVtZW50CgBFcnJvciByZWFkaW5nIFNRY2Qgb3IgU1FjYyBlbGVtZW50CgBBIEJQQ0MgaGVhZGVyIGJveCBpcyBhdmFpbGFibGUgYWx0aG91Z2ggQlBDIGdpdmVuIGJ5IHRoZSBJSERSIGJveCAoJWQpIGluZGljYXRlIGNvbXBvbmVudHMgYml0IGRlcHRoIGlzIGNvbnN0YW50CgBFcnJvciB3aXRoIFNJWiBtYXJrZXI6IGlsbGVnYWwgdGlsZSBvZmZzZXQKAEludmFsaWQgcHJlY2luY3QKAE5vdCBlbm91Z2ggbWVtb3J5IHRvIGhhbmRsZSBiYW5kIHByZWNpbnRzCgBGYWlsZWQgdG8gZGVjb2RlIGFsbCB1c2VkIGNvbXBvbmVudHMKAFNpemUgb2YgY29kZSBibG9jayBkYXRhIGV4Y2VlZHMgc3lzdGVtIGxpbWl0cwoAU2l6ZSBvZiB0aWxlIGRhdGEgZXhjZWVkcyBzeXN0ZW0gbGltaXRzCgBDYW5ub3QgdGFrZSBpbiBjaGFyZ2UgbXVsdGlwbGUgTUNUIG1hcmtlcnMKAENvcnJ1cHRlZCBQUE0gbWFya2VycwoATm90IGVub3VnaCBtZW1vcnkgZm9yIHRpbGUgcmVzb2x1dGlvbnMKAENhbm5vdCB0YWtlIGluIGNoYXJnZSBtdWx0aXBsZSBjb2xsZWN0aW9ucwoASW52YWxpZCBQQ0xSIGJveC4gUmVwb3J0cyAwIHBhbGV0dGUgY29sdW1ucwoAV2UgZG8gbm90IHN1cHBvcnQgUk9JIGluIGRlY29kaW5nIEhUIGNvZGVibG9ja3MKAENhbm5vdCBoYW5kbGUgYm94IG9mIHVuZGVmaW5lZCBzaXplcwoAQ2Fubm90IHRha2UgaW4gY2hhcmdlIGNvbGxlY3Rpb25zIHdpdGhvdXQgc2FtZSBudW1iZXIgb2YgaW5kaXhlcwoASW52YWxpZCB0aWxlYy0+d2luX3h4eCB2YWx1ZXMKAENhbm5vdCBoYW5kbGUgYm94IG9mIGxlc3MgdGhhbiA4IGJ5dGVzCgBDYW5ub3QgaGFuZGxlIFhMIGJveCBvZiBsZXNzIHRoYW4gMTYgYnl0ZXMKAENvbXBvbmVudCBpbmRleCAldSB1c2VkIHNldmVyYWwgdGltZXMKAEludmFsaWQgUENMUiBib3guIFJlcG9ydHMgJWQgZW50cmllcwoATm90IGVub3VnaCBtZW1vcnkgdG8gY3JlYXRlIFRhZy10cmVlIG5vZGVzCgBDYW5ub3QgdGFrZSBpbiBjaGFyZ2UgbWN0IGRhdGEgd2l0aGluIG11bHRpcGxlIE1DVCByZWNvcmRzCgBDYW5ub3QgZGVjb2RlIHRpbGUsIG1lbW9yeSBlcnJvcgoAb3BqX2oya19hcHBseV9uYl90aWxlX3BhcnRzX2NvcnJlY3Rpb24gZXJyb3IKAFByb2JsZW0gd2l0aCBza2lwcGluZyBKUEVHMjAwMCBib3gsIHN0cmVhbSBlcnJvcgoAUHJvYmxlbSB3aXRoIHJlYWRpbmcgSlBFRzIwMDAgYm94LCBzdHJlYW0gZXJyb3IKAFVua25vd24gbWFya2VyCgBOb3QgZW5vdWdoIG1lbW9yeSB0byBhZGQgdGwgbWFya2VyCgBOb3QgZW5vdWdoIG1lbW9yeSB0byBhZGQgbWggbWFya2VyCgBOb3QgZW5vdWdoIG1lbW9yeSB0byB0YWtlIGluIGNoYXJnZSBTSVogbWFya2VyCgBFcnJvciByZWFkaW5nIFBQVCBtYXJrZXIKAE5vdCBlbm91Z2ggbWVtb3J5IHRvIHJlYWQgUFBUIG1hcmtlcgoARXJyb3IgcmVhZGluZyBTT1QgbWFya2VyCgBEaWQgbm90IGdldCBleHBlY3RlZCBTT1QgbWFya2VyCgBFcnJvciByZWFkaW5nIFBMVCBtYXJrZXIKAEVycm9yIHJlYWRpbmcgTUNUIG1hcmtlcgoATm90IGVub3VnaCBtZW1vcnkgdG8gcmVhZCBNQ1QgbWFya2VyCgBOb3QgZW5vdWdoIHNwYWNlIGZvciBleHBlY3RlZCBTT1AgbWFya2VyCgBFeHBlY3RlZCBTT1AgbWFya2VyCgBFcnJvciByZWFkaW5nIE1DTyBtYXJrZXIKAEVycm9yIHJlYWRpbmcgUkdOIG1hcmtlcgoARXJyb3IgcmVhZGluZyBQUE0gbWFya2VyCgBOb3QgZW5vdWdoIG1lbW9yeSB0byByZWFkIFBQTSBtYXJrZXIKAEVycm9yIHJlYWRpbmcgUExNIG1hcmtlcgoARXhwZWN0ZWQgRVBIIG1hcmtlcgoATm90IGVub3VnaCBzcGFjZSBmb3IgcmVxdWlyZWQgRVBIIG1hcmtlcgoARXJyb3IgcmVhZGluZyBDUkcgbWFya2VyCgBVbmtub3duIHByb2dyZXNzaW9uIG9yZGVyIGluIENPRCBtYXJrZXIKAFVua25vd24gU2NvZCB2YWx1ZSBpbiBDT0QgbWFya2VyCgBFcnJvciByZWFkaW5nIENPRCBtYXJrZXIKAEVycm9yIHJlYWRpbmcgUUNEIG1hcmtlcgoAQ3Jyb3IgcmVhZGluZyBDQkQgbWFya2VyCgBFcnJvciByZWFkaW5nIFBPQyBtYXJrZXIKAEVycm9yIHJlYWRpbmcgQ09DIG1hcmtlcgoARXJyb3IgcmVhZGluZyBRQ0MgbWFya2VyCgBFcnJvciByZWFkaW5nIE1DQyBtYXJrZXIKAE5vdCBlbm91Z2ggbWVtb3J5IHRvIHJlYWQgTUNDIG1hcmtlcgoAcmVxdWlyZWQgU0laIG1hcmtlciBub3QgZm91bmQgaW4gbWFpbiBoZWFkZXIKAHJlcXVpcmVkIENPRCBtYXJrZXIgbm90IGZvdW5kIGluIG1haW4gaGVhZGVyCgByZXF1aXJlZCBRQ0QgbWFya2VyIG5vdCBmb3VuZCBpbiBtYWluIGhlYWRlcgoATm90IGVub3VnaCBtZW1vcnkgdG8gaGFuZGxlIGpwZWcyMDAwIGZpbGUgaGVhZGVyCgBOb3QgZW5vdWdoIG1lbW9yeSB0byByZWFkIGhlYWRlcgoARXJyb3Igd2l0aCBKUCBTaWduYXR1cmUgOiBiYWQgbWFnaWMgbnVtYmVyCgBJbiBTT1QgbWFya2VyLCBUUFNvdCAoJWQpIGlzIG5vdCB2YWxpZCByZWdhcmRzIHRvIHRoZSBjdXJyZW50IG51bWJlciBvZiB0aWxlLXBhcnQgKCVkKSwgZ2l2aW5nIHVwCgBJbiBTT1QgbWFya2VyLCBUUFNvdCAoJWQpIGlzIG5vdCB2YWxpZCByZWdhcmRzIHRvIHRoZSBwcmV2aW91cyBudW1iZXIgb2YgdGlsZS1wYXJ0ICglZCksIGdpdmluZyB1cAoASW4gU09UIG1hcmtlciwgVFBTb3QgKCVkKSBpcyBub3QgdmFsaWQgcmVnYXJkcyB0byB0aGUgY3VycmVudCBudW1iZXIgb2YgdGlsZS1wYXJ0IChoZWFkZXIpICglZCksIGdpdmluZyB1cAoAdGlsZXMgcmVxdWlyZSBhdCBsZWFzdCBvbmUgcmVzb2x1dGlvbgoATWFya2VyIGlzIG5vdCBjb21wbGlhbnQgd2l0aCBpdHMgcG9zaXRpb24KAFByb2JsZW0gd2l0aCBzZWVrIGZ1bmN0aW9uCgBFcnJvciByZWFkaW5nIFNQQ29kIFNQQ29jIGVsZW1lbnQsIEludmFsaWQgY2Jsa3cvY2Jsa2ggY29tYmluYXRpb24KAEludmFsaWQgbXVsdGlwbGUgY29tcG9uZW50IHRyYW5zZm9ybWF0aW9uCgBDYW5ub3QgdGFrZSBpbiBjaGFyZ2UgY29sbGVjdGlvbnMgb3RoZXIgdGhhbiBhcnJheSBkZWNvcnJlbGF0aW9uCgBUb28gbGFyZ2UgdmFsdWUgZm9yIE5wcG0KAE5vdCBlbm91Z2ggYnl0ZXMgdG8gcmVhZCBOcHBtCgBiYWQgcGxhY2VkIGpwZWcgY29kZXN0cmVhbQoACSBNYWluIGhlYWRlciBzdGFydCBwb3NpdGlvbj0lbGxpCgkgTWFpbiBoZWFkZXIgZW5kIHBvc2l0aW9uPSVsbGkKAE1hcmtlciBzaXplIGluY29uc2lzdGVudCB3aXRoIHN0cmVhbSBsZW5ndGgKAFRpbGUgcGFydCBsZW5ndGggc2l6ZSBpbmNvbnNpc3RlbnQgd2l0aCBzdHJlYW0gbGVuZ3RoCgBDYW5ub3QgdGFrZSBpbiBjaGFyZ2UgbXVsdGlwbGUgZGF0YSBzcGFubmluZwoAV3JvbmcgZmxhZwoARXJyb3Igd2l0aCBGVFlQIHNpZ25hdHVyZSBCb3ggc2l6ZQoARXJyb3Igd2l0aCBKUCBzaWduYXR1cmUgQm94IHNpemUKAEludmFsaWQgcHJlY2luY3Qgc2l6ZQoASW5jb25zaXN0ZW50IG1hcmtlciBzaXplCgBJbnZhbGlkIG1hcmtlciBzaXplCgBFcnJvciB3aXRoIFNJWiBtYXJrZXIgc2l6ZQoATm90IGVub3VnaCBtZW1vcnkgdG8gYWRkIGEgbmV3IHZhbGlkYXRpb24gcHJvY2VkdXJlCgBOb3QgZW5vdWdoIG1lbW9yeSB0byBkZWNvZGUgdGlsZQoARmFpbGVkIHRvIGRlY29kZSB0aGUgY29kZXN0cmVhbSBpbiB0aGUgSlAyIGZpbGUKAENhbm5vdCB0YWtlIGluIGNoYXJnZSBjb2xsZWN0aW9ucyB3aXRoIGluZGl4IHNodWZmbGUKAENhbm5vdCBhbGxvY2F0ZSBUaWVyIDEgaGFuZGxlCgBObyBkZWNvZGVkIGFyZWEgcGFyYW1ldGVycywgc2V0IHRoZSBkZWNvZGVkIGFyZWEgdG8gdGhlIHdob2xlIGltYWdlCgBOb3QgZW5vdWdoIG1lbW9yeSB0byBjcmVhdGUgVGFnLXRyZWUKAE5vdCBlbm91Z2ggbWVtb3J5IHRvIHJlaW5pdGlhbGl6ZSB0aGUgdGFnIHRyZWUKAEVycm9yIHJlYWRpbmcgU1BDb2QgU1BDb2MgZWxlbWVudCwgSW52YWxpZCB0cmFuc2Zvcm1hdGlvbiBmb3VuZAoARXJyb3IgcmVhZGluZyBTUENvZCBTUENvYyBlbGVtZW50LiBVbnN1cHBvcnRlZCBNaXhlZCBIVCBjb2RlLWJsb2NrIHN0eWxlIGZvdW5kCgBUaWxlIFkgY29vcmRpbmF0ZXMgYXJlIG5vdCBzdXBwb3J0ZWQKAFRpbGUgWCBjb29yZGluYXRlcyBhcmUgbm90IHN1cHBvcnRlZAoASW1hZ2UgY29vcmRpbmF0ZXMgYWJvdmUgSU5UX01BWCBhcmUgbm90IHN1cHBvcnRlZAoASlBFRzIwMDAgSGVhZGVyIGJveCBub3QgcmVhZCB5ZXQsICclYyVjJWMlYycgYm94IHdpbGwgYmUgaWdub3JlZAoAb3BqX2oya19tZXJnZV9wcHQoKSBoYXMgYWxyZWFkeSBiZWVuIGNhbGxlZAoAb3BqX2oya19idWlsZF90cF9pbmRleF9mcm9tX3RsbSgpOiB0aWxlIGluZGV4IGFsbG9jYXRpb24gZmFpbGVkCgBOb3QgZW5vdWdoIG1lbW9yeSB0byByZWFkIFNPVCBtYXJrZXIuIFRpbGUgaW5kZXggYWxsb2NhdGlvbiBmYWlsZWQKAElnbm9yaW5nIGloZHIgYm94LiBGaXJzdCBpaGRyIGJveCBhbHJlYWR5IHJlYWQKAFpwcHQgJXUgYWxyZWFkeSByZWFkCgBacHBtICV1IGFscmVhZHkgcmVhZAoAUFRFUk0gY2hlY2sgZmFpbHVyZTogJWQgc3ludGhlc2l6ZWQgMHhGRiBtYXJrZXJzIHJlYWQKAAkJCSBjYmxrdz0yXiVkCgAJCQkgY2Jsa2g9Ml4lZAoACQkJIHFudHN0eT0lZAoAJXMgZHg9JWQsIGR5PSVkCgAJCQkgcm9pc2hpZnQ9JWQKAAkJCSBudW1nYml0cz0lZAoACQkgbnVtbGF5ZXJzPSVkCgAlcyBudW1jb21wcz0lZAoAb3BqX2pwMl9hcHBseV9jZGVmOiBhY249JWQsIG51bWNvbXBzPSVkCgBvcGpfanAyX2FwcGx5X2NkZWY6IGNuPSVkLCBudW1jb21wcz0lZAoACQkJIG51bXJlc29sdXRpb25zPSVkCgAJCSB0eXBlPSUjeCwgcG9zPSVsbGksIGxlbj0lZAoAJXMgc2duZD0lZAoACQkJIHFtZmJpZD0lZAoAJXMgcHJlYz0lZAoACQkgbmIgb2YgdGlsZS1wYXJ0IGluIHRpbGUgWyVkXT0lZAoAJXMgeDE9JWQsIHkxPSVkCgAlcyB4MD0lZCwgeTA9JWQKAEZhaWxlZCB0byBkZWNvZGUgdGlsZSAlZC8lZAoAU2V0dGluZyBkZWNvZGluZyBhcmVhIHRvICVkLCVkLCVkLCVkCgBGYWlsZWQgdG8gZGVjb2RlIGNvbXBvbmVudCAlZAoASW52YWxpZCB2YWx1ZSBmb3IgbnVtcmVzb2x1dGlvbnMgOiAlZCwgbWF4IHZhbHVlIGlzIHNldCBpbiBvcGVuanBlZy5oIGF0ICVkCgBJbnZhbGlkIGNvbXBvbmVudCBudW1iZXI6ICVkLCByZWdhcmRpbmcgdGhlIG51bWJlciBvZiBjb21wb25lbnRzICVkCgBUb28gbWFueSBQT0NzICVkCgBvcGpfajJrX3JlYWRfdGxtKCk6IGludmFsaWQgdGlsZSBudW1iZXIgJWQKAEludmFsaWQgdGlsZSBudW1iZXIgJWQKAEludmFsaWQgdGlsZSBwYXJ0IGluZGV4IGZvciB0aWxlIG51bWJlciAlZC4gR290ICVkLCBleHBlY3RlZCAlZAoARXJyb3Igd2l0aCBTSVogbWFya2VyOiBudW1iZXIgb2YgY29tcG9uZW50IGlzIGlsbGVnYWwgLT4gJWQKAE5vdCBlbm91Z2ggbWVtb3J5IGZvciBjaWVsYWIKAENhbm5vdCBhbGxvY2F0ZSBjYmxrLT5kZWNvZGVkX2RhdGEKAEZhaWxlZCB0byBtZXJnZSBQUFQgZGF0YQoARmFpbGVkIHRvIG1lcmdlIFBQTSBkYXRhCgBJbnZhbGlkIG51bWJlciBvZiBsYXllcnMgaW4gQ09EIG1hcmtlciA6ICVkIG5vdCBpbiByYW5nZSBbMS02NTUzNV0KACVzOiVkOmNvbG9yX2NteWtfdG9fcmdiCglDQU4gTk9UIENPTlZFUlQKACVzOiVkOmNvbG9yX2VzeWNjX3RvX3JnYgoJQ0FOIE5PVCBDT05WRVJUCgAlczolZDpjb2xvcl9zeWNjX3RvX3JnYgoJQ0FOIE5PVCBDT05WRVJUCgBTdHJlYW0gdG9vIHNob3J0LCBleHBlY3RlZCBTT1QKAFVuYWJsZSB0byBzZXQgdDEgaGFuZGxlIGFzIFRMUwoAU290IGxlbmd0aCBpcyBsZXNzIHRoYW4gbWFya2VyIHNpemUgKyBtYXJrZXIgSUQKAFN0cmVhbSBkb2VzIG5vdCBlbmQgd2l0aCBFT0MKAENhbm5vdCBoYW5kbGUgYm94IHNpemVzIGhpZ2hlciB0aGFuIDJeMzIKAG9wal9waV9uZXh0X2xyY3AoKTogaW52YWxpZCBjb21wbm8wL2NvbXBubzEKAG9wal9waV9uZXh0X3JsY3AoKTogaW52YWxpZCBjb21wbm8wL2NvbXBubzEKAG9wal9waV9uZXh0X2NwcmwoKTogaW52YWxpZCBjb21wbm8wL2NvbXBubzEKAG9wal9waV9uZXh0X3BjcmwoKTogaW52YWxpZCBjb21wbm8wL2NvbXBubzEKAG9wal9waV9uZXh0X3JwY2woKTogaW52YWxpZCBjb21wbm8wL2NvbXBubzEKAG9wal90MV9kZWNvZGVfY2JsaygpOiB1bnN1cHBvcnRlZCBicG5vX3BsdXNfb25lID0gJWQgPj0gMzEKAEZhaWxlZCB0byBkZWNvZGUgdGlsZSAxLzEKAEluc3VmZmljaWVudCBkYXRhIGZvciBDTUFQIGJveC4KAE5lZWQgdG8gcmVhZCBhIFBDTFIgYm94IGJlZm9yZSB0aGUgQ01BUCBib3guCgBJbnN1ZmZpY2llbnQgZGF0YSBmb3IgQ0RFRiBib3guCgBOdW1iZXIgb2YgY2hhbm5lbCBkZXNjcmlwdGlvbiBpcyBlcXVhbCB0byB6ZXJvIGluIENERUYgYm94LgoAU3RyZWFtIGVycm9yIHdoaWxlIHJlYWRpbmcgSlAyIEhlYWRlciBib3g6IG5vICdpaGRyJyBib3guCgBOb24gY29uZm9ybWFudCBjb2Rlc3RyZWFtIFRQc290PT1UTnNvdC4KAFN0cmVhbSBlcnJvciB3aGlsZSByZWFkaW5nIEpQMiBIZWFkZXIgYm94OiBib3ggbGVuZ3RoIGlzIGluY29uc2lzdGVudC4KAEJveCBsZW5ndGggaXMgaW5jb25zaXN0ZW50LgoAUmVzb2x1dGlvbiBmYWN0b3IgaXMgZ3JlYXRlciB0aGFuIHRoZSBtYXhpbXVtIHJlc29sdXRpb24gaW4gdGhlIGNvbXBvbmVudC4KAENvbXBvbmVudCBtYXBwaW5nIHNlZW1zIHdyb25nLiBUcnlpbmcgdG8gY29ycmVjdC4KAG9wal9qMmtfYnVpbGRfdHBfaW5kZXhfZnJvbV90bG0oKTogdGlsZSAlZCBoYXMgbm8gcmVnaXN0ZXJlZCB0aWxlLXBhcnQgaW4gVExNIG1hcmtlciBzZWdtZW50cy4KAG9wal9qMmtfcmVhZF90bG0oKTogdG9vIG1hbnkgVExNIG1hcmtlcnMuCgBvcGpfajJrX3JlYWRfdGxtKCk6IGNhbm5vdCBhbGxvY2F0ZSBtX3RpbGVfcGFydF9pbmZvcy4KAEluY29tcGxldGUgY2hhbm5lbCBkZWZpbml0aW9ucy4KAE1hbGZvcm1lZCBIVCBjb2RlYmxvY2suIEludmFsaWQgY29kZWJsb2NrIGxlbmd0aCB2YWx1ZXMuCgBXZSBkbyBub3Qgc3VwcG9ydCBtb3JlIHRoYW4gMyBjb2RpbmcgcGFzc2VzIGluIGFuIEhUIGNvZGVibG9jazsgVGhpcyBjb2RlYmxvY2tzIGhhcyAlZCBwYXNzZXMuCgBNYWxmb3JtZWQgSFQgY29kZWJsb2NrLiBEZWNvZGluZyB0aGlzIGNvZGVibG9jayBpcyBzdG9wcGVkLiBUaGVyZSBhcmUgJWQgemVybyBiaXRwbGFuZXMgaW4gJWQgYml0cGxhbmVzLgoAQ2Fubm90IHRha2UgaW4gY2hhcmdlIG11bHRpcGxlIHRyYW5zZm9ybWF0aW9uIHN0YWdlcy4KAFVua25vd24gbWFya2VyIGhhcyBiZWVuIGRldGVjdGVkIGFuZCBnZW5lcmF0ZWQgZXJyb3IuCgBDb2RlYyBwcm92aWRlZCB0byB0aGUgb3BqX3NldHVwX2RlY29kZXIgZnVuY3Rpb24gaXMgbm90IGEgZGVjb21wcmVzc29yIGhhbmRsZXIuCgBDb2RlYyBwcm92aWRlZCB0byB0aGUgb3BqX3JlYWRfaGVhZGVyIGZ1bmN0aW9uIGlzIG5vdCBhIGRlY29tcHJlc3NvciBoYW5kbGVyLgoARXJyb3IgcmVhZGluZyBUTE0gbWFya2VyLgoAVGlsZXMgZG9uJ3QgYWxsIGhhdmUgdGhlIHNhbWUgZGltZW5zaW9uLiBTa2lwIHRoZSBNQ1Qgc3RlcC4KAE51bWJlciBvZiBjb21wb25lbnRzICglZCkgaXMgaW5jb25zaXN0ZW50IHdpdGggYSBNQ1QuIFNraXAgdGhlIE1DVCBzdGVwLgoASlAyIGJveCB3aGljaCBhcmUgYWZ0ZXIgdGhlIGNvZGVzdHJlYW0gd2lsbCBub3QgYmUgcmVhZCBieSB0aGlzIGZ1bmN0aW9uLgoATWFsZm9ybWVkIEhUIGNvZGVibG9jay4gV2hlbiB0aGUgbnVtYmVyIG9mIHplcm8gcGxhbmVzIGJpdHBsYW5lcyBpcyBlcXVhbCB0byB0aGUgbnVtYmVyIG9mIGJpdHBsYW5lcywgb25seSB0aGUgY2xlYW51cCBwYXNzIG1ha2VzIHNlbnNlLCBidXQgd2UgaGF2ZSAlZCBwYXNzZXMgaW4gdGhpcyBjb2RlYmxvY2suIFRoZXJlZm9yZSwgb25seSB0aGUgY2xlYW51cCBwYXNzIHdpbGwgYmUgZGVjb2RlZC4gVGhpcyBtZXNzYWdlIHdpbGwgbm90IGJlIGRpc3BsYXllZCBhZ2Fpbi4KAEltYWdlIGhhcyBsZXNzIGNvbXBvbmVudHMgdGhhbiBjb2Rlc3RyZWFtLgoATmVlZCB0byBkZWNvZGUgdGhlIG1haW4gaGVhZGVyIGJlZm9yZSBiZWdpbiB0byBkZWNvZGUgdGhlIHJlbWFpbmluZyBjb2Rlc3RyZWFtLgoAUHNvdCB2YWx1ZSBvZiB0aGUgY3VycmVudCB0aWxlLXBhcnQgaXMgZXF1YWwgdG8gemVybywgd2UgYXNzdW1pbmcgaXQgaXMgdGhlIGxhc3QgdGlsZS1wYXJ0IG9mIHRoZSBjb2Rlc3RyZWFtLgoAQSBtYWxmb3JtZWQgY29kZWJsb2NrIHRoYXQgaGFzIG1vcmUgdGhhbiBvbmUgY29kaW5nIHBhc3MsIGJ1dCB6ZXJvIGxlbmd0aCBmb3IgMm5kIGFuZCBwb3RlbnRpYWxseSB0aGUgM3JkIHBhc3MgaW4gYW4gSFQgY29kZWJsb2NrLgoACQkJIHRpbGUtcGFydFslZF06IHN0YXJfcG9zPSVsbGksIGVuZF9oZWFkZXI9JWxsaSwgZW5kX3Bvcz0lbGxpLgoAVGlsZSAldSBoYXMgVFBzb3QgPT0gMCBhbmQgVE5zb3QgPT0gMCwgYnV0IG5vIG90aGVyIHRpbGUtcGFydHMgd2VyZSBmb3VuZC4gRU9DIGlzIGFsc28gbWlzc2luZy4KAENvbXBvbmVudCAlZCBkb2Vzbid0IGhhdmUgYSBtYXBwaW5nLgoAb3BqX2oya19yZWFkX3RsbSgpOiBUTE0gbWFya2VyIG5vdCBvZiBleHBlY3RlZCBzaXplLgoAQSBjb25mb3JtaW5nIEpQMiByZWFkZXIgc2hhbGwgaWdub3JlIGFsbCBDb2xvdXIgU3BlY2lmaWNhdGlvbiBib3hlcyBhZnRlciB0aGUgZmlyc3QsIHNvIHdlIGlnbm9yZSB0aGlzIG9uZS4KAFRoZSBzaWduYXR1cmUgYm94IG11c3QgYmUgdGhlIGZpcnN0IGJveCBpbiB0aGUgZmlsZS4KAFRoZSAgYm94IG11c3QgYmUgdGhlIGZpcnN0IGJveCBpbiB0aGUgZmlsZS4KAFRoZSBmdHlwIGJveCBtdXN0IGJlIHRoZSBzZWNvbmQgYm94IGluIHRoZSBmaWxlLgoARmFpbGVkIHRvIGRlY29kZS4KAE1hbGZvcm1lZCBIVCBjb2RlYmxvY2suIEluY29ycmVjdCBNRUwgc2VnbWVudCBzZXF1ZW5jZS4KAENvbXBvbmVudCAlZCBpcyBtYXBwZWQgdHdpY2UuCgBvcGpfajJrX3JlYWRfdGxtKCk6IFNUID0gMyBpcyBpbnZhbGlkLgoAT25seSBvbmUgQ01BUCBib3ggaXMgYWxsb3dlZC4KAFdlIG5lZWQgYW4gaW1hZ2UgcHJldmlvdXNseSBjcmVhdGVkLgoASUhEUiBib3hfbWlzc2luZy4gUmVxdWlyZWQuCgBKUDJIIGJveCBtaXNzaW5nLiBSZXF1aXJlZC4KAE5vdCBzdXJlIGhvdyB0aGF0IGhhcHBlbmVkLgoATWFpbiBoZWFkZXIgaGFzIGJlZW4gY29ycmVjdGx5IGRlY29kZWQuCgBUaWxlICVkLyVkIGhhcyBiZWVuIGRlY29kZWQuCgBIZWFkZXIgb2YgdGlsZSAlZCAvICVkIGhhcyBiZWVuIHJlYWQuCgBFbXB0eSBTT1QgbWFya2VyIGRldGVjdGVkOiBQc290PSVkLgoARGlyZWN0IHVzZSBhdCAjJWQgaG93ZXZlciBwY29sPSVkLgoASW1wbGVtZW50YXRpb24gbGltaXRhdGlvbjogZm9yIHBhbGV0dGUgbWFwcGluZywgcGNvbFslZF0gc2hvdWxkIGJlIGVxdWFsIHRvICVkLCBidXQgaXMgZXF1YWwgdG8gJWQuCgBJbnZhbGlkIGNvbXBvbmVudC9wYWxldHRlIGluZGV4IGZvciBkaXJlY3QgbWFwcGluZyAlZC4KAEludmFsaWQgdmFsdWUgZm9yIGNtYXBbJWRdLm10eXAgPSAlZC4KAFBzb3QgdmFsdWUgaXMgbm90IGNvcnJlY3QgcmVnYXJkcyB0byB0aGUgSlBFRzIwMDAgbm9ybTogJWQuCgBNYWxmb3JtZWQgSFQgY29kZWJsb2NrLiBWTEMgY29kZSBwcm9kdWNlcyBzaWduaWZpY2FudCBzYW1wbGVzIG91dHNpZGUgdGhlIGNvZGVibG9jayBhcmVhLgoAVW5leHBlY3RlZCBPT00uCgAzMiBiaXRzIGFyZSBub3QgZW5vdWdoIHRvIGRlY29kZSB0aGlzIGNvZGVibG9jaywgc2luY2UgdGhlIG51bWJlciBvZiBiaXRwbGFuZSwgJWQsIGlzIGxhcmdlciB0aGFuIDMwLgoAQm90dG9tIHBvc2l0aW9uIG9mIHRoZSBkZWNvZGVkIGFyZWEgKHJlZ2lvbl95MT0lZCkgc2hvdWxkIGJlID4gMC4KAFJpZ2h0IHBvc2l0aW9uIG9mIHRoZSBkZWNvZGVkIGFyZWEgKHJlZ2lvbl94MT0lZCkgc2hvdWxkIGJlID4gMC4KAFVwIHBvc2l0aW9uIG9mIHRoZSBkZWNvZGVkIGFyZWEgKHJlZ2lvbl95MD0lZCkgc2hvdWxkIGJlID49IDAuCgBMZWZ0IHBvc2l0aW9uIG9mIHRoZSBkZWNvZGVkIGFyZWEgKHJlZ2lvbl94MD0lZCkgc2hvdWxkIGJlID49IDAuCgBFcnJvciByZWFkaW5nIFBQVCBtYXJrZXI6IHBhY2tldCBoZWFkZXIgaGF2ZSBiZWVuIHByZXZpb3VzbHkgZm91bmQgaW4gdGhlIG1haW4gaGVhZGVyIChQUE0gbWFya2VyKS4KAFN0YXJ0IHRvIHJlYWQgajJrIG1haW4gaGVhZGVyICglbGxkKS4KAEJvdHRvbSBwb3NpdGlvbiBvZiB0aGUgZGVjb2RlZCBhcmVhIChyZWdpb25feTE9JWQpIGlzIG91dHNpZGUgdGhlIGltYWdlIGFyZWEgKFlzaXo9JWQpLgoAVXAgcG9zaXRpb24gb2YgdGhlIGRlY29kZWQgYXJlYSAocmVnaW9uX3kwPSVkKSBpcyBvdXRzaWRlIHRoZSBpbWFnZSBhcmVhIChZc2l6PSVkKS4KAFJpZ2h0IHBvc2l0aW9uIG9mIHRoZSBkZWNvZGVkIGFyZWEgKHJlZ2lvbl94MT0lZCkgaXMgb3V0c2lkZSB0aGUgaW1hZ2UgYXJlYSAoWHNpej0lZCkuCgBMZWZ0IHBvc2l0aW9uIG9mIHRoZSBkZWNvZGVkIGFyZWEgKHJlZ2lvbl94MD0lZCkgaXMgb3V0c2lkZSB0aGUgaW1hZ2UgYXJlYSAoWHNpej0lZCkuCgBCb3R0b20gcG9zaXRpb24gb2YgdGhlIGRlY29kZWQgYXJlYSAocmVnaW9uX3kxPSVkKSBpcyBvdXRzaWRlIHRoZSBpbWFnZSBhcmVhIChZT3Npej0lZCkuCgBVcCBwb3NpdGlvbiBvZiB0aGUgZGVjb2RlZCBhcmVhIChyZWdpb25feTA9JWQpIGlzIG91dHNpZGUgdGhlIGltYWdlIGFyZWEgKFlPc2l6PSVkKS4KAFJpZ2h0IHBvc2l0aW9uIG9mIHRoZSBkZWNvZGVkIGFyZWEgKHJlZ2lvbl94MT0lZCkgaXMgb3V0c2lkZSB0aGUgaW1hZ2UgYXJlYSAoWE9zaXo9JWQpLgoATGVmdCBwb3NpdGlvbiBvZiB0aGUgZGVjb2RlZCBhcmVhIChyZWdpb25feDA9JWQpIGlzIG91dHNpZGUgdGhlIGltYWdlIGFyZWEgKFhPc2l6PSVkKS4KAFNpemUgeCBvZiB0aGUgZGVjb2RlZCBjb21wb25lbnQgaW1hZ2UgaXMgaW5jb3JyZWN0IChjb21wWyVkXS53PSVkKS4KAFNpemUgeSBvZiB0aGUgZGVjb2RlZCBjb21wb25lbnQgaW1hZ2UgaXMgaW5jb3JyZWN0IChjb21wWyVkXS5oPSVkKS4KAFRpbGUgcmVhZCwgZGVjb2RlZCBhbmQgdXBkYXRlZCBpcyBub3QgdGhlIGRlc2lyZWQgb25lICglZCB2cyAlZCkuCgBJbnZhbGlkIGNvbXBvbmVudCBpbmRleCAlZCAoPj0gJWQpLgoAb3BqX3JlYWRfaGVhZGVyKCkgc2hvdWxkIGJlIGNhbGxlZCBiZWZvcmUgb3BqX3NldF9kZWNvZGVkX2NvbXBvbmVudHMoKS4KAE1lbW9yeSBhbGxvY2F0aW9uIGZhaWx1cmUgaW4gb3BqX2pwMl9hcHBseV9wY2xyKCkuCgBpbWFnZS0+Y29tcHNbJWRdLmRhdGEgPT0gTlVMTCBpbiBvcGpfanAyX2FwcGx5X3BjbHIoKS4KAGludmFsaWQgYm94IHNpemUgJWQgKCV4KQoARmFpbCB0byByZWFkIHRoZSBjdXJyZW50IG1hcmtlciBzZWdtZW50ICglI3gpCgBFcnJvciB3aXRoIFNJWiBtYXJrZXI6IElIRFIgdygldSkgaCgldSkgdnMuIFNJWiB3KCV1KSBoKCV1KQoARXJyb3IgcmVhZGluZyBDT0MgbWFya2VyIChiYWQgbnVtYmVyIG9mIGNvbXBvbmVudHMpCgBJbnZhbGlkIG51bWJlciBvZiB0aWxlcyA6ICV1IHggJXUgKG1heGltdW0gZml4ZWQgYnkganBlZzIwMDAgbm9ybSBpcyA2NTUzNSB0aWxlcykKAEludmFsaWQgbnVtYmVyIG9mIGNvbXBvbmVudHMgKGloZHIpCgBOb3QgZW5vdWdoIG1lbW9yeSB0byBoYW5kbGUgaW1hZ2UgaGVhZGVyIChpaGRyKQoAV3JvbmcgdmFsdWVzIGZvcjogdyglZCkgaCglZCkgbnVtY29tcHMoJWQpIChpaGRyKQoASW52YWxpZCB2YWx1ZXMgZm9yIGNvbXAgPSAlZCA6IGR4PSV1IGR5PSV1IChzaG91bGQgYmUgYmV0d2VlbiAxIGFuZCAyNTUgYWNjb3JkaW5nIHRvIHRoZSBKUEVHMjAwMCBub3JtKQoAQmFkIGltYWdlIGhlYWRlciBib3ggKGJhZCBzaXplKQoAQmFkIENPTFIgaGVhZGVyIGJveCAoYmFkIHNpemUpCgBCYWQgQlBDQyBoZWFkZXIgYm94IChiYWQgc2l6ZSkKAEVycm9yIHdpdGggU0laIG1hcmtlcjogbmVnYXRpdmUgb3IgemVybyBpbWFnZSBzaXplICglbGxkIHggJWxsZCkKAHNraXA6IHNlZ21lbnQgdG9vIGxvbmcgKCVkKSB3aXRoIG1heCAoJWQpIGZvciBjb2RlYmxvY2sgJWQgKHA9JWQsIGI9JWQsIHI9JWQsIGM9JWQpCgByZWFkOiBzZWdtZW50IHRvbyBsb25nICglZCkgd2l0aCBtYXggKCVkKSBmb3IgY29kZWJsb2NrICVkIChwPSVkLCBiPSVkLCByPSVkLCBjPSVkKQoARGVzcGl0ZSBKUDIgQlBDIT0yNTUsIHByZWNpc2lvbiBhbmQvb3Igc2duZCB2YWx1ZXMgZm9yIGNvbXBbJWRdIGlzIGRpZmZlcmVudCB0aGFuIGNvbXBbMF06CiAgICAgICAgWzBdIHByZWMoJWQpIHNnbmQoJWQpIFslZF0gcHJlYyglZCkgc2duZCglZCkKAGJhZCBjb21wb25lbnQgbnVtYmVyIGluIFJHTiAoJWQgd2hlbiB0aGVyZSBhcmUgb25seSAlZCkKAEVycm9yIHdpdGggU0laIG1hcmtlcjogbnVtYmVyIG9mIGNvbXBvbmVudCBpcyBub3QgY29tcGF0aWJsZSB3aXRoIHRoZSByZW1haW5pbmcgbnVtYmVyIG9mIHBhcmFtZXRlcnMgKCAlZCB2cyAlZCkKAEVycm9yIHdpdGggU0laIG1hcmtlcjogaW52YWxpZCB0aWxlIHNpemUgKHRkeDogJWQsIHRkeTogJWQpCgBCYWQgQ09MUiBoZWFkZXIgYm94IChiYWQgc2l6ZTogJWQpCgBCYWQgQ09MUiBoZWFkZXIgYm94IChDSUVMYWIsIGJhZCBzaXplOiAlZCkKAFBURVJNIGNoZWNrIGZhaWx1cmU6ICVkIHJlbWFpbmluZyBieXRlcyBpbiBjb2RlIGJsb2NrICglZCB1c2VkIC8gJWQpCgBNYWxmb3JtZWQgSFQgY29kZWJsb2NrLiBPbmUgb2YgdGhlIGZvbGxvd2luZyBjb25kaXRpb24gaXMgbm90IG1ldDogMiA8PSBTY3VwIDw9IG1pbihMY3VwLCA0MDc5KQoASW52YWxpZCB2YWx1ZXMgZm9yIGNvbXAgPSAlZCA6IHByZWM9JXUgKHNob3VsZCBiZSBiZXR3ZWVuIDEgYW5kIDM4IGFjY29yZGluZyB0byB0aGUgSlBFRzIwMDAgbm9ybS4gT3BlbkpwZWcgb25seSBzdXBwb3J0cyB1cCB0byAzMSkKAEludmFsaWQgYml0IG51bWJlciAlZCBpbiBvcGpfdDJfcmVhZF9wYWNrZXRfaGVhZGVyKCkKAFN0cmVhbSBlcnJvciEKAEVycm9yIG9uIHdyaXRpbmcgc3RyZWFtIQoAU3RyZWFtIHJlYWNoZWQgaXRzIGVuZCAhCgBFeHBlY3RlZCBhIFNPQyBtYXJrZXIgCgBJbnZhbGlkIGJveCBzaXplICVkIGZvciBib3ggJyVjJWMlYyVjJy4gTmVlZCAlZCBieXRlcywgJWQgYnl0ZXMgcmVtYWluaW5nIAoATWFsZm9ybWVkIEhUIGNvZGVibG9jay4gRGVjb2RpbmcgdGhpcyBjb2RlYmxvY2sgaXMgc3RvcHBlZC4gVV9xIGlzIGxhcmdlciB0aGFuIHplcm8gYml0cGxhbmVzICsgMSAKAE1hbGZvcm1lZCBIVCBjb2RlYmxvY2suIERlY29kaW5nIHRoaXMgY29kZWJsb2NrIGlzIHN0b3BwZWQuIFVfcSBpc2xhcmdlciB0aGFuIGJpdHBsYW5lcyArIDEgCgBDT0xSIEJPWCBtZXRoIHZhbHVlIGlzIG5vdCBhIHJlZ3VsYXIgdmFsdWUgKCVkKSwgc28gd2Ugd2lsbCBpZ25vcmUgdGhlIGVudGlyZSBDb2xvdXIgU3BlY2lmaWNhdGlvbiBib3guIAoAV2hpbGUgcmVhZGluZyBDQ1BfUU5UU1RZIGVsZW1lbnQgaW5zaWRlIFFDRCBvciBRQ0MgbWFy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AAAAAFQAAKBTAAABJAAAAAAAABBUAACwUwAAASQAAAEAAAAgVAAAwFMAAAEiAAAAAAAAMFQAANBTAAABIgAAAQAAAEBUAADgUwAAARwAAAAAAABQVAAA8FMAAAEcAAABAAAAYFQAAABUAAABGAAAAAAAAHBUAAAQVAAAARgAAAEAAACAVAAAIFQAAAEWAAAAAAAAkFQAADBUAAABFgAAAQAAAKBUAABAVAAAARQAAAAAAACwVAAAUFQAAAEUAAABAAAAwFQAAGBUAAABEgAAAAAAANBUAABwVAAAARIAAAEAAADgVAAAgFQAAAERAAAAAAAA8FQAAJBUAAABEQAAAQAAAABVAACgVAAAwQoAAAAAAAAQVQAAsFQAAMEKAAABAAAAIFUAAMBUAADBCQAAAAAAADBVAADQVAAAwQkAAAEAAABAVQAA4FQAAKEIAAAAAAAAUFUAAPBUAAChCAAAAQAAAGBVAAAAVQAAIQUAAAAAAABwVQAAEFUAACEFAAABAAAAgFUAACBVAABBBAAAAAAAAJBVAAAwVQAAQQQAAAEAAACgVQAAQFUAAKECAAAAAAAAsFUAAFBVAAChAgAAAQAAAMBVAABgVQAAIQIAAAAAAADQVQAAcFUAACECAAABAAAA4FUAAIBVAABBAQAAAAAAAPBVAACQVQAAQQEAAAEAAAAAVgAAoFUAABEBAAAAAAAAEFYAALBVAAARAQAAAQAAACBWAADAVQAAhQAAAAAAAAAwVgAA0FUAAIUAAAABAAAAQFYAAOBVAABJAAAAAAAAAFBWAADwVQAASQAAAAEAAABgVgAAAFYAACUAAAAAAAAAcFYAABBWAAAlAAAAAQAAAIBWAAAgVgAAFQAAAAAAAACQVgAAMFYAABUAAAABAAAAoFYAAEBWAAAJAAAAAAAAALBWAABQVgAACQAAAAEAAADAVgAAYFYAAAUAAAAAAAAA0FYAAHBWAAAFAAAAAQAAAOBWAACAVgAAAQAAAAAAAADQVgAAkFYAAAEAAAABAAAA4FYAAKBWAAABVgAAAAAAAPBWAADwVgAAAVYAAAEAAAAAVwAAAFcAAAABAwMBAgMDBQYHBwYGBwcAAQMDAQIDAwUGBwcGBgcHBQYHBwYGBwcICAgICAgICAUGBwcGBgcHCAgICAgICAgBAgMDAgIDAwYGBwcGBgcHAQIDAwICAwMGBgcHBgYHBwYGBwcGBgcHCAgICAgICAgGBgcHBgYHBwgICAgICAgIAwMEBAMDBAQHBwcHBwcHBwMDBAQDAwQEBwcHBwcHBwcHBwcHBwcHBwgICAgICAgIBwcHBwcHBwcICAgICAgICAMDBAQDAwQEBwcHBwcHBwcDAwQEAwMEBAcHBwcHBwcHBwcHBwcHBwcICAgICAgICAcHBwcHBwcHCAgICAgICAgBAgMDAgIDAwYGBwcGBgcHAQIDAwICAwMGBgcHBgYHBwYGBwcGBgcHCAgICAgICAgGBgcHBgYHBwgICAgICAgIAgIDAwICAwMGBgcHBgYHBwICAwMCAgMDBgYHBwYGBwcGBgcHBgYHBwgICAgICAgIBgYHBwYGBwcICAgICAgICAMDBAQDAwQEBwcHBwcHBwcDAwQEAwMEBAcHBwcHBwcHBwcHBwcHBwcICAgICAgICAcHBwcHBwcHCAgICAgICAgDAwQEAwMEBAcHBwcHBwcHAwMEBAMDBAQHBwcHBwcHBwcHBwcHBwcHCAgICAgICAgHBwcHBwcHBwgICAgICAgIAAEFBgECBgYDAwcHAwMHBwABBQYBAgYGAwMHBwMDBwcDAwcHAwMHBwQEBwcEBAcHAwMHBwMDBwcEBAcHBAQHBwECBgYCAgYGAwMHBwMDBwcBAgYGAgIGBgMDBwcDAwcHAwMHBwMDBwcEBAcHBAQHBwMDBwcDAwcHBAQHBwQEBwcFBggIBgYICAcHCAgHBwgIBQYICAYGCAgHBwgIBwcICAcHCAgHBwgIBwcICAcHCAgHBwgIBwcICAcHCAgHBwgIBgYICAYGCAgHBwgIBwcICAYGCAgGBggIBwcICAcHCAgHBwgIBwcICAcHCAgHBwgIBwcICAcHCAgHBwgIBwcICAECBgYCAgYGAwMHBwMDBwcBAgYGAgIGBgMDBwcDAwcHAwMHBwMDBwcEBAcHBAQHBwMDBwcDAwcHBAQHBwQEBwcCAgYGAgIGBgMDBwcDAwcHAgIGBgICBgYDAwcHAwMHBwMDBwcDAwcHBAQHBwQEBwcDAwcHAwMHBwQEBwcEBAcHBgYICAYGCAgHBwgIBwcICAYGCAgGBggIBwcICAcHCAgHBwgIBwcICAcHCAgHBwgIBwcICAcHCAgHBwgIBwcICAYGCAgGBggIBwcICAcHCAgGBggIBgYICAcHCAgHBwgIBwcICAcHCAgHBwgIBwcICAcHCAgHBwgIBwcICAcHCAgAAQMDAQIDAwUGBwcGBgcHAAEDAwECAwMFBgcHBgYHBwUGBwcGBgcHCAgICAgICAgFBgcHBgYHBwgICAgICAgIAQIDAwICAwMGBgcHBgYHBwECAwMCAgMDBgYHBwYGBwcGBgcHBgYHBwgICAgICAgIBgYHBwYGBwcICAgICAgICAMDBAQDAwQEBwcHBwcHBwcDAwQEAwMEBAcHBwcHBwcHBwcHBwcHBwcICAgICAgICAcHBwcHBwcHCAgICAgICAgDAwQEAwMEBAcHBwcHBwcHAwMEBAMDBAQHBwcHBwcHBwcHBwcHBwcHCAgICAgICAgHBwcHBwcHBwgICAgICAgIAQIDAwICAwMGBgcHBgYHBwECAwMCAgMDBgYHBwYGBwcGBgcHBgYHBwgICAgICAgIBgYHBwYGBwcICAgICAgICAICAwMCAgMDBgYHBwYGBwcCAgMDAgIDAwYGBwcGBgcHBgYHBwYGBwcICAgICAgICAYGBwcGBgcHCAgICAgICAgDAwQEAwMEBAcHBwcHBwcHAwMEBAMDBAQHBwcHBwcHBwcHBwcHBwcHCAgICAgICAgHBwcHBwcHBwgICAgICAgIAwMEBAMDBAQHBwcHBwcHBwMDBAQDAwQEBwcHBwcHBwcHBwcHBwcHBwgICAgICAgIBwcHBwcHBwcICAgICAgICAADAQQDBgQHAQQCBQQHBQcAAwEEAwYEBwEEAgUEBwUHAQQCBQQHBQcCBQIFBQcFBwEEAgUEBwUHAgUCBQUHBQcDBgQHBggHCAQHBQcHCAcIAwYEBwYIBwgEBwUHBwgHCAQHBQcHCAcIBQcFBwcIBwgEBwUHBwgHCAUHBQcHCAcIAQQCBQQHBQcCBQIFBQcFBwEEAgUEBwUHAgUCBQUHBQcCBQIFBQcFBwIFAgUFBwUHAgUCBQUHBQcCBQIFBQcFBwQHBQcHCAcIBQcFBwcIBwgEBwUHBwgHCAUHBQcHCAcIBQcFBwcIBwgFBwUHBwgHCAUHBQcHCAcIBQcFBwcIBwgDBgQHBggHCAQHBQcHCAcIAwYEBwYIBwgEBwUHBwgHCAQHBQcHCAcIBQcFBwcIBwgEBwUHBwgHCAUHBQcHCAcIBggHCAgICAgHCAcICAgICAYIBwgICAgIBwgHCAgICAgH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";
      return f;
    }
    var wasmBinaryFile;
    function getBinarySync(file) {
      if (file == wasmBinaryFile && wasmBinary) {
        return new Uint8Array(wasmBinary);
      }
      var binary = tryParseAsDataURI(file);
      if (binary) {
        return binary;
      }
      if (readBinary) {
        return readBinary(file);
      }
      throw 'sync fetching of the wasm failed: you can preload it to Module["wasmBinary"] manually, or emcc.py will do that for you when generating HTML (but not JS)';
    }
    function instantiateSync(file, info) {
      var module;
      var binary = getBinarySync(file);
      module = new WebAssembly.Module(binary);
      var instance = new WebAssembly.Instance(module, info);
      return [instance, module];
    }
    function getWasmImports() {
      return {
        a: wasmImports
      };
    }
    function createWasm() {
      function receiveInstance(instance, module) {
        wasmExports = instance.exports;
        wasmMemory = wasmExports["t"];
        updateMemoryViews();
        addOnInit(wasmExports["u"]);
        removeRunDependency("wasm-instantiate");
        return wasmExports;
      }
      addRunDependency("wasm-instantiate");
      var info = getWasmImports();
      if (Module["instantiateWasm"]) {
        try {
          return Module["instantiateWasm"](info, receiveInstance);
        } catch (e) {
          err(`Module.instantiateWasm callback failed with error: ${e}`);
          readyPromiseReject(e);
        }
      }
      wasmBinaryFile ??= findWasmBinary();
      var result = instantiateSync(wasmBinaryFile, info);
      return receiveInstance(result[0]);
    }
    class ExitStatus {
      name = "ExitStatus";
      constructor(status) {
        this.message = `Program terminated with exit(${status})`;
        this.status = status;
      }
    }
    var callRuntimeCallbacks = callbacks => {
      while (callbacks.length > 0) {
        callbacks.shift()(Module);
      }
    };
    var noExitRuntime = Module["noExitRuntime"] || true;
    var __abort_js = () => abort("");
    var __emscripten_memcpy_js = (dest, src, num) => HEAPU8.copyWithin(dest, src, src + num);
    var runtimeKeepaliveCounter = 0;
    var __emscripten_runtime_keepalive_clear = () => {
      noExitRuntime = false;
      runtimeKeepaliveCounter = 0;
    };
    var timers = {};
    var handleException = e => {
      if (e instanceof ExitStatus || e == "unwind") {
        return EXITSTATUS;
      }
      quit_(1, e);
    };
    var keepRuntimeAlive = () => noExitRuntime || runtimeKeepaliveCounter > 0;
    var _proc_exit = code => {
      EXITSTATUS = code;
      if (!keepRuntimeAlive()) {
        Module["onExit"]?.(code);
        ABORT = true;
      }
      quit_(code, new ExitStatus(code));
    };
    var exitJS = (status, implicit) => {
      EXITSTATUS = status;
      _proc_exit(status);
    };
    var _exit = exitJS;
    var maybeExit = () => {
      if (!keepRuntimeAlive()) {
        try {
          _exit(EXITSTATUS);
        } catch (e) {
          handleException(e);
        }
      }
    };
    var callUserCallback = func => {
      if (ABORT) {
        return;
      }
      try {
        func();
        maybeExit();
      } catch (e) {
        handleException(e);
      }
    };
    var _emscripten_get_now = () => performance.now();
    var __setitimer_js = (which, timeout_ms) => {
      if (timers[which]) {
        clearTimeout(timers[which].id);
        delete timers[which];
      }
      if (!timeout_ms) return 0;
      var id = setTimeout(() => {
        delete timers[which];
        callUserCallback(() => __emscripten_timeout(which, _emscripten_get_now()));
      }, timeout_ms);
      timers[which] = {
        id,
        timeout_ms
      };
      return 0;
    };
    function _copy_pixels_1(compG_ptr, nb_pixels) {
      compG_ptr >>= 2;
      const imageData = Module.imageData = new Uint8ClampedArray(nb_pixels);
      const compG = Module.HEAP32.subarray(compG_ptr, compG_ptr + nb_pixels);
      imageData.set(compG);
    }
    function _copy_pixels_3(compR_ptr, compG_ptr, compB_ptr, nb_pixels) {
      compR_ptr >>= 2;
      compG_ptr >>= 2;
      compB_ptr >>= 2;
      const imageData = Module.imageData = new Uint8ClampedArray(nb_pixels * 3);
      const compR = Module.HEAP32.subarray(compR_ptr, compR_ptr + nb_pixels);
      const compG = Module.HEAP32.subarray(compG_ptr, compG_ptr + nb_pixels);
      const compB = Module.HEAP32.subarray(compB_ptr, compB_ptr + nb_pixels);
      for (let i = 0; i < nb_pixels; i++) {
        imageData[3 * i] = compR[i];
        imageData[3 * i + 1] = compG[i];
        imageData[3 * i + 2] = compB[i];
      }
    }
    function _copy_pixels_4(compR_ptr, compG_ptr, compB_ptr, compA_ptr, nb_pixels) {
      compR_ptr >>= 2;
      compG_ptr >>= 2;
      compB_ptr >>= 2;
      compA_ptr >>= 2;
      const imageData = Module.imageData = new Uint8ClampedArray(nb_pixels * 4);
      const compR = Module.HEAP32.subarray(compR_ptr, compR_ptr + nb_pixels);
      const compG = Module.HEAP32.subarray(compG_ptr, compG_ptr + nb_pixels);
      const compB = Module.HEAP32.subarray(compB_ptr, compB_ptr + nb_pixels);
      const compA = Module.HEAP32.subarray(compA_ptr, compA_ptr + nb_pixels);
      for (let i = 0; i < nb_pixels; i++) {
        imageData[4 * i] = compR[i];
        imageData[4 * i + 1] = compG[i];
        imageData[4 * i + 2] = compB[i];
        imageData[4 * i + 3] = compA[i];
      }
    }
    var getHeapMax = () => 2147483648;
    var alignMemory = (size, alignment) => Math.ceil(size / alignment) * alignment;
    var growMemory = size => {
      var b = wasmMemory.buffer;
      var pages = (size - b.byteLength + 65535) / 65536 | 0;
      try {
        wasmMemory.grow(pages);
        updateMemoryViews();
        return 1;
      } catch (e) {}
    };
    var _emscripten_resize_heap = requestedSize => {
      var oldSize = HEAPU8.length;
      requestedSize >>>= 0;
      var maxHeapSize = getHeapMax();
      if (requestedSize > maxHeapSize) {
        return false;
      }
      for (var cutDown = 1; cutDown <= 4; cutDown *= 2) {
        var overGrownHeapSize = oldSize * (1 + .2 / cutDown);
        overGrownHeapSize = Math.min(overGrownHeapSize, requestedSize + 100663296);
        var newSize = Math.min(maxHeapSize, alignMemory(Math.max(requestedSize, overGrownHeapSize), 65536));
        var replacement = growMemory(newSize);
        if (replacement) {
          return true;
        }
      }
      return false;
    };
    var ENV = {};
    var getExecutableName = () => thisProgram || "./this.program";
    var getEnvStrings = () => {
      if (!getEnvStrings.strings) {
        var lang = (typeof navigator == "object" && navigator.languages && navigator.languages[0] || "C").replace("-", "_") + ".UTF-8";
        var env = {
          USER: "web_user",
          LOGNAME: "web_user",
          PATH: "/",
          PWD: "/",
          HOME: "/home/web_user",
          LANG: lang,
          _: getExecutableName()
        };
        for (var x in ENV) {
          if (ENV[x] === undefined) delete env[x];else env[x] = ENV[x];
        }
        var strings = [];
        for (var x in env) {
          strings.push(`${x}=${env[x]}`);
        }
        getEnvStrings.strings = strings;
      }
      return getEnvStrings.strings;
    };
    var stringToAscii = (str, buffer) => {
      for (var i = 0; i < str.length; ++i) {
        HEAP8[buffer++] = str.charCodeAt(i);
      }
      HEAP8[buffer] = 0;
    };
    var _environ_get = (__environ, environ_buf) => {
      var bufSize = 0;
      getEnvStrings().forEach((string, i) => {
        var ptr = environ_buf + bufSize;
        HEAPU32[__environ + i * 4 >> 2] = ptr;
        stringToAscii(string, ptr);
        bufSize += string.length + 1;
      });
      return 0;
    };
    var _environ_sizes_get = (penviron_count, penviron_buf_size) => {
      var strings = getEnvStrings();
      HEAPU32[penviron_count >> 2] = strings.length;
      var bufSize = 0;
      strings.forEach(string => bufSize += string.length + 1);
      HEAPU32[penviron_buf_size >> 2] = bufSize;
      return 0;
    };
    var _fd_close = fd => 52;
    var convertI32PairToI53Checked = (lo, hi) => hi + 2097152 >>> 0 < 4194305 - !!lo ? (lo >>> 0) + hi * 4294967296 : NaN;
    function _fd_seek(fd, offset_low, offset_high, whence, newOffset) {
      var offset = convertI32PairToI53Checked(offset_low, offset_high);
      return 70;
    }
    var printCharBuffers = [null, [], []];
    var UTF8Decoder = typeof TextDecoder != "undefined" ? new TextDecoder() : undefined;
    var UTF8ArrayToString = (heapOrArray, idx = 0, maxBytesToRead = NaN) => {
      var endIdx = idx + maxBytesToRead;
      var endPtr = idx;
      while (heapOrArray[endPtr] && !(endPtr >= endIdx)) ++endPtr;
      if (endPtr - idx > 16 && heapOrArray.buffer && UTF8Decoder) {
        return UTF8Decoder.decode(heapOrArray.subarray(idx, endPtr));
      }
      var str = "";
      while (idx < endPtr) {
        var u0 = heapOrArray[idx++];
        if (!(u0 & 128)) {
          str += String.fromCharCode(u0);
          continue;
        }
        var u1 = heapOrArray[idx++] & 63;
        if ((u0 & 224) == 192) {
          str += String.fromCharCode((u0 & 31) << 6 | u1);
          continue;
        }
        var u2 = heapOrArray[idx++] & 63;
        if ((u0 & 240) == 224) {
          u0 = (u0 & 15) << 12 | u1 << 6 | u2;
        } else {
          u0 = (u0 & 7) << 18 | u1 << 12 | u2 << 6 | heapOrArray[idx++] & 63;
        }
        if (u0 < 65536) {
          str += String.fromCharCode(u0);
        } else {
          var ch = u0 - 65536;
          str += String.fromCharCode(55296 | ch >> 10, 56320 | ch & 1023);
        }
      }
      return str;
    };
    var printChar = (stream, curr) => {
      var buffer = printCharBuffers[stream];
      if (curr === 0 || curr === 10) {
        (stream === 1 ? out : err)(UTF8ArrayToString(buffer));
        buffer.length = 0;
      } else {
        buffer.push(curr);
      }
    };
    var UTF8ToString = (ptr, maxBytesToRead) => ptr ? UTF8ArrayToString(HEAPU8, ptr, maxBytesToRead) : "";
    var _fd_write = (fd, iov, iovcnt, pnum) => {
      var num = 0;
      for (var i = 0; i < iovcnt; i++) {
        var ptr = HEAPU32[iov >> 2];
        var len = HEAPU32[iov + 4 >> 2];
        iov += 8;
        for (var j = 0; j < len; j++) {
          printChar(fd, HEAPU8[ptr + j]);
        }
        num += len;
      }
      HEAPU32[pnum >> 2] = num;
      return 0;
    };
    function _gray_to_rgba(compG_ptr, nb_pixels) {
      compG_ptr >>= 2;
      const imageData = Module.imageData = new Uint8ClampedArray(nb_pixels * 4);
      const compG = Module.HEAP32.subarray(compG_ptr, compG_ptr + nb_pixels);
      for (let i = 0; i < nb_pixels; i++) {
        imageData[4 * i] = imageData[4 * i + 1] = imageData[4 * i + 2] = compG[i];
        imageData[4 * i + 3] = 255;
      }
    }
    function _graya_to_rgba(compG_ptr, compA_ptr, nb_pixels) {
      compG_ptr >>= 2;
      compA_ptr >>= 2;
      const imageData = Module.imageData = new Uint8ClampedArray(nb_pixels * 4);
      const compG = Module.HEAP32.subarray(compG_ptr, compG_ptr + nb_pixels);
      const compA = Module.HEAP32.subarray(compA_ptr, compA_ptr + nb_pixels);
      for (let i = 0; i < nb_pixels; i++) {
        imageData[4 * i] = imageData[4 * i + 1] = imageData[4 * i + 2] = compG[i];
        imageData[4 * i + 3] = compA[i];
      }
    }
    function _jsPrintWarning(message_ptr) {
      const message = UTF8ToString(message_ptr);
      (Module.warn || console.warn)(`OpenJPEG: ${message}`);
    }
    function _rgb_to_rgba(compR_ptr, compG_ptr, compB_ptr, nb_pixels) {
      compR_ptr >>= 2;
      compG_ptr >>= 2;
      compB_ptr >>= 2;
      const imageData = Module.imageData = new Uint8ClampedArray(nb_pixels * 4);
      const compR = Module.HEAP32.subarray(compR_ptr, compR_ptr + nb_pixels);
      const compG = Module.HEAP32.subarray(compG_ptr, compG_ptr + nb_pixels);
      const compB = Module.HEAP32.subarray(compB_ptr, compB_ptr + nb_pixels);
      for (let i = 0; i < nb_pixels; i++) {
        imageData[4 * i] = compR[i];
        imageData[4 * i + 1] = compG[i];
        imageData[4 * i + 2] = compB[i];
        imageData[4 * i + 3] = 255;
      }
    }
    function _storeErrorMessage(message_ptr) {
      const message = UTF8ToString(message_ptr);
      if (!Module.errorMessages) {
        Module.errorMessages = message;
      } else {
        Module.errorMessages += "\n" + message;
      }
    }
    var wasmImports = {
      m: __abort_js,
      c: __emscripten_memcpy_js,
      l: __emscripten_runtime_keepalive_clear,
      n: __setitimer_js,
      g: _copy_pixels_1,
      f: _copy_pixels_3,
      e: _copy_pixels_4,
      o: _emscripten_resize_heap,
      p: _environ_get,
      q: _environ_sizes_get,
      r: _fd_close,
      j: _fd_seek,
      b: _fd_write,
      s: _gray_to_rgba,
      i: _graya_to_rgba,
      d: _jsPrintWarning,
      k: _proc_exit,
      h: _rgb_to_rgba,
      a: _storeErrorMessage
    };
    var wasmExports = createWasm();
    var ___wasm_call_ctors = wasmExports["u"];
    var _malloc = Module["_malloc"] = wasmExports["v"];
    var _free = Module["_free"] = wasmExports["w"];
    var _jp2_decode = Module["_jp2_decode"] = wasmExports["y"];
    var __emscripten_timeout = wasmExports["z"];
    var calledRun;
    dependenciesFulfilled = function runCaller() {
      if (!calledRun) run();
      if (!calledRun) dependenciesFulfilled = runCaller;
    };
    function run() {
      if (runDependencies > 0) {
        return;
      }
      preRun();
      if (runDependencies > 0) {
        return;
      }
      function doRun() {
        if (calledRun) return;
        calledRun = true;
        Module["calledRun"] = true;
        if (ABORT) return;
        initRuntime();
        readyPromiseResolve(Module);
        Module["onRuntimeInitialized"]?.();
        postRun();
      }
      if (Module["setStatus"]) {
        Module["setStatus"]("Running...");
        setTimeout(() => {
          setTimeout(() => Module["setStatus"](""), 1);
          doRun();
        }, 1);
      } else {
        doRun();
      }
    }
    if (Module["preInit"]) {
      if (typeof Module["preInit"] == "function") Module["preInit"] = [Module["preInit"]];
      while (Module["preInit"].length > 0) {
        Module["preInit"].pop()();
      }
    }
    run();
    moduleRtn = Module;
    return moduleRtn;
  };
})();
/* harmony default export */ const openjpeg = (OpenJPEG);
;// ./src/core/stream.js








class Stream extends base_stream_BaseStream {
  constructor(arrayBuffer, start, length, dict) {
    super();
    this.bytes = arrayBuffer instanceof Uint8Array ? arrayBuffer : new Uint8Array(arrayBuffer);
    this.start = start || 0;
    this.pos = this.start;
    this.end = start + length || this.bytes.length;
    this.dict = dict;
  }
  get length() {
    return this.end - this.start;
  }
  get isEmpty() {
    return this.length === 0;
  }
  getByte() {
    if (this.pos >= this.end) {
      return -1;
    }
    return this.bytes[this.pos++];
  }
  getBytes(length) {
    const bytes = this.bytes;
    const pos = this.pos;
    const strEnd = this.end;
    if (!length) {
      return bytes.subarray(pos, strEnd);
    }
    let end = pos + length;
    if (end > strEnd) {
      end = strEnd;
    }
    this.pos = end;
    return bytes.subarray(pos, end);
  }
  getByteRange(begin, end) {
    if (begin < 0) {
      begin = 0;
    }
    if (end > this.end) {
      end = this.end;
    }
    return this.bytes.subarray(begin, end);
  }
  reset() {
    this.pos = this.start;
  }
  moveStart() {
    this.start = this.pos;
  }
  makeSubStream(start, length, dict = null) {
    return new Stream(this.bytes.buffer, start, length, dict);
  }
}
class StringStream extends Stream {
  constructor(str) {
    super(stringToBytes(str));
  }
}
class NullStream extends Stream {
  constructor() {
    super(new Uint8Array(0));
  }
}

;// ./src/core/jpx.js






class JpxError extends BaseException {
  constructor(msg) {
    super(msg, "JpxError");
  }
}
class JpxImage {
  static #module = null;
  static decode(data, decoderOptions) {
    decoderOptions ||= {};
    this.#module ||= openjpeg({
      warn: util_warn
    });
    const imageData = this.#module.decode(data, decoderOptions);
    if (typeof imageData === "string") {
      throw new JpxError(imageData);
    }
    return imageData;
  }
  static cleanup() {
    this.#module = null;
  }
  static parseImageProperties(stream) {
    if (stream instanceof ArrayBuffer || ArrayBuffer.isView(stream)) {
      stream = new Stream(stream);
    } else {
      throw new JpxError("Invalid data format, must be a TypedArray.");
    }
    let newByte = stream.getByte();
    while (newByte >= 0) {
      const oldByte = newByte;
      newByte = stream.getByte();
      const code = oldByte << 8 | newByte;
      if (code === 0xff51) {
        stream.skip(4);
        const Xsiz = stream.getInt32() >>> 0;
        const Ysiz = stream.getInt32() >>> 0;
        const XOsiz = stream.getInt32() >>> 0;
        const YOsiz = stream.getInt32() >>> 0;
        stream.skip(16);
        const Csiz = stream.getUint16();
        return {
          width: Xsiz - XOsiz,
          height: Ysiz - YOsiz,
          bitsPerComponent: 8,
          componentsCount: Csiz
        };
      }
    }
    throw new JpxError("No size marker found in JPX stream");
  }
}

;// ./src/pdf.image_decoders.js




const pdfjsVersion = "4.10.38";
const pdfjsBuild = "f9bea397f";

var __webpack_exports__Jbig2Error = __webpack_exports__.Jbig2Error;
var __webpack_exports__Jbig2Image = __webpack_exports__.Jbig2Image;
var __webpack_exports__JpegError = __webpack_exports__.JpegError;
var __webpack_exports__JpegImage = __webpack_exports__.JpegImage;
var __webpack_exports__JpxError = __webpack_exports__.JpxError;
var __webpack_exports__JpxImage = __webpack_exports__.JpxImage;
var __webpack_exports__VerbosityLevel = __webpack_exports__.VerbosityLevel;
var __webpack_exports__getVerbosityLevel = __webpack_exports__.getVerbosityLevel;
var __webpack_exports__setVerbosityLevel = __webpack_exports__.setVerbosityLevel;
export { __webpack_exports__Jbig2Error as Jbig2Error, __webpack_exports__Jbig2Image as Jbig2Image, __webpack_exports__JpegError as JpegError, __webpack_exports__JpegImage as JpegImage, __webpack_exports__JpxError as JpxError, __webpack_exports__JpxImage as JpxImage, __webpack_exports__VerbosityLevel as VerbosityLevel, __webpack_exports__getVerbosityLevel as getVerbosityLevel, __webpack_exports__setVerbosityLevel as setVerbosityLevel };

//# sourceMappingURL=pdf.image_decoders.mjs.map