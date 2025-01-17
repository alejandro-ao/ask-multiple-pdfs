[![npm](https://img.shields.io/npm/v/merge-refs.svg)](https://www.npmjs.com/package/merge-refs) ![downloads](https://img.shields.io/npm/dt/merge-refs.svg) [![CI](https://github.com/wojtekmaj/merge-refs/actions/workflows/ci.yml/badge.svg)](https://github.com/wojtekmaj/merge-refs/actions)

# Merge-Refs

A function that merges React refs into one. Filters out invalid (eg. falsy) refs as well and returns original ref if only one valid ref was given.

## tl;dr

- Install by executing `npm install merge-refs` or `yarn add merge-refs`.
- Import by adding `import mergeRefs from 'merge-refs'`.
- Use it in `ref` like so: `<div ref={mergeRefs(ref, someOtherRef)} />`

## Accepted refs

- Refs created using `createRef()`
- Refs created using `useRef()`
- Functional refs

## Example

```tsx
function Hello() {
  const ref1 = useRef<HTMLDivElement>(); // I'm going to be updated!
  const ref2 = (element: HTMLDivElement) => {
    // I'm going to be called!
  };

  return <div ref={mergeRefs(ref1, ref2)} />;
}
```

## License

The MIT License.

## Author

<table>
  <tr>
    <td >
      <img src="https://avatars.githubusercontent.com/u/5426427?v=4&s=128" width="64" height="64" alt="Wojciech Maj">
    </td>
    <td>
      <a href="https://github.com/wojtekmaj">Wojciech Maj</a>
    </td>
  </tr>
</table>
