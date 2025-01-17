import { describe, expect, it, vi } from 'vitest';
import { createRef } from 'react';
import { render } from '@testing-library/react';

import mergeRefs from './index.js';

describe('mergeRefs()', () => {
  it('returns falsy result given no arguments', () => {
    const result = mergeRefs();

    expect(result).toBeFalsy();
  });

  it('returns falsy result given falsy arguments', () => {
    const result = mergeRefs(null, null);

    expect(result).toBeFalsy();
  });

  it('returns original ref given only one ref', () => {
    const ref = vi.fn();

    const result = mergeRefs(ref);

    expect(result).toBe(ref);
  });

  it('returns original ref given one ref and one falsy argument', () => {
    const ref = vi.fn();

    const result = mergeRefs(ref, null);

    expect(result).toBe(ref);
  });

  it('returns merged refs properly', () => {
    const ref1 = vi.fn();
    const ref2 = createRef<HTMLDivElement>();

    const result = mergeRefs(ref1, ref2);

    expect(result).not.toBe(ref1);
    expect(result).toEqual(expect.any(Function));
  });

  it('handles merged functional refs properly', () => {
    const ref1 = vi.fn();
    const ref2 = createRef<HTMLDivElement>();

    const mergedRef = mergeRefs(ref1, ref2);

    const { container } = render(<div ref={mergedRef} />);

    expect(ref1).toHaveBeenCalledTimes(1);
    expect(ref1).toHaveBeenCalledWith(container.firstElementChild);
  });

  it('handles merged object refs properly', () => {
    const ref1 = createRef<HTMLDivElement>();
    const ref2 = vi.fn();

    const mergedRef = mergeRefs(ref1, ref2);

    const { container } = render(<div ref={mergedRef} />);

    expect(ref1.current).toBe(container.firstElementChild);
  });
});
