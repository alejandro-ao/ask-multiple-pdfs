import { describe, expect, it, vi } from 'vitest';

import makeCancellablePromise from './index.js';

vi.useFakeTimers();

describe('makeCancellablePromise()', () => {
  function resolveInFiveSeconds(): Promise<string> {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve('Success');
      }, 5000);
    });
  }

  function rejectInFiveSeconds() {
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        reject(new Error('Error'));
      }, 5000);
    });
  }

  it('resolves promise if not cancelled', async () => {
    const resolve = vi.fn();
    const reject = vi.fn();

    const { promise } = makeCancellablePromise(resolveInFiveSeconds());

    vi.advanceTimersByTime(5000);
    await promise.then(resolve).catch(reject);

    expect(resolve).toHaveBeenCalledWith('Success');
    expect(reject).not.toHaveBeenCalled();
  });

  it('rejects promise if not cancelled', async () => {
    const resolve = vi.fn();
    const reject = vi.fn();

    const { promise } = makeCancellablePromise(rejectInFiveSeconds());

    vi.runAllTimers();
    await promise.then(resolve).catch(reject);

    expect(resolve).not.toHaveBeenCalled();
    expect(reject).toHaveBeenCalledWith(expect.any(Error));
  });

  it('does not resolve promise if cancelled', async () => {
    expect.assertions(0);

    const resolve = vi.fn(() => {
      // Will fail because of expect.assertions(0);
      expect(true).toBe(true);
    });
    const reject = vi.fn(() => {
      // Will fail because of expect.assertions(0);
      expect(true).toBe(true);
    });

    const { promise, cancel } = makeCancellablePromise(rejectInFiveSeconds());
    promise.then(resolve).catch(reject);

    vi.advanceTimersByTime(2500);
    cancel();
    vi.advanceTimersByTime(2500);
  });

  it('does not reject promise if cancelled', () => {
    expect.assertions(0);

    const resolve = vi.fn(() => {
      // Will fail because of expect.assertions(0);
      expect(true).toBe(true);
    });
    const reject = vi.fn(() => {
      // Will fail because of expect.assertions(0);
      expect(true).toBe(true);
    });

    const { promise, cancel } = makeCancellablePromise(rejectInFiveSeconds());
    promise.then(resolve).catch(reject);

    vi.advanceTimersByTime(2500);
    cancel();
    vi.advanceTimersByTime(2500);
  });
});
