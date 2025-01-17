export default function makeCancellablePromise<T>(promise: Promise<T>) {
  let isCancelled = false;

  const wrappedPromise: typeof promise = new Promise((resolve, reject) => {
    promise
      .then((value) => !isCancelled && resolve(value))
      .catch((error) => !isCancelled && reject(error));
  });

  return {
    promise: wrappedPromise,
    cancel() {
      isCancelled = true;
    },
  };
}
