// As defined on the list of supported events: https://reactjs.org/docs/events.html
export const clipboardEvents = ['onCopy', 'onCut', 'onPaste'] as const;
export const compositionEvents = [
  'onCompositionEnd',
  'onCompositionStart',
  'onCompositionUpdate',
] as const;
export const focusEvents = ['onFocus', 'onBlur'] as const;
export const formEvents = ['onInput', 'onInvalid', 'onReset', 'onSubmit'] as const;
export const imageEvents = ['onLoad', 'onError'] as const;
export const keyboardEvents = ['onKeyDown', 'onKeyPress', 'onKeyUp'] as const;
export const mediaEvents = [
  'onAbort',
  'onCanPlay',
  'onCanPlayThrough',
  'onDurationChange',
  'onEmptied',
  'onEncrypted',
  'onEnded',
  'onError',
  'onLoadedData',
  'onLoadedMetadata',
  'onLoadStart',
  'onPause',
  'onPlay',
  'onPlaying',
  'onProgress',
  'onRateChange',
  'onSeeked',
  'onSeeking',
  'onStalled',
  'onSuspend',
  'onTimeUpdate',
  'onVolumeChange',
  'onWaiting',
] as const;
export const mouseEvents = [
  'onClick',
  'onContextMenu',
  'onDoubleClick',
  'onMouseDown',
  'onMouseEnter',
  'onMouseLeave',
  'onMouseMove',
  'onMouseOut',
  'onMouseOver',
  'onMouseUp',
] as const;
export const dragEvents = [
  'onDrag',
  'onDragEnd',
  'onDragEnter',
  'onDragExit',
  'onDragLeave',
  'onDragOver',
  'onDragStart',
  'onDrop',
] as const;
export const selectionEvents = ['onSelect'] as const;
export const touchEvents = ['onTouchCancel', 'onTouchEnd', 'onTouchMove', 'onTouchStart'] as const;
export const pointerEvents = [
  'onPointerDown',
  'onPointerMove',
  'onPointerUp',
  'onPointerCancel',
  'onGotPointerCapture',
  'onLostPointerCapture',
  'onPointerEnter',
  'onPointerLeave',
  'onPointerOver',
  'onPointerOut',
] as const;
export const uiEvents = ['onScroll'] as const;
export const wheelEvents = ['onWheel'] as const;
export const animationEvents = [
  'onAnimationStart',
  'onAnimationEnd',
  'onAnimationIteration',
] as const;
export const transitionEvents = ['onTransitionEnd'] as const;
export const otherEvents = ['onToggle'] as const;
export const changeEvents = ['onChange'] as const;

export const allEvents = [
  ...clipboardEvents,
  ...compositionEvents,
  ...focusEvents,
  ...formEvents,
  ...imageEvents,
  ...keyboardEvents,
  ...mediaEvents,
  ...mouseEvents,
  ...dragEvents,
  ...selectionEvents,
  ...touchEvents,
  ...pointerEvents,
  ...uiEvents,
  ...wheelEvents,
  ...animationEvents,
  ...transitionEvents,
  ...changeEvents,
  ...otherEvents,
] as const;

type AllEvents = (typeof allEvents)[number];

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type EventHandler<ArgsType> = (event: any, args: ArgsType) => void;

// Creates inferred type for event handler without args.
type EventHandlerWithoutArgs<ArgsType, OriginalEventHandler> = OriginalEventHandler extends (
  event: infer Event,
  args: ArgsType,
) => void
  ? (event: Event) => void
  : never;

export type EventProps<ArgsType> = {
  [K in AllEvents]?: EventHandler<ArgsType>;
};

type Props<ArgsType> = Record<string, unknown> & EventProps<ArgsType>;

type EventPropsWithoutArgs<ArgsType, PropsType> = {
  [K in keyof PropsType as K extends AllEvents ? K : never]: EventHandlerWithoutArgs<
    ArgsType,
    PropsType[K]
  >;
};

/**
 * Returns an object with on-event callback props curried with provided args.
 * @param {Object} props Props passed to a component.
 * @param {Function=} getArgs A function that returns argument(s) on-event callbacks
 *   shall be curried with.
 */
export default function makeEventProps<
  ArgsType,
  PropsType extends Props<ArgsType> = Props<ArgsType>,
>(
  props: PropsType,
  getArgs?: (eventName: string) => ArgsType,
): EventPropsWithoutArgs<ArgsType, PropsType> {
  const eventProps: EventPropsWithoutArgs<ArgsType, PropsType> = {} as EventPropsWithoutArgs<
    ArgsType,
    PropsType
  >;

  allEvents.forEach((eventName) => {
    type EventHandlerType = EventPropsWithoutArgs<ArgsType, PropsType>[typeof eventName];

    const eventHandler = props[eventName];

    if (!eventHandler) {
      return;
    }

    if (getArgs) {
      eventProps[eventName] = ((event) =>
        eventHandler(event, getArgs(eventName))) as EventHandlerType;
    } else {
      eventProps[eventName] = eventHandler as EventHandlerType;
    }
  });

  return eventProps;
}
