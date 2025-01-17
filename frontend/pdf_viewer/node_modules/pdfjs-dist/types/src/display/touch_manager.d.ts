export class TouchManager {
    constructor({ container, isPinchingDisabled, isPinchingStopped, onPinchStart, onPinching, onPinchEnd, signal, }: {
        container: any;
        isPinchingDisabled?: null | undefined;
        isPinchingStopped?: null | undefined;
        onPinchStart?: null | undefined;
        onPinching?: null | undefined;
        onPinchEnd?: null | undefined;
        signal: any;
    });
    get MIN_TOUCH_DISTANCE_TO_PINCH(): any;
    destroy(): void;
    #private;
}
