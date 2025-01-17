# path2d

[![CI](https://github.com/nilzona/path2d-polyfill/actions/workflows/ci.yml/badge.svg)](https://github.com/nilzona/path2d-polyfill/actions/workflows/ci.yml)

Implements `Path2D` api and `roundRect` for CanvasRenderingContext2D

## Usage

```shell
npm install --save path2d
```

## Use in a node environment

The package exports a few functions that can be used in a node environment:

- `Path2D` - class to create Path2D objects used by the polyfill methods
- `parsePath` - function for parsing an SVG path string into canvas commands
- 'roundRect' - implementation of roundRect using canvas commands
- `applyPath2DToCanvasRenderingContext` - Adds Path2D functions (if needed) to a CanvasRenderingContext and augments the fill, stroke and clip command
- `applyRoundRectToCanvasRenderingContext2D` - Adds roundRect function (if needed) to a CanvasRenderingContext

```js
import { Path2D } from "path2d";
```

### usage with node-canvas

To get Path2D features with the [node-canvas library](https://github.com/Automattic/node-canvas) use the following pattern:

```js
const { createCanvas, CanvasRenderingContext2D } = require("canvas");
const { applyPath2DToCanvasRenderingContext, Path2D } = require("path2d");

applyPath2DToCanvasRenderingContext(CanvasRenderingContext2D);
// Path2D features has now been added to CanvasRenderingContext2D

const canvas = createCanvas(200, 200);
const ctx = canvas.getContext("2d");

const p = new Path2D("M10 10 l 20 0 l 0 20 Z");
ctx.fillStyle = "green";
ctx.fill(p);
```

A working example of a node express server that serves an image drawn with canvas can be seen [here](https://gist.github.com/nilzona/e611c99336d8ea1f645bd391a459c24f)

## Support table

| Method               | Supported |
| -------------------- | :-------: |
| constructor(SVGPath) |    Yes    |
| addPath()            |    Yes    |
| closePath()          |    Yes    |
| moveTo()             |    Yes    |
| lineTo()             |    Yes    |
| bezierCurveTo()      |    Yes    |
| quadraticCurveTo()   |    Yes    |
| arc()                |    Yes    |
| ellipse()            |    Yes    |
| rect()               |    Yes    |
| roundRect()          |    Yes    |

## See it in action

Clone [path2d-polyfill](https://github.com/nilzona/path2d-polyfill)

```shell
pnpm install
pnpm dev
```

open <http://localhost:5173/> to see the example page.

## Contributing

Recommended to use vscode with the prettier extension to keep formatting intact.
