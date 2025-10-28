// // helper.ts

// // Calculate pool floor area
// export function calculateArea(length: number, width: number): number {
//   return length * width;
// }

// // Calculate walls area (using average depth)
// export function calculateWalls(length: number, averageDepth: number): number {
//   return length * averageDepth * 2; // two long walls
// }

// // Shallow wall area
// export function shallowArea(width: number, shallowDepth: number): number {
//   return width * shallowDepth;
// }

// // Deepest wall area
// export function deepestArea(width: number, deepestDepth: number): number {
//   return width * deepestDepth;
// }

// // Convert tile dimensions (mm) to m²
// export function tileArea(length: number, width: number): number {
//   return length * 0.001 * (width * 0.001);
// }

// // Coping/band area
// export function copeArea(length: number, width: number): number {
//   return length * width;
// }

// // Total surface area calculation
// export function totalSurfaceArea(
//   poolArea: number,
//   wallsArea: number,
//   shallow: number,
//   deepest: number,
//   cope: number
// ): number {
//   return poolArea + wallsArea + shallow + deepest + cope;
// }

// // Calculate number of tiles needed
// export function numberOfTiles(
//   totalArea: number,
//   singleTileArea: number
// ): number {
//   if (singleTileArea <= 0) return 0; // prevent division by zero
//   return Math.ceil(totalArea / singleTileArea);
// }

// // Calculate total tiles including margin
// export function subtotalMargin(
//   numberOfTiles: number,
//   marginPercent: number
// ): number {
//   return Math.ceil(numberOfTiles * (1 + marginPercent / 100));
// }
