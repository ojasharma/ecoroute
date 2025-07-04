// types/route.ts

export type Coordinate = [number, number];

export interface RouteRequest {
    from: Coordinate;
    to: Coordinate;
}

export interface RouteResponse {
    path: Coordinate[];
}
