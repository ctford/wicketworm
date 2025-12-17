# live-proxy

Node.js/Bun server that normalizes live match data from various providers into canonical WicketWorm types.

## Features

- Provider-agnostic data normalization
- Per-over caching
- Rate limiting
- CORS support for browser clients

## API

### Get match state
```
GET /api/match/:id/state
```

Returns current `GameState`

### Get ball events
```
GET /api/match/:id/balls?since=<cursor>
```

Returns `BallEvent[]` since cursor

## Data providers

### Cricsheet
- Recent Test match data
- Free, OSS-friendly

### ESPN Cricinfo (planned)
- Live match data
- Requires rate limiting

## Setup

```bash
pnpm install
pnpm dev
```

Server runs on `http://localhost:3000`

## Environment variables

```
PORT=3000
CRICINFO_API_KEY=  # Optional, for ESPN Cricinfo
```
