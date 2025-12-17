# WicketWorm — Claude Implementation Plan

You are building **WicketWorm**, an OSS HTML tool that visualises **Win / Draw / Loss probabilities** for a **live Test match** as a *worm chart* over **overs**, using only **anonymised scorecard state** (no team strength).

Follow this plan strictly. Prefer simple, explainable solutions.

---

## 0. Constraints & principles

- Test matches only  
- Team-agnostic, venue-agnostic  
- X-axis is **overs**, not time  
- Probabilities must sum to 1 at every point  
- Model is trained offline, runs in the browser  
- UI must work in **offline replay mode first**  
- Everything is OSS-friendly  

---

## 1. Repository layout

Create a monorepo:

```
wicketworm/
  packages/
    model-train/        # Python: train + export model.json
    live-proxy/         # Node/Bun: live match data normalisation
    ui/                 # Vite + TS + D3: HTML worm
  README.md
  LICENSE (MIT)
```

Use `pnpm` workspaces.

---

## 2. Canonical data model (shared across packages)

### BallEvent
```ts
type BallEvent = {
  innings: 1 | 2 | 3 | 4;
  over: number;
  ball: number;
  runs: number;
  wickets: 0 | 1;
  extras?: number;
  kind?: string;
};
```

### GameState
```ts
type GameState = {
  matchId: string;
  innings: 1 | 2 | 3 | 4;
  runsFor: number;
  wicketsDown: number;
  ballsBowled: number;
  lead: number;
  target?: number;
  matchOversLimit: number;
  ballsRemaining: number;
  completedInnings: number;
  isChasing: boolean;
};
```

### ProbPoint
```ts
type ProbPoint = {
  xOver: number;
  innings: 1 | 2 | 3 | 4;
  over: number;
  pWin: number;
  pDraw: number;
  pLoss: number;
};
```

### Annotation
```ts
type Annotation = {
  xOver: number;
  kind: "wicket" | "declare" | "inningsEnd" | "leadChange" | "milestone";
  label?: string;
};
```

---

## 3. X-axis definition

```ts
const OVER_OFFSET = 500;
xOver = (innings - 1) * OVER_OFFSET + over;
```

---

## 4. Model training (Python)

- Use Cricsheet Test data
- Aggregate per over
- Train multinomial logistic regression
- Export model.json with coefficients, means, stds

---

## 5. Browser inference

- Load model.json
- Compute features
- Apply softmax
- Output pWin / pDraw / pLoss

---

## 6. Live proxy

Endpoints:

```
GET /api/match/:id/state
GET /api/match/:id/balls?since=<cursor>
```

Normalise provider payloads into canonical types.

---

## 7. UI

- Vite + TypeScript + D3
- Poll per over
- Render stacked worm
- Overlay annotations

---

## 8. Offline replay mode

Must work without live data.

---

## 9. Milestones

1. Offline replay works
2. Model validated in browser
3. Live proxy connected
4. Live worm updates
5. Annotations
6. Docs polish

---

## Success

You can watch a live Ashes Test and see the W/L/D worm evolve over overs.
