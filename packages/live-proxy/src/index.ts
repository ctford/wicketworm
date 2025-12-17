import { Hono } from 'hono';
import { cors } from 'hono/cors';
import type { GameState, BallEvent } from '@wicketworm/shared-types';

const app = new Hono();

// Enable CORS for browser clients
app.use('/*', cors());

// Health check
app.get('/', (c) => {
  return c.json({
    name: 'WicketWorm Live Proxy',
    version: '0.1.0',
    status: 'ok'
  });
});

// Get match state
app.get('/api/match/:id/state', async (c) => {
  const matchId = c.req.param('id');

  // TODO: Fetch from data provider
  // TODO: Normalize to GameState

  return c.json({
    error: 'Not implemented',
    message: 'Data provider integration pending'
  }, 501);
});

// Get ball events
app.get('/api/match/:id/balls', async (c) => {
  const matchId = c.req.param('id');
  const since = c.req.query('since');

  // TODO: Fetch from data provider
  // TODO: Normalize to BallEvent[]

  return c.json({
    error: 'Not implemented',
    message: 'Data provider integration pending'
  }, 501);
});

const port = process.env.PORT || 3000;

console.log(`🏏 WicketWorm Live Proxy running on http://localhost:${port}`);

export default {
  port,
  fetch: app.fetch
};
