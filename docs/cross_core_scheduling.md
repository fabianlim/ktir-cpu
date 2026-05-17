# Cross-Core Scheduling — Roles & Boundaries

## One scheduler path

There is exactly one way cores communicate during execution:

```
client code           ctx.send_to(dst, tile)        — fire-and-forget
client code (yields)  RecvRequest(src)              — blocks until tile arrives
GridExecutor          owns the message queue and drives every core's generator
```

Everything else is either above this path (clients) or off it entirely
(remote-LX peeks).

---

## Architecture

```
KTIRInterpreter
  └── GridExecutor.execute_with_communication(ops, input_ptrs, execute_op)
        │
        ├── CoreExecutionStack (one per core)   ← only generator-aware layer
        │     ├── _execute_until_block(ops, execute_op)   generator
        │     │     for each op:
        │     │       result = execute_op(op, core)
        │     │       if isgenerator(result):
        │     │           result = yield from result   ← bubbles RecvRequests up
        │     │           _store(op, result, core)
        │     └── resume(send_val) / is_blocked() / waiting_on
        │
        ├── CoreContext (one per core)
        │     ├── SSA value map (_scope_stack)
        │     ├── LX scratchpad + usage tracking
        │     ├── send_to(dst, tile)   ← wired to scheduler _enqueue before run
        │     └── get_lx(core_id)      ← returns local LX or delegates to ring_backend
        │
        ├── RingBackend (injected into CoreContext at construction)
        │     └── DirectLXBackend [current]
        │           get_lx(core_id) → direct lookup into SpyreMemoryHierarchy
        │           used for pre-seeded distributed views
        │
        └── Scheduler loop
              messages: {(src,dst): deque[Tile]}
              waiting:  {core_id: src_core}
              while stacks:
                  if no _try_deliver succeeds → RuntimeError("Deadlock detected")
```

---

## Layers

| Layer | Responsibility | Notes |
|---|---|---|
| **`CommOps`** | High-level **per-core** comm primitives. Each call describes one core's view of the algorithm — its sends, its recvs, its result. Today: `reduce` (ring) and `transfer`. | Stable surface. Dialect handlers and tests should call into here, not into scheduler primitives directly. |
| **`CoreContext`** | Per-core SSA scope, LX scratchpad, and `send_to`. `send_to` is wired by `GridExecutor` to enqueue into the scheduler's message queue. | |
| **`CoreExecutionStack`** | Wraps one core's op list as a generator. The only place that calls `.send()` / `next()` on client generators. Bubbles `RecvRequest` to the scheduler via `yield from`. | Internal. |
| **`GridExecutor`** | Owns cores, the message queue, and the scheduler loop. Drives every core's generator to completion or raises on deadlock. | Internal. |
| **`RingBackend.get_lx`** | Remote LX peek for cross-core memory views. Off the comm path — returns a scratchpad reference, no messages, no scheduling. | Only `get_lx` is live (see "Marked for deletion" below). |

---

## Per-core semantics

Every `CommOps` primitive describes **one core's behavior**. The function
runs once per participating core; the scheduler runs N copies concurrently;
they cooperate via the message queue.

### `CommOps.reduce` — ring reduction

Implements ring reduction from the perspective of one core. Each round:
send to the next neighbor, recv from the previous neighbor, accumulate.
After N-1 rounds, each core holds the full reduction.

```
my_idx   = group.index(ctx.core_id)
result   = tile
for _ in range(len(group) - 1):
    ctx.send_to(group[(my_idx + 1) % n], result)
    received = yield RecvRequest(src=group[(my_idx - 1) % n])
    result   = fn(result, received)
return result
```

Yields **N-1 `RecvRequest`s** for the calling core (one per ring hop).
Every participating core runs its own copy. Cores not in `group` simply
don't call `CommOps.reduce` — they're not participants, no special case.

### `CommOps.transfer` — multi-destination send

Plain function, no yields. Calls `ctx.send_to(d, tile)` for each `d` in
`dst_cores`. The receiving cores must call into `CommOps` (or a future
recv primitive) to consume the tile.

---

## Generator vs plain function — the rule

A `CommOps` primitive is a generator iff it can block.

- Generator: contains `yield RecvRequest(...)`. Calling it returns a
  generator object; the body has not yet run. The scheduler drives it.
- Plain function: no `yield`. Calling it runs the body to completion.

| Primitive | Shape |
|---|---|
| `CommOps.reduce` | generator (yields N-1 times for the calling core) |
| `CommOps.transfer` | plain |

A handler that calls a generator primitive must **propagate** the
generator — don't call it and discard the result:

```python
return CommOps.reduce(ctx, tile, group, fn)              # OK — scheduler drives it
return (yield from CommOps.reduce(ctx, tile, group, fn)) # OK — handler is itself a generator
CommOps.reduce(ctx, tile, group, fn)                     # BUG — generator created, body never runs
```

**Key invariant:** `send_to` is fire-and-forget — the sender never blocks.
Only `yield RecvRequest` suspends a core. This prevents sender-side
deadlock in symmetric patterns (all-to-all, ring).

---

## Control flow: blocking recv (end-to-end)

```
dialect handler / test handler
  └── CommOps.reduce(ctx, tile, group, fn)        ← generator object
        for each round:
          ctx.send_to(next_core, result)          ← enqueues immediately
          received = yield RecvRequest(src=prev)  ← suspends here
          result   = fn(result, received)
        return result

CoreExecutionStack._execute_until_block
  result = execute_op(op, core)                   ← gets the generator
  result = yield from result                      ← forwards RecvRequest up

GridExecutor.execute_with_communication
  _advance(core_id):
    request = next(stack._gen)
    if isinstance(request, RecvRequest):
        waiting[core_id] = request.src            ← park
    else:
        results[core_id] = done

  _try_deliver(core_id):
    tile = _pop(src, core_id)
    if tile: del waiting[core_id]; _advance(core_id, tile)

  while stacks:
    if not any(_try_deliver(c) for c in stacks):
        raise RuntimeError("Deadlock detected: ...")
```

---

## Deadlock detection

After each scheduler pass, if every active core is parked in `waiting`
and no message arrived to unblock anyone:

```
RuntimeError("Deadlock detected: core 0 waiting on recv from core 1; ...")
```

This catches both flat deadlocks (mutual recv with no sends) and
loop-induced deadlocks (one side exhausts its sends before the other
finishes recving).

---

## Marked for deletion

These exist from an earlier iteration of the comm design and are **not on
the live scheduler path**. They will be removed once the scheduler test
scaffold lands.

| Symbol | Location | Why dead |
|---|---|---|
| `RingBackend.send` / `RingBackend.recv` | `ktir_cpu/ops/comm_ops.py` | No caller on the scheduler path. `CommOps` and `CoreContext.send_to` do not use them. |
| `RingNetwork` | `ktir_cpu/ops/comm_ops.py` | Parallel message-buffer that the scheduler doesn't read from or write to. |
| `DirectLXBackend.send` / `DirectLXBackend.recv` | `ktir_cpu/ops/comm_ops.py` | Pass-through to `RingNetwork`; same reason. |

`RingBackend.get_lx` and `DirectLXBackend.get_lx` are live and stay —
they support remote LX peeks for distributed memory views, which is a
separate concern from the comm path.

The future `RingTransferBackend` (mentioned in the docstring of
`RingBackend` in `comm_ops.py`) is not a peer of the current backends;
when built, it will be a client of the scheduler protocol just like
`CommOps` is. It does not require new infrastructure.

---

## Where things are defined

| Symbol | File |
|---|---|
| `RecvRequest` | `ktir_cpu/grid.py` (lives here, not `comm_ops.py`, to avoid circular import) |
| `CoreContext.send_to`, `CoreContext.get_lx` | `ktir_cpu/grid.py` |
| `CoreExecutionStack` | `ktir_cpu/grid.py` |
| `GridExecutor.execute_with_communication` | `ktir_cpu/grid.py` |
| `CommOps.reduce`, `CommOps.transfer` | `ktir_cpu/ops/comm_ops.py` |
| `RingBackend`, `DirectLXBackend` | `ktir_cpu/ops/comm_ops.py` |
